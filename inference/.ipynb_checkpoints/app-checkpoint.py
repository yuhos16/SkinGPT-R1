# app.py
import uvicorn
import os
import shutil
import uuid
import json
import re
import asyncio
from typing import Optional
from io import BytesIO
from contextlib import asynccontextmanager
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from model_utils import SkinGPTModel
from deepseek_service import get_deepseek_service, DeepSeekService

# === Configuration ===
MODEL_PATH = "../checkpoint"
TEMP_DIR = "./temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# DeepSeek API Key
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-b221f29be052460f9e0fe12d88dd343c")

# Global DeepSeek service instance
deepseek_service: Optional[DeepSeekService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化 DeepSeek 服务
    await init_deepseek()
    yield
    print("\nShutting down service...")

app = FastAPI(
    title="SkinGPT-R1 皮肤诊断系统",
    description="智能皮肤诊断助手",
    version="1.0.0",
    lifespan=lifespan
)

# CORS配置 - 允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量存储状态
# chat_states: 存储对话历史 (List of messages for Qwen)
# pending_images: 存储已上传但尚未发送给LLM的图片路径 (State ID -> Image Path)
chat_states = {} 
pending_images = {}

def parse_diagnosis_result(raw_text: str) -> dict:
    """
    解析诊断结果中的think和answer标签
    
    参数:
    - raw_text: 原始诊断文本
    
    返回:
    - dict: 包含thinking, answer, raw字段的字典
    """
    import re
    
    # 尝试匹配完整的标签
    think_match = re.search(r'<think>([\s\S]*?)</think>', raw_text)
    answer_match = re.search(r'<answer>([\s\S]*?)</answer>', raw_text)
    
    thinking = None
    answer = None
    
    # 处理think标签
    if think_match:
        thinking = think_match.group(1).strip()
    else:
        # 尝试匹配未闭合的think标签（输出被截断的情况）
        unclosed_think = re.search(r'<think>([\s\S]*?)(?=<answer>|$)', raw_text)
        if unclosed_think:
            thinking = unclosed_think.group(1).strip()
    
    # 处理answer标签
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        # 尝试匹配未闭合的answer标签
        unclosed_answer = re.search(r'<answer>([\s\S]*?)$', raw_text)
        if unclosed_answer:
            answer = unclosed_answer.group(1).strip()
    
    # 如果仍然没有找到answer，清理原始文本作为answer
    if not answer:
        # 移除所有标签及其内容
        cleaned = re.sub(r'<think>[\s\S]*?</think>', '', raw_text)
        cleaned = re.sub(r'<think>[\s\S]*', '', cleaned)  # 移除未闭合的think
        cleaned = re.sub(r'</?answer>', '', cleaned)  # 移除answer标签
        cleaned = cleaned.strip()
        answer = cleaned if cleaned else raw_text
    
    # 清理可能残留的标签
    if answer:
        answer = re.sub(r'</?think>|</?answer>', '', answer).strip()
    if thinking:
        thinking = re.sub(r'</?think>|</?answer>', '', thinking).strip()
    
    # 处理 "Final Answer:" 格式，提取其后的内容
    if answer:
        final_answer_match = re.search(r'Final Answer:\s*([\s\S]*)', answer, re.IGNORECASE)
        if final_answer_match:
            answer = final_answer_match.group(1).strip()
    
    return {
        "thinking": thinking if thinking else None,
        "answer": answer,
        "raw": raw_text
    }

print("Initializing Model Service...")
# 全局加载模型
gpt_model = SkinGPTModel(MODEL_PATH)
print("Service Ready.")

# 初始化 DeepSeek 服务（异步）
async def init_deepseek():
    global deepseek_service
    print("\nInitializing DeepSeek service...")
    deepseek_service = await get_deepseek_service(api_key=DEEPSEEK_API_KEY)
    if deepseek_service and deepseek_service.is_loaded:
        print("DeepSeek service is ready!")
    else:
        print("DeepSeek service not available, will return raw results")

@app.post("/v1/upload/{state_id}")
async def upload_file(state_id: str, file: UploadFile = File(...), survey: str = Form(None)):
    """
    接收图片上传。
    逻辑：将图片保存到本地临时目录，并标记该 state_id 有一张待处理图片。
    """
    try:
        # 1. 保存图片到本地临时文件
        file_extension = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        unique_name = f"{state_id}_{uuid.uuid4().hex}.{file_extension}"
        file_path = os.path.join(TEMP_DIR, unique_name)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. 记录图片路径等待下一次 predict 调用时使用
        # 如果是多图模式，这里可以改成 list，目前演示单图覆盖或更新
        pending_images[state_id] = file_path
        
        # 3. 初始化对话状态（如果是新会话）
        if state_id not in chat_states:
            chat_states[state_id] = []
            
        return {"message": "Image uploaded successfully", "path": file_path}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/v1/predict/{state_id}")
async def v1_predict(request: Request, state_id: str):
    """
    接收文本并执行推理。
    逻辑：检查是否有待处理图片。如果有，将其与文本组合成 multimodal 消息。
    """
    try:
        data = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")
        
    user_message = data.get("message", "")
    if not user_message:
        raise HTTPException(status_code=400, detail="Missing 'message' field")

    # 获取或初始化历史
    history = chat_states.get(state_id, [])
    
    # 构建当前轮次的用户内容
    current_content = []
    
    # 1. 检查是否有刚刚上传的图片
    if state_id in pending_images:
        img_path = pending_images.pop(state_id) # 取出并移除
        current_content.append({"type": "image", "image": img_path})
        
        # 如果是第一次对话，加上 System Prompt
        if not history:
             system_prompt = "You are a professional AI dermatology assistant. "
             user_message = f"{system_prompt}\n\n{user_message}"

    # 2. 添加文本
    current_content.append({"type": "text", "text": user_message})
    
    # 3. 更新历史
    history.append({"role": "user", "content": current_content})
    chat_states[state_id] = history

    # 4. 运行推理 (在线程池中运行以防阻塞)
    try:
        response_text = await run_in_threadpool(
            gpt_model.generate_response, 
            messages=history
        )
    except Exception as e:
        # 回滚历史（移除刚才出错的用户提问）
        chat_states[state_id].pop()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # 5. 将回复加入历史
    history.append({"role": "assistant", "content": [{"type": "text", "text": response_text}]})
    chat_states[state_id] = history

    return {"message": response_text}

@app.post("/v1/reset/{state_id}")
async def reset_chat(state_id: str):
    """清除会话状态"""
    if state_id in chat_states:
        del chat_states[state_id]
    if state_id in pending_images:
        # 可选：删除临时文件
        try:
            os.remove(pending_images[state_id])
        except:
            pass
        del pending_images[state_id]
    return {"message": "Chat history reset"}

@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "SkinGPT-R1 皮肤诊断系统",
        "version": "1.0.0",
        "status": "running",
        "description": "智能皮肤诊断助手"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": True
    }

@app.post("/diagnose/stream")
async def diagnose_stream(
    image: Optional[UploadFile] = File(None),
    text: str = Form(...),
    language: str = Form("zh"),
):
    """
    SSE流式诊断接口（用于前端）
    支持图片上传和文本输入，返回真正的流式响应
    使用 DeepSeek API 优化输出格式
    """
    from queue import Queue, Empty
    from threading import Thread
    
    language = language if language in ("zh", "en") else "zh"
    
    # 处理图片
    pil_image = None
    temp_image_path = None
    
    if image:
        contents = await image.read()
        pil_image = Image.open(BytesIO(contents)).convert("RGB")
    
    # 创建队列用于线程间通信
    result_queue = Queue()
    # 用于存储完整响应和解析结果
    generation_result = {"full_response": [], "parsed": None, "temp_image_path": None}
    
    def run_generation():
        """在后台线程中运行流式生成"""
        full_response = []
        
        try:
            # 构建消息
            messages = []
            current_content = []
            
            # 添加系统提示
            system_prompt = "You are a professional AI dermatology assistant." if language == "en" else "你是一个专业的AI皮肤科助手。"
            
            # 如果有图片，保存到临时文件
            if pil_image:
                generation_result["temp_image_path"] = os.path.join(TEMP_DIR, f"temp_{uuid.uuid4().hex}.jpg")
                pil_image.save(generation_result["temp_image_path"])
                current_content.append({"type": "image", "image": generation_result["temp_image_path"]})
            
            # 添加文本
            prompt = f"{system_prompt}\n\n{text}"
            current_content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": current_content})
            
            # 流式生成 - 每个 chunk 立即放入队列
            for chunk in gpt_model.generate_response_stream(
                messages=messages,
                max_new_tokens=2048,
                temperature=0.7
            ):
                full_response.append(chunk)
                result_queue.put(("delta", chunk))
            
            # 解析结果
            response_text = "".join(full_response)
            parsed = parse_diagnosis_result(response_text)
            generation_result["full_response"] = full_response
            generation_result["parsed"] = parsed
            
            # 标记生成完成
            result_queue.put(("generation_done", None))
            
        except Exception as e:
            result_queue.put(("error", str(e)))
    
    async def event_generator():
        """异步生成SSE事件"""
        # 在后台线程启动生成（非阻塞）
        gen_thread = Thread(target=run_generation)
        gen_thread.start()
        
        loop = asyncio.get_event_loop()
        
        # 从队列中读取并发送流式内容
        while True:
            try:
                # 非阻塞获取
                msg_type, data = await loop.run_in_executor(
                    None, 
                    lambda: result_queue.get(timeout=0.1)
                )
                
                if msg_type == "generation_done":
                    # 流式生成完成，准备处理最终结果
                    break
                elif msg_type == "delta":
                    yield_chunk = json.dumps({"type": "delta", "text": data}, ensure_ascii=False)
                    yield f"data: {yield_chunk}\n\n"
                elif msg_type == "error":
                    yield f"data: {json.dumps({'type': 'error', 'message': data}, ensure_ascii=False)}\n\n"
                    gen_thread.join()
                    return
                    
            except Empty:
                # 队列暂时为空，继续等待
                await asyncio.sleep(0.01)
                continue
        
        gen_thread.join()
        
        # 获取解析结果
        parsed = generation_result["parsed"]
        if not parsed:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Failed to parse response'}, ensure_ascii=False)}\n\n"
            return
        
        raw_thinking = parsed["thinking"]
        raw_answer = parsed["answer"]
        
        # 使用 DeepSeek 优化结果
        refined_by_deepseek = False
        description = None
        thinking = raw_thinking
        answer = raw_answer
        
        if deepseek_service and deepseek_service.is_loaded:
            try:
                print(f"Calling DeepSeek to refine diagnosis (language={language})...")
                refined = await deepseek_service.refine_diagnosis(
                    raw_answer=raw_answer,
                    raw_thinking=raw_thinking,
                    language=language,
                )
                if refined["success"]:
                    description = refined["description"]
                    thinking = refined["analysis_process"]
                    answer = refined["diagnosis_result"]
                    refined_by_deepseek = True
                    print(f"DeepSeek refinement completed successfully")
            except Exception as e:
                print(f"DeepSeek refinement failed, using original: {e}")
        else:
            print("DeepSeek service not available, using raw results")
        
        success_msg = "Diagnosis completed" if language == "en" else "诊断完成"
        
        # 返回格式与参考项目保持一致
        final_payload = {
            "description": description,              # 图片描述（从 thinking 中提取）
            "thinking": thinking,                    # 分析过程（DeepSeek 优化后）
            "answer": answer,                        # 诊断结果（DeepSeek 优化后）
            "raw": parsed["raw"],                    # 原始响应
            "refined_by_deepseek": refined_by_deepseek,  # 是否被 DeepSeek 优化
            "success": True,
            "message": success_msg
        }
        yield_final = json.dumps({"type": "final", "result": final_payload}, ensure_ascii=False)
        yield f"data: {yield_final}\n\n"
        
        # 清理临时图片
        temp_path = generation_result.get("temp_image_path")
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=5900, reload=False)