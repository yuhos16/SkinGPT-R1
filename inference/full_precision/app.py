from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

try:
    from .deepseek_service import DeepSeekService, get_deepseek_service
    from .model_utils import DEFAULT_MODEL_PATH, SkinGPTModel, resolve_model_path
except ImportError:
    from deepseek_service import DeepSeekService, get_deepseek_service
    from model_utils import DEFAULT_MODEL_PATH, SkinGPTModel, resolve_model_path

MODEL_PATH = resolve_model_path(DEFAULT_MODEL_PATH)
TEMP_DIR = Path(__file__).resolve().parents[1] / "temp_uploads"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

deepseek_service: Optional[DeepSeekService] = None


def parse_diagnosis_result(raw_text: str) -> dict:
    import re

    think_match = re.search(r"<think>([\s\S]*?)</think>", raw_text)
    answer_match = re.search(r"<answer>([\s\S]*?)</answer>", raw_text)

    thinking = think_match.group(1).strip() if think_match else None
    answer = answer_match.group(1).strip() if answer_match else None

    if not thinking:
        unclosed_think = re.search(r"<think>([\s\S]*?)(?=<answer>|$)", raw_text)
        if unclosed_think:
            thinking = unclosed_think.group(1).strip()

    if not answer:
        unclosed_answer = re.search(r"<answer>([\s\S]*?)$", raw_text)
        if unclosed_answer:
            answer = unclosed_answer.group(1).strip()

    if not answer:
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw_text)
        cleaned = re.sub(r"<think>[\s\S]*", "", cleaned)
        cleaned = re.sub(r"</?answer>", "", cleaned)
        answer = cleaned.strip() or raw_text

    if answer:
        answer = re.sub(r"</?think>|</?answer>", "", answer).strip()
        final_answer_match = re.search(r"Final Answer:\s*([\s\S]*)", answer, re.IGNORECASE)
        if final_answer_match:
            answer = final_answer_match.group(1).strip()

    if thinking:
        thinking = re.sub(r"</?think>|</?answer>", "", thinking).strip()

    return {"thinking": thinking or None, "answer": answer, "raw": raw_text}


print("Initializing Model Service...")
gpt_model = SkinGPTModel(MODEL_PATH)
print("Service Ready.")


async def init_deepseek():
    global deepseek_service
    print("\nInitializing DeepSeek service...")
    deepseek_service = await get_deepseek_service(api_key=DEEPSEEK_API_KEY)
    if deepseek_service and deepseek_service.is_loaded:
        print("DeepSeek service is ready!")
    else:
        print("DeepSeek service not available, will return raw results")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_deepseek()
    yield
    print("\nShutting down service...")


app = FastAPI(
    title="SkinGPT-R1 Full Precision API",
    description="Full-precision dermatology assistant backend",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_states = {}
pending_images = {}


@app.post("/v1/upload/{state_id}")
async def upload_file(state_id: str, file: UploadFile = File(...), survey: str = Form(None)):
    del survey
    try:
        file_extension = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        unique_name = f"{state_id}_{uuid.uuid4().hex}.{file_extension}"
        file_path = TEMP_DIR / unique_name

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        pending_images[state_id] = str(file_path)

        if state_id not in chat_states:
            chat_states[state_id] = []

        return {"message": "Image uploaded successfully", "path": str(file_path)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc


@app.post("/v1/predict/{state_id}")
async def v1_predict(request: Request, state_id: str):
    try:
        data = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    user_message = data.get("message", "")
    if not user_message:
        raise HTTPException(status_code=400, detail="Missing 'message' field")

    history = chat_states.get(state_id, [])
    current_content = []

    if state_id in pending_images:
        img_path = pending_images.pop(state_id)
        current_content.append({"type": "image", "image": img_path})
        if not history:
            user_message = f"You are a professional AI dermatology assistant.\n\n{user_message}"

    current_content.append({"type": "text", "text": user_message})
    history.append({"role": "user", "content": current_content})
    chat_states[state_id] = history

    try:
        response_text = await run_in_threadpool(gpt_model.generate_response, messages=history)
    except Exception as exc:
        chat_states[state_id].pop()
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    history.append({"role": "assistant", "content": [{"type": "text", "text": response_text}]})
    chat_states[state_id] = history
    return {"message": response_text}


@app.post("/v1/reset/{state_id}")
async def reset_chat(state_id: str):
    if state_id in chat_states:
        del chat_states[state_id]
    if state_id in pending_images:
        try:
            Path(pending_images[state_id]).unlink(missing_ok=True)
        except Exception:
            pass
        del pending_images[state_id]
    return {"message": "Chat history reset"}


@app.get("/")
async def root():
    return {
        "name": "SkinGPT-R1 Full Precision API",
        "version": "1.1.0",
        "status": "running",
        "description": "Full-precision dermatology assistant",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}


@app.post("/diagnose/stream")
async def diagnose_stream(
    image: Optional[UploadFile] = File(None),
    text: str = Form(...),
    language: str = Form("zh"),
):
    language = language if language in ("zh", "en") else "zh"
    pil_image = None

    if image:
        contents = await image.read()
        pil_image = Image.open(BytesIO(contents)).convert("RGB")

    result_queue = Queue()
    generation_result = {"full_response": [], "parsed": None, "temp_image_path": None}

    def run_generation():
        full_response = []
        try:
            messages = []
            current_content = []
            system_prompt = (
                "You are a professional AI dermatology assistant."
                if language == "en"
                else "你是一个专业的AI皮肤科助手。"
            )

            if pil_image:
                temp_image_path = TEMP_DIR / f"temp_{uuid.uuid4().hex}.jpg"
                pil_image.save(temp_image_path)
                generation_result["temp_image_path"] = str(temp_image_path)
                current_content.append({"type": "image", "image": str(temp_image_path)})

            current_content.append({"type": "text", "text": f"{system_prompt}\n\n{text}"})
            messages.append({"role": "user", "content": current_content})

            for chunk in gpt_model.generate_response_stream(
                messages=messages,
                max_new_tokens=2048,
                temperature=0.7,
            ):
                full_response.append(chunk)
                result_queue.put(("delta", chunk))

            response_text = "".join(full_response)
            generation_result["full_response"] = full_response
            generation_result["parsed"] = parse_diagnosis_result(response_text)
            result_queue.put(("generation_done", None))
        except Exception as exc:
            result_queue.put(("error", str(exc)))

    async def event_generator():
        gen_thread = Thread(target=run_generation)
        gen_thread.start()

        loop = asyncio.get_event_loop()
        while True:
            try:
                msg_type, data = await loop.run_in_executor(
                    None,
                    lambda: result_queue.get(timeout=0.1),
                )
                if msg_type == "generation_done":
                    break
                if msg_type == "delta":
                    yield f"data: {json.dumps({'type': 'delta', 'text': data}, ensure_ascii=False)}\n\n"
                elif msg_type == "error":
                    yield f"data: {json.dumps({'type': 'error', 'message': data}, ensure_ascii=False)}\n\n"
                    gen_thread.join()
                    return
            except Empty:
                await asyncio.sleep(0.01)

        gen_thread.join()

        parsed = generation_result["parsed"]
        if not parsed:
            yield "data: {\"type\": \"error\", \"message\": \"Failed to parse response\"}\n\n"
            return

        raw_thinking = parsed["thinking"]
        raw_answer = parsed["answer"]
        refined_by_deepseek = False
        description = None
        thinking = raw_thinking
        answer = raw_answer

        if deepseek_service and deepseek_service.is_loaded:
            try:
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
            except Exception as exc:
                print(f"DeepSeek refinement failed, using original: {exc}")
        else:
            print("DeepSeek service not available, using raw results")

        final_payload = {
            "description": description,
            "thinking": thinking,
            "answer": answer,
            "raw": parsed["raw"],
            "refined_by_deepseek": refined_by_deepseek,
            "success": True,
            "message": "Diagnosis completed" if language == "en" else "诊断完成",
        }
        yield f"data: {json.dumps({'type': 'final', 'result': final_payload}, ensure_ascii=False)}\n\n"

        temp_path = generation_result.get("temp_image_path")
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def main() -> None:
    uvicorn.run("app:app", host="0.0.0.0", port=5900, reload=False)


if __name__ == "__main__":
    main()
