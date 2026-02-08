# model_utils.py
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer
from qwen_vl_utils import process_vision_info
from PIL import Image
import os
from threading import Thread

class SkinGPTModel:
    def __init__(self, model_path, device=None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {model_path} on {self.device}...")
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
            attn_implementation="flash_attention_2" if self.device == "cuda" else None,
            device_map="auto" if self.device != "mps" else None,
            trust_remote_code=True
        )
        
        if self.device == "mps":
            self.model = self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            min_pixels=256*28*28, 
            max_pixels=1280*28*28
        )
        print("Model loaded successfully.")

    def generate_response(self, messages, max_new_tokens=1024, temperature=0.7):
        """
        处理多轮对话的历史消息列表并生成回复
        messages format: 
        [
            {'role': 'user', 'content': [{'type': 'image', 'image': 'path...'}, {'type': 'text', 'text': '...'}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': '...'}]}
        ]
        """
        # 预处理文本模板
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 预处理视觉信息
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True
            )

        # 解码输出 (去除输入的token)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0]
    
    def generate_response_stream(self, messages, max_new_tokens=2048, temperature=0.7):
        """
        流式生成响应
        返回一个生成器，逐个yield生成的文本chunk
        """
        # 预处理文本模板
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 预处理视觉信息
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        # 创建 TextIteratorStreamer 用于流式输出
        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # 准备生成参数
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "do_sample": True,
            "streamer": streamer,
        }
        
        # 在单独的线程中运行生成
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # 逐个yield生成的文本
        for text_chunk in streamer:
            yield text_chunk
        
        thread.join()