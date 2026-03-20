from __future__ import annotations

from pathlib import Path
from threading import Thread
from typing import List

import torch
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TextIteratorStreamer,
)

DEFAULT_MODEL_PATH = "./checkpoints"
DEFAULT_SYSTEM_PROMPT = "You are a professional AI dermatology assistant."


def resolve_model_path(model_path: str = DEFAULT_MODEL_PATH) -> str:
    """Resolve a model path for both cloned-repo and local-dev layouts."""
    raw_path = Path(model_path).expanduser()
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [raw_path]

    if not raw_path.is_absolute():
        candidates.append(Path.cwd() / raw_path)
        candidates.append(repo_root / raw_path)
        if raw_path.parts and raw_path.parts[0] == repo_root.name:
            candidates.append(repo_root.joinpath(*raw_path.parts[1:]))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(raw_path)


def build_single_turn_messages(
    image_path: str,
    prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> List[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"{system_prompt}\n\n{prompt}"},
            ],
        }
    ]


class SkinGPTModel:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, device: str | None = None):
        resolved_model_path = resolve_model_path(model_path)
        self.model_path = resolved_model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {resolved_model_path} on {self.device}...")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            resolved_model_path,
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
            attn_implementation="flash_attention_2" if self.device == "cuda" else None,
            device_map="auto" if self.device != "mps" else None,
            trust_remote_code=True,
        )

        if self.device == "mps":
            self.model = self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            resolved_model_path,
            trust_remote_code=True,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        print("Model loaded successfully.")

    def generate_response(
        self,
        messages,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
    ) -> str:
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
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
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                top_p=0.9,
                do_sample=True,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]

    def generate_response_stream(
        self,
        messages,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
    ):
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "top_p": 0.9,
            "do_sample": True,
            "streamer": streamer,
        }

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for text_chunk in streamer:
            yield text_chunk

        thread.join()
