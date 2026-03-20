from __future__ import annotations

from pathlib import Path
from threading import Thread
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)

DEFAULT_MODEL_PATH = "./checkpoints"
DEFAULT_SYSTEM_PROMPT = (
    "You are a professional AI dermatology assistant. "
    "Reason step by step, keep the reasoning concise, avoid repetition, "
    "and always finish with <answer>...</answer>."
)
DEFAULT_MAX_NEW_TOKENS = 768
DEFAULT_CONTINUE_TOKENS = 256
DEFAULT_DO_SAMPLE = False
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.15
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3
DEFAULT_PROMPT = (
    "Act as a dermatologist. Analyze the visual features of this skin lesion "
    "step by step, and provide a final diagnosis."
)


def resolve_model_path(model_path: str = DEFAULT_MODEL_PATH) -> str:
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
) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"{system_prompt}\n\n{prompt}"},
            ],
        }
    ]


def build_quantization_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def resolve_quantized_device_map():
    if not torch.cuda.is_available():
        raise RuntimeError("INT4 quantized inference requires a CUDA GPU.")
    return {"": f"cuda:{torch.cuda.current_device()}"}


class StopOnTokenSequence(StoppingCriteria):
    def __init__(self, stop_ids: list[int]):
        super().__init__()
        self.stop_ids = stop_ids
        self.stop_length = len(stop_ids)

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if self.stop_length == 0 or input_ids.shape[1] < self.stop_length:
            return False
        return input_ids[0, -self.stop_length :].tolist() == self.stop_ids


class ExpertBlock(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class SkinAwareMoEAdapter(nn.Module):
    def __init__(self, hidden_dim, num_experts=8, top_k=2, bottleneck_dim=64):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_img = nn.Linear(hidden_dim, num_experts, bias=False)
        self.router_skin = nn.Linear(3, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [ExpertBlock(hidden_dim, bottleneck_dim) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor, skin_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        img_logits = self.router_img(x)
        skin_bias = self.router_skin(skin_probs)
        router_logits = img_logits + skin_bias
        router_probs = F.softmax(router_logits, dim=-1)

        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-6)

        final_output = torch.zeros_like(x)
        for expert_idx, expert in enumerate(self.experts):
            expert_mask = top_k_indices == expert_idx
            if expert_mask.any():
                rows, k_indices = torch.where(expert_mask)
                inp = x[rows]
                out = expert(inp)
                weights = top_k_probs[rows, k_indices].unsqueeze(-1)
                final_output.index_add_(0, rows, (out * weights).to(final_output.dtype))

        mean_prob = router_probs.mean(0)
        mask_all = torch.zeros_like(router_probs)
        mask_all.scatter_(1, top_k_indices, 1.0)
        mean_freq = mask_all.mean(0)
        aux_loss = (mean_prob * mean_freq).sum() * self.num_experts

        return x + final_output, aux_loss


class PatchDistillHead(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        adapter_layers: int = 4,
        in_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_proj = None if in_dim is None else nn.Linear(in_dim, embed_dim, bias=False)
        self.skin_classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
        self.adapters = nn.ModuleList(
            [
                SkinAwareMoEAdapter(embed_dim, num_experts=num_experts, top_k=top_k)
                for _ in range(adapter_layers)
            ]
        )
        self.out_proj: nn.Module = (
            nn.Identity() if out_dim is None else nn.Linear(embed_dim, out_dim)
        )

    def _ensure_in_proj(self, din: int, device, dtype):
        if self.in_proj is None:
            self.in_proj = nn.Linear(din, self.embed_dim, bias=False).to(device=device, dtype=dtype)

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> dict:
        _, din = pixel_values.shape
        counts = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).tolist()
        device, dtype = pixel_values.device, pixel_values.dtype
        self._ensure_in_proj(din, device, dtype)
        chunks = torch.split(pixel_values, counts, dim=0)

        pooled, all_skin_logits = [], []
        total_aux_loss = torch.tensor(0.0, device=device, dtype=dtype)

        for x in chunks:
            h = self.in_proj(x)
            global_feat = h.mean(dim=0, keepdim=True)
            skin_logits = self.skin_classifier(global_feat)
            skin_probs = F.softmax(skin_logits, dim=-1)
            all_skin_logits.append(skin_logits)
            skin_probs_expanded = skin_probs.expand(h.size(0), -1)

            for adapter in self.adapters:
                h, layer_loss = adapter(h, skin_probs_expanded)
                total_aux_loss += layer_loss
            pooled.append(h.mean(dim=0))

        vision_embed = torch.stack(pooled, dim=0)
        vision_proj = self.out_proj(vision_embed)
        return {
            "vision_embed": vision_embed,
            "vision_proj": vision_proj,
            "aux_loss": total_aux_loss,
            "skin_logits": torch.cat(all_skin_logits, dim=0),
        }

    def configure_out_dim(self, out_dim: int):
        if isinstance(self.out_proj, nn.Linear) and self.out_proj.out_features == out_dim:
            return
        self.out_proj = (
            nn.Linear(self.embed_dim, out_dim, bias=False)
            if out_dim != self.embed_dim
            else nn.Identity()
        )
        try:
            params = next(self.parameters())
            self.out_proj.to(device=params.device, dtype=params.dtype)
        except StopIteration:
            pass


class SkinVLModelWithAdapter(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.distill_head = PatchDistillHead(
            embed_dim=1024,
            adapter_layers=4,
            num_experts=8,
            top_k=2,
            in_dim=1176,
        )
        bottleneck = 64
        self.text_bias = nn.Sequential(
            nn.Linear(1024, bottleneck, bias=False),
            nn.Tanh(),
            nn.Linear(bottleneck, config.hidden_size, bias=False),
        )
        self.logit_bias_scale = nn.Parameter(torch.tensor(2.5, dtype=torch.bfloat16))

    def forward(self, *args, **kwargs):
        skin_vocab_mask = kwargs.pop("skin_vocab_mask", None)
        skin_labels = kwargs.get("skin_labels", None)
        pixel_values = kwargs.get("pixel_values", None)
        image_grid_thw = kwargs.get("image_grid_thw", None)

        if isinstance(pixel_values, list):
            try:
                pixel_values = torch.stack(pixel_values)
                kwargs["pixel_values"] = pixel_values
            except Exception:
                pass

        outputs = super().forward(*args, **kwargs)

        vision_embed = None
        loss_skin = torch.tensor(0.0, device=outputs.logits.device)
        aux_loss = torch.tensor(0.0, device=outputs.logits.device)

        if pixel_values is not None and image_grid_thw is not None:
            if not isinstance(pixel_values, torch.Tensor):
                if isinstance(pixel_values, list):
                    pixel_values = torch.stack(pixel_values)
                else:
                    pixel_values = torch.tensor(pixel_values)

            image_grid_thw = image_grid_thw.to(pixel_values.device)
            side = self.distill_head(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
            vision_embed = side["vision_embed"]
            aux_loss = side["aux_loss"]

            if skin_labels is not None:
                skin_labels = skin_labels.to(side["skin_logits"].device)
                loss_skin = nn.CrossEntropyLoss()(side["skin_logits"], skin_labels)

            setattr(outputs, "vision_embed", vision_embed)
            setattr(outputs, "vision_proj", side["vision_proj"])
            setattr(outputs, "loss_skin", loss_skin)
            setattr(outputs, "aux_loss", aux_loss)
            setattr(outputs, "skin_logits", side["skin_logits"])

            pack_vision_proj = (
                side["vision_proj"]
                if side["vision_proj"] is not None
                else torch.tensor(0.0, device=aux_loss.device)
            )
            pack_skin_logits = (
                side["skin_logits"]
                if side["skin_logits"] is not None
                else torch.tensor(0.0, device=aux_loss.device)
            )
            outputs.attentions = (pack_vision_proj, aux_loss, pack_skin_logits)

            self.latest_side_output = {
                "vision_proj": side["vision_proj"],
                "aux_loss": aux_loss,
                "skin_logits": side["skin_logits"],
            }

        if hasattr(outputs, "logits") and vision_embed is not None and skin_vocab_mask is not None:
            bias_features = self.text_bias(vision_embed.to(self.logit_bias_scale.dtype))
            lm_weight = self.lm_head.weight.to(bias_features.dtype)
            vocab_bias = F.linear(bias_features, lm_weight)
            scale = self.logit_bias_scale.to(outputs.logits.dtype)
            outputs.logits = outputs.logits + (scale * vocab_bias[:, None, :] * skin_vocab_mask)

        if outputs.loss is not None:
            outputs.loss = outputs.loss + loss_skin + (0.01 * aux_loss)

        return outputs

    def freeze_all_but_distill(self):
        self.requires_grad_(False)
        for params in self.distill_head.parameters():
            params.requires_grad_(True)
        for params in self.text_bias.parameters():
            params.requires_grad_(True)
        self.logit_bias_scale.requires_grad_(True)

    def configure_out_dim(self, out_dim: int):
        self.distill_head.configure_out_dim(out_dim)

    def project_only(self, vision_embed: torch.Tensor) -> torch.Tensor:
        return self.distill_head.out_proj(vision_embed)


def load_quantized_model_and_processor(model_path: str = DEFAULT_MODEL_PATH):
    resolved_model_path = resolve_model_path(model_path)
    quantization_config = build_quantization_config()
    model = SkinVLModelWithAdapter.from_pretrained(
        resolved_model_path,
        device_map=resolve_quantized_device_map(),
        quantization_config=quantization_config,
        attn_implementation="sdpa",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        resolved_model_path,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )
    return model, processor


def get_model_device(model) -> torch.device:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def prepare_inputs(processor, model, messages: list[dict]):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(get_model_device(model))
    inputs.pop("mm_token_type_ids", None)
    return inputs


class QuantizedSkinGPTModel:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        resolved_model_path = resolve_model_path(model_path)
        print(f"Loading INT4 model from {resolved_model_path}...")
        self.model, self.processor = load_quantized_model_and_processor(resolved_model_path)
        self.model_path = resolved_model_path
        self.device = get_model_device(self.model)
        self.stop_ids = self.processor.tokenizer.encode("</answer>", add_special_tokens=False)
        print(f"Model loaded successfully on {self.device}.")

    @staticmethod
    def has_complete_answer(text: str) -> bool:
        return "<answer>" in text and "</answer>" in text

    def _build_generation_kwargs(
        self,
        inputs,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        repetition_penalty: float,
        top_p: float,
        no_repeat_ngram_size: int,
        streamer=None,
    ) -> dict:
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "use_cache": True,
            "stopping_criteria": StoppingCriteriaList([StopOnTokenSequence(self.stop_ids)]),
        }
        if streamer is not None:
            generation_kwargs["streamer"] = streamer
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        return generation_kwargs

    def _generate_text(
        self,
        messages,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        repetition_penalty: float,
        top_p: float,
        no_repeat_ngram_size: int,
    ) -> str:
        inputs = prepare_inputs(self.processor, self.model, messages)
        generation_kwargs = self._build_generation_kwargs(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        with torch.inference_mode():
            generated_ids = self.model.generate(**generation_kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]

    def generate_response(
        self,
        messages,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        continue_tokens: int = DEFAULT_CONTINUE_TOKENS,
        do_sample: bool = DEFAULT_DO_SAMPLE,
        temperature: float = DEFAULT_TEMPERATURE,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        top_p: float = DEFAULT_TOP_P,
        no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
    ) -> str:
        output_text = self._generate_text(
            messages=messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        if not self.has_complete_answer(output_text) and continue_tokens > 0:
            output_text = self._generate_text(
                messages=messages,
                max_new_tokens=max_new_tokens + continue_tokens,
                do_sample=do_sample,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        return output_text

    def generate_response_stream(
        self,
        messages,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        continue_tokens: int = DEFAULT_CONTINUE_TOKENS,
        do_sample: bool = DEFAULT_DO_SAMPLE,
        temperature: float = DEFAULT_TEMPERATURE,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        top_p: float = DEFAULT_TOP_P,
        no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
    ):
        inputs = prepare_inputs(self.processor, self.model, messages)
        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs = self._build_generation_kwargs(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
            streamer=streamer,
        )

        def _generate():
            with torch.inference_mode():
                self.model.generate(**generation_kwargs)

        thread = Thread(target=_generate)
        thread.start()

        partial_chunks = []
        for text_chunk in streamer:
            partial_chunks.append(text_chunk)
            yield text_chunk

        thread.join()

        partial_text = "".join(partial_chunks)
        if not self.has_complete_answer(partial_text) and continue_tokens > 0:
            completed_text = self._generate_text(
                messages=messages,
                max_new_tokens=max_new_tokens + continue_tokens,
                do_sample=do_sample,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            if completed_text.startswith(partial_text):
                tail_text = completed_text[len(partial_text) :]
                if tail_text:
                    yield tail_text
