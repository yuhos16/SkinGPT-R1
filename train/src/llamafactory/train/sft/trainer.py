# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput
    from ...hparams import FinetuningArguments, ModelArguments

logger = logging.get_logger(__name__)

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if model_args is not None and model_args.fp8:
            configure_fp8_environment(model_args)
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            self.model_accepts_loss_kwargs = False
            self.add_callback(SaveProcessorCallback(processor))

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            self._gen_kwargs = gen_kwargs

    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute the SFT, visual alignment, routing and skin-label losses."""
        
        root_model = model.module if hasattr(model, "module") else model
        
        
        if hasattr(root_model, "text_bias"):
            labels = inputs.get("labels")
            if labels is not None:
                
                inputs["skin_vocab_mask"] = (labels != IGNORE_INDEX).unsqueeze(-1).to(device=labels.device)
                
                inputs["lm_head_weight"] = root_model.lm_head.weight.to(device=labels.device)

        
        
        outputs = model(**inputs)

        
        teacher = inputs.get("teacher_feat", None)
        skin_labels = inputs.get("skin_labels", None)
        
        # [DEBUG] Check if critical inputs are present
        # if self.state.global_step % 1 == 0: # Print every step for now
        #     print(f"\n[Trainer-Debug] Step {self.state.global_step}")
        #     print(f"  > teacher_feat: {'PRESENT' if teacher is not None else 'MISSING'}")
        #     print(f"  > skin_labels : {'PRESENT' if skin_labels is not None else 'MISSING'}")
        #     print(f"  > pixel_values: {'PRESENT' if 'pixel_values' in inputs else 'MISSING'}")
        #     print(f"  > image_grid_thw: {'PRESENT' if 'image_grid_thw' in inputs else 'MISSING'}")
        #     if teacher is not None:
        #         print(f"  > teacher shape: {teacher.shape}")

        vproj = getattr(outputs, "vision_proj", None)
        if vproj is None and isinstance(outputs, dict):
            vproj = outputs.get("vision_proj")

        aux_loss = getattr(outputs, "aux_loss", None)
        if aux_loss is None and isinstance(outputs, dict):
            aux_loss = outputs.get("aux_loss")

        skin_logits = getattr(outputs, "skin_logits", None)
        if skin_logits is None and isinstance(outputs, dict):
            skin_logits = outputs.get("skin_logits")

        # [HACK] Recover from 'attentions' if DDP stripped attributes
        if vproj is None:
            atts = getattr(outputs, "attentions", None)
            if atts is None and isinstance(outputs, dict):
                atts = outputs.get("attentions")
            
            if atts is not None and isinstance(atts, tuple) and len(atts) == 3:
                # [Trainer-Debug] Recovering...
                # print("[Trainer-Debug] ✅ SUCCESS: Recovered attributes from 'attentions' hack!")
                vproj_cand, aux_loss_cand, skin_logits_cand = atts
                
                # Check if they are valid tensors (not dummy 0.0)
                if isinstance(vproj_cand, torch.Tensor) and vproj_cand.dim() > 0:
                    vproj = vproj_cand
                if isinstance(aux_loss_cand, torch.Tensor):
                    aux_loss = aux_loss_cand
                if isinstance(skin_logits_cand, torch.Tensor) and skin_logits_cand.dim() > 0:
                    skin_logits = skin_logits_cand

        # [HACK 3 - ULTIMATE] Recover from model instance attribute
        if vproj is None:
            # Try to unwrap model to find 'latest_side_output'
            model_to_check = model
            if hasattr(model, "module"):
                model_to_check = model.module
            
            if hasattr(model_to_check, "latest_side_output"):
                side_data = model_to_check.latest_side_output
                # print("[Trainer-Debug] ✅ SUCCESS: Recovered attributes from 'model.latest_side_output'!")
                vproj = side_data.get("vision_proj", vproj)
                aux_loss = side_data.get("aux_loss", aux_loss)
                skin_logits = side_data.get("skin_logits", skin_logits)

        # [DEBUG] Check outputs from model forward
        # if self.state.global_step == 0:
        #     print(f"  > outputs type: {type(outputs)}")
        #     print(f"  > vision_proj : {'PRESENT' if vproj is not None else 'MISSING'}")
        #     print(f"  > skin_logits : {'PRESENT' if skin_logits is not None else 'MISSING'}")
        #     print(f"  > aux_loss    : {aux_loss if aux_loss is not None else 'MISSING'}")

        
        if teacher is not None:
            
            root_model.configure_out_dim(teacher.size(-1))

        
        
        
        
        loss_sft = torch.tensor(0.0, device=self.args.device, requires_grad=True)
        if "labels" in inputs:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()
            loss_sft = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1), 
                ignore_index=IGNORE_INDEX
            )

        
        loss_distill = torch.tensor(0.0, device=self.args.device, requires_grad=True)
        if teacher is not None and vproj is not None:

            cos_sim = F.cosine_similarity(vproj.float(), teacher.float(), dim=-1)
            loss_distill = (1.0 - cos_sim).mean() * 10.0

        
        if aux_loss is None:
            aux_loss = torch.tensor(0.0, device=self.args.device, requires_grad=True)
        
        
        loss_skin = torch.tensor(0.0, device=self.args.device, requires_grad=True)
        if skin_labels is not None and skin_logits is not None:
            loss_skin = F.cross_entropy(skin_logits.float(), skin_labels.long())

        
        alpha = float(os.getenv("SFT_WEIGHT", "1.0"))
        beta  = float(os.getenv("DISTILL_WEIGHT", "0.1"))
        gamma = float(os.getenv("MOE_AUX_WEIGHT", "0.001"))
        delta = float(os.getenv("SKIN_LOSS_WEIGHT", "0.1"))

        
        target_dtype = loss_sft.dtype
        total = (alpha * loss_sft + 
                 beta * loss_distill.to(target_dtype) + 
                 gamma * aux_loss.to(target_dtype) + 
                 delta * loss_skin.to(target_dtype))

        
        if not total.requires_grad:
            
            total = total + (root_model.logit_bias_scale * 0.0)

        
        if self.state.global_step % self.args.logging_steps == 0:
            log_payload = {
                "loss_sft": float(loss_sft),
                "loss_distill": float(loss_distill),
                "loss_aux": float(aux_loss),
                "loss_skin": float(loss_skin),
                "loss_total": float(total),
            }
            if skin_labels is not None and skin_logits is not None:
                with torch.no_grad():
                    acc = (skin_logits.argmax(-1) == skin_labels).float().mean()
                    log_payload["acc_skin"] = float(acc)
            self.log(log_payload)

        return (total, outputs) if return_outputs else total

    
    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        if self.args.predict_with_generate:
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")
        loss, generated_tokens, _ = super().prediction_step(model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs)
        return loss, generated_tokens, labels