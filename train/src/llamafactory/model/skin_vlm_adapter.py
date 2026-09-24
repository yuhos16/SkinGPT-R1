import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any, Dict
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration


class ExpertBlock(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, hidden_dim)
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
        self.experts = nn.ModuleList([
            ExpertBlock(hidden_dim, bottleneck_dim) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor, skin_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        img_logits = self.router_img(x)
        skin_bias = self.router_skin(skin_probs)
        router_logits = img_logits + skin_bias
        router_probs = F.softmax(router_logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-6)
        
        final_output = torch.zeros_like(x)
        for expert_idx, expert in enumerate(self.experts):
            expert_mask = (top_k_indices == expert_idx)
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
    def __init__(self, embed_dim: int = 1024, adapter_layers: int = 4,
                 in_dim: Optional[int] = None, out_dim: Optional[int] = None,
                 num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_proj = None if in_dim is None else nn.Linear(in_dim, embed_dim, bias=False)
        self.skin_classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.adapters = nn.ModuleList([
            SkinAwareMoEAdapter(embed_dim, num_experts=num_experts, top_k=top_k) 
            for _ in range(adapter_layers)
        ])
        self.out_proj: nn.Module = nn.Identity() if out_dim is None else nn.Linear(embed_dim, out_dim)

    def _ensure_in_proj(self, din: int, device, dtype):
        if self.in_proj is None:
            self.in_proj = nn.Linear(din, self.embed_dim, bias=False).to(device=device, dtype=dtype)

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> dict:
        P, Din = pixel_values.shape
        counts = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).tolist()
        device, dtype = pixel_values.device, pixel_values.dtype
        self._ensure_in_proj(Din, device, dtype)
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
        vision_proj  = self.out_proj(vision_embed)
        return {
            "vision_embed": vision_embed, 
            "vision_proj": vision_proj, 
            "aux_loss": total_aux_loss,      
            "skin_logits": torch.cat(all_skin_logits, dim=0) 
        }

    def configure_out_dim(self, out_dim: int):
        if isinstance(self.out_proj, nn.Linear) and self.out_proj.out_features == out_dim: return
        self.out_proj = nn.Linear(self.embed_dim, out_dim, bias=False) if out_dim != self.embed_dim else nn.Identity()
        try:
            p = next(self.parameters())
            self.out_proj.to(device=p.device, dtype=p.dtype)
        except StopIteration: pass


class SkinVLModelWithAdapter(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.distill_head = PatchDistillHead(embed_dim=1024, adapter_layers=4, num_experts=8, top_k=2)
        bottleneck = 64
        self.text_bias = nn.Sequential(
            nn.Linear(1024, bottleneck, bias=False),
            nn.Tanh(),
            nn.Linear(bottleneck, config.hidden_size, bias=False)
        )
        self.logit_bias_scale = nn.Parameter(torch.tensor(2.5, dtype=torch.bfloat16))
        
    def forward(self, *args, **kwargs):
        
        skin_vocab_mask = kwargs.pop("skin_vocab_mask", None)
        skin_labels = kwargs.get("skin_labels", None) 
        pixel_values = kwargs.get("pixel_values", None)
        image_grid_thw = kwargs.get("image_grid_thw", None)
        
        # DEBUG
        # print(f"[Adapter] Forward keys: {list(kwargs.keys())}")
        # print(f"[Adapter] pixel_values type: {type(pixel_values)}")
        # if isinstance(pixel_values, torch.Tensor):
        #     print(f"[Adapter] pixel_values shape: {pixel_values.shape}")

        # [DEBUG] Dump all keys to find where images are
        if pixel_values is None:
             print(f"\n[Adapter-Debug] ⚠️ pixel_values MISSING! Available keys: {list(kwargs.keys())}")
        
        if image_grid_thw is None:
             print(f"\n[Adapter-Debug] ⚠️ image_grid_thw MISSING! Available keys: {list(kwargs.keys())}")

        
        if isinstance(pixel_values, list):
            try:
                
                pixel_values = torch.cat(pixel_values, dim=0)
                kwargs["pixel_values"] = pixel_values 
            except Exception:
                pass 

        
        outputs = super().forward(*args, **kwargs)

        
        vision_embed = None
        loss_skin = torch.tensor(0.0, device=outputs.logits.device)
        aux_loss = torch.tensor(0.0, device=outputs.logits.device)

        # DEBUG
        # if pixel_values is None:
        #      print("[Adapter] ⚠️ No pixel_values found!")
        # if image_grid_thw is None:
        #      print("[Adapter] ⚠️ No image_grid_thw found!")

        
        if pixel_values is not None and image_grid_thw is not None:
            # [DEBUG] Entering MoE block
            # print("[Adapter-Debug] Entering MoE calculation block!")
            
            
            if not isinstance(pixel_values, torch.Tensor):
                 if isinstance(pixel_values, list):
                     pixel_values = torch.cat(pixel_values, dim=0)
                 else:
                     pixel_values = torch.tensor(pixel_values)
            
            
            image_grid_thw = image_grid_thw.to(pixel_values.device)

            
            side = self.distill_head(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
            vision_embed = side["vision_embed"]
            aux_loss = side["aux_loss"]
            
            
            if skin_labels is not None:
                
                skin_labels = skin_labels.to(side["skin_logits"].device)
                
                loss_fct = nn.CrossEntropyLoss()
                loss_skin = loss_fct(side["skin_logits"], skin_labels)

            
            setattr(outputs, "vision_embed", vision_embed)
            setattr(outputs, "vision_proj",  side["vision_proj"]) 
            setattr(outputs, "loss_skin",    loss_skin)
            setattr(outputs, "aux_loss",     aux_loss)
            setattr(outputs, "skin_logits",  side["skin_logits"])

            # [HACK] DDP Strips custom attributes. We use 'attentions' field to pass data to Trainer.
            # We pack (vision_proj, aux_loss, skin_logits) into attentions.
            # Ensure they are tensors.
            pack_vision_proj = side["vision_proj"] if side["vision_proj"] is not None else torch.tensor(0.0, device=aux_loss.device)
            pack_skin_logits = side["skin_logits"] if side["skin_logits"] is not None else torch.tensor(0.0, device=aux_loss.device)
            
            outputs.attentions = (pack_vision_proj, aux_loss, pack_skin_logits)

            # [HACK 3 - ULTIMATE] Direct storage on model instance
            # DDP might wrap this, so Trainer needs to access model.module.latest_side_output
            self.latest_side_output = {
                "vision_proj": side["vision_proj"],
                "aux_loss": aux_loss,
                "skin_logits": side["skin_logits"]
            }

            # [DEBUG] Confirm attributes set (Only print once)
            # print(f"[Adapter-Debug] Set attributes: vision_proj={side['vision_proj'] is not None}, aux_loss={aux_loss is not None} (Hack applied)")
        
        
        if hasattr(outputs, "logits") and vision_embed is not None and skin_vocab_mask is not None:
            
            bias_features = self.text_bias(vision_embed.to(self.logit_bias_scale.dtype)) 
            
            
            lm_weight = self.lm_head.weight.to(bias_features.dtype)
            vocab_bias = F.linear(bias_features, lm_weight) # [B, V]
            
            
            scale = self.logit_bias_scale.to(outputs.logits.dtype)
            outputs.logits = outputs.logits + (scale * vocab_bias[:, None, :] * skin_vocab_mask)

        
        
        if outputs.loss is not None:
            
            outputs.loss = outputs.loss + loss_skin + (0.01 * aux_loss)

        return outputs

    def freeze_all_but_distill(self):
        self.requires_grad_(False)
        for p in self.distill_head.parameters(): p.requires_grad_(True)
        for p in self.text_bias.parameters(): p.requires_grad_(True)
        self.logit_bias_scale.requires_grad_(True)

    def configure_out_dim(self, out_dim: int):
        self.distill_head.configure_out_dim(out_dim)

    def project_only(self, vision_embed: torch.Tensor) -> torch.Tensor:
        return self.distill_head.out_proj(vision_embed)