"""CPU checks of the actual bundled MoE/skin head; no model/data downloads."""
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
sys.dont_write_bytecode = True

import torch
import torch.nn.functional as F

root = Path(__file__).resolve().parents[1]
source = root / 'src/llamafactory/model/skin_vlm_adapter.py'
spec = importlib.util.spec_from_file_location('skingpt_training_core', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.manual_seed(42)
torch.set_num_threads(2)

# Actual default architecture: four layers, eight experts, top-2 routing.
head = module.PatchDistillHead(in_dim=1176)
assert len(head.adapters) == 4
assert all(layer.num_experts == 8 and layer.top_k == 2 for layer in head.adapters)
pixels = torch.randn(8, 1176)
grid = torch.tensor([[1, 2, 2], [1, 2, 2]])
output = head(pixels, grid)
assert output['skin_logits'].shape == (2, 3)
assert output['vision_proj'].shape == (2, 1024)
teacher = F.normalize(torch.randn(2, 1024), dim=-1)
loss = (
    0.1 * F.cross_entropy(output['skin_logits'], torch.tensor([0, 2]))
    + 0.001 * output['aux_loss']
    + 0.1 * 10.0 * (1.0 - F.cosine_similarity(output['vision_proj'], teacher)).mean()
)
assert torch.isfinite(loss)
loss.backward()
for name in ['in_proj.weight', 'skin_classifier.0.weight',
             'adapters.0.router_img.weight', 'adapters.0.router_skin.weight']:
    gradient = dict(head.named_parameters())[name].grad
    assert gradient is not None and torch.isfinite(gradient).all(), name
    assert gradient.abs().sum() > 0, name
assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.adapters[0].experts.parameters())

# Verify the source's lazy projection interface as well as explicit initialization.
small = module.PatchDistillHead(embed_dim=16, adapter_layers=1, num_experts=4, top_k=2)
small.configure_out_dim(8)
small_output = small(torch.randn(4, 12), torch.tensor([[1, 2, 2]]))
assert small_output['vision_proj'].shape == (1, 8)
print(json.dumps({'status': 'passed', 'device': 'cpu', 'adapter_layers': 4,
                  'experts_per_layer': 8, 'top_k': 2, 'skin_classes': 3,
                  'finite_loss_and_gradients': True,
                  'scope': 'Synthetic-tensor adapter forward and backward checks'}, indent=2))
