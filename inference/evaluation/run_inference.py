"""Run the documented single-image or batched diagnostic inference example.

Invoke from the repository root with python -m inference.evaluation.run_inference.
The dry-run path uses only the Python standard library and never loads weights.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

from .protocol import REPO_ROOT, build_messages, generation_kwargs, load_labels


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, action='append', required=True,
                        help='Local image path. Repeat this flag for a batch.')
    parser.add_argument('--model-path', type=Path, default=REPO_ROOT / 'checkpoint')
    parser.add_argument('--config', type=Path, default=REPO_ROOT / 'configs/evaluation.json')
    parser.add_argument('--mode', choices=['ddi', 'classification'], default='ddi')
    parser.add_argument('--labels', type=Path, help='Ordered JSON array of candidate labels.')
    parser.add_argument('--skin-vocab-mask', type=Path,
                        help='Original binary training-vocabulary mask as a JSON array.')
    parser.add_argument('--attn-implementation', choices=['flash_attention_2', 'sdpa'])
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--max-new-tokens', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--greedy', action='store_true')
    parser.add_argument('--output', type=Path, help='Save JSON messages, settings, and outputs.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the request without GPU, image decoding, or weight loading.')
    args = parser.parse_args()
    if args.mode == 'classification' and args.labels is None:
        parser.error('--mode classification requires --labels')
    if args.mode == 'ddi' and args.labels is not None:
        parser.error('--labels is only valid with --mode classification')
    for path in args.image:
        if not path.is_file():
            parser.error(f'Image file does not exist: {path}')
    return args


def effective_config(args):
    config = json.loads(args.config.read_text(encoding='utf-8'))
    for key in ['attn_implementation', 'batch_size', 'max_new_tokens', 'seed']:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.greedy:
        config['do_sample'] = False
    for key in ['batch_size', 'max_new_tokens', 'min_pixels', 'max_pixels']:
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer.')
    if config['min_pixels'] > config['max_pixels']:
        raise ValueError('min_pixels cannot exceed max_pixels.')
    if config['attn_implementation'] not in ['flash_attention_2', 'sdpa']:
        raise ValueError('Unsupported attention implementation.')
    if config['dtype'] != 'bfloat16' or config['padding_side'] != 'left':
        raise ValueError('This example uses bfloat16 and left padding.')
    if not 0 < config['top_p'] <= 1 or config['temperature'] <= 0:
        raise ValueError('Require 0 < top_p <= 1 and temperature > 0.')
    if config['repetition_penalty'] <= 0:
        raise ValueError('repetition_penalty must be positive.')
    return config


def validate_mask(path, vocab_size):
    values = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(values, list) or len(values) != vocab_size:
        raise ValueError('The mask length must equal the model vocabulary size.')
    if any(type(v) not in [int, float, bool] or v not in [0, 1] for v in values):
        raise ValueError('The vocabulary mask must contain only binary values.')
    if not any(values):
        raise ValueError('The vocabulary mask cannot be all zero.')
    return values


def check_local_checkpoint(path):
    """Fail before model loading for a sparse checkout or Git LFS pointers."""
    index = path / 'model.safetensors.index.json'
    if not index.is_file():
        raise FileNotFoundError(f'A local checkpoint index is required: {index}')
    weight_map = json.loads(index.read_text())['weight_map']
    for filename in sorted(set(weight_map.values())):
        shard = path / filename
        if not shard.is_file():
            raise FileNotFoundError(f'Local weight shard is required: {shard}')
        with shard.open('rb') as stream:
            if stream.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):
                raise ValueError(f'{shard.name} is a Git LFS pointer, not a weight tensor.')
    config_path = path / 'config.json'
    checkpoint_config = json.loads(config_path.read_text())
    if 'SkinVLModelWithAdapter' not in checkpoint_config.get('architectures', []):
        raise ValueError('This entry point expects the SkinVLModelWithAdapter checkpoint.')
    return checkpoint_config


def run_model(args, config, messages):
    checkpoint = args.model_path.expanduser().resolve()
    checkpoint_config = check_local_checkpoint(checkpoint)
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, GenerationConfig, set_seed
    from inference.int4_quantized.model_utils import SkinVLModelWithAdapter

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This full-precision example requires a CUDA GPU with bfloat16 support.')
    if config['attn_implementation'] == 'flash_attention_2':
        try:
            importlib.metadata.version('flash-attn')
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError('Install flash-attn for FlashAttention-2, or explicitly select sdpa.') from exc
    torch.backends.cuda.matmul.allow_tf32 = config['allow_tf32']
    set_seed(config['seed'])

    class EvaluationModel(SkinVLModelWithAdapter):
        # Expose the optional mask to generation's keyword validation. The Qwen
        # generation input preparation carries this keyword through to forward.
        def forward(self, *args, skin_vocab_mask=None, **kwargs):
            return super().forward(*args, skin_vocab_mask=skin_vocab_mask, **kwargs)

    model, loading = EvaluationModel.from_pretrained(
        str(checkpoint), dtype=torch.bfloat16,
        device_map={'': 'cuda:0'}, attn_implementation=config['attn_implementation'],
        local_files_only=True, output_loading_info=True,
    )
    if loading.get('missing_keys') or loading.get('unexpected_keys') or loading.get('mismatched_keys'):
        raise RuntimeError(f'Checkpoint/model mismatch: {loading}')
    model.eval()
    model.config.use_cache = True
    if hasattr(model.config, 'text_config'):
        model.config.text_config.use_cache = True
    processor = AutoProcessor.from_pretrained(
        str(checkpoint), min_pixels=config['min_pixels'], max_pixels=config['max_pixels'],
        padding_side='left', local_files_only=True,
    )
    processor.tokenizer.padding_side = 'left'
    # Build a fresh configuration to avoid hidden sampling overrides in a checkpoint.
    generation_config = GenerationConfig(
        bos_token_id=processor.tokenizer.bos_token_id,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        **generation_kwargs(config),
    )
    model.generation_config.use_cache = True
    extra = {}
    if args.skin_vocab_mask:
        vocab_size = checkpoint_config.get('vocab_size', model.config.text_config.vocab_size)
        mask = validate_mask(args.skin_vocab_mask, vocab_size)
        extra['skin_vocab_mask'] = torch.tensor(mask, dtype=torch.bfloat16, device='cuda:0')
    outputs = []
    for start in range(0, len(messages), config['batch_size']):
        batch = messages[start:start + config['batch_size']]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
        images, videos = process_vision_info(batch)
        inputs = processor(text=texts, images=images, videos=videos, padding=True, return_tensors='pt')
        inputs.pop('mm_token_type_ids', None)
        inputs = inputs.to('cuda:0')
        with torch.inference_mode():
            generated = model.generate(**inputs, **extra, generation_config=generation_config)
        new_tokens = generated[:, inputs.input_ids.shape[1]:]
        outputs.extend(processor.batch_decode(new_tokens, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False))
    versions = {'python': platform.python_version(), 'cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(0),
                'generation_config': generation_config.to_dict()}
    for package in ['torch', 'transformers', 'qwen-vl-utils', 'flash-attn']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return outputs, versions, hashlib.sha256((checkpoint / 'config.json').read_bytes()).hexdigest()


def main():
    args = parse_args()
    config = effective_config(args)
    labels = load_labels(args.labels) if args.labels else None
    messages = [build_messages(str(p.expanduser().resolve()), args.mode, labels) for p in args.image]
    payload = {
        'mode': args.mode, 'dry_run': args.dry_run, 'model_path': str(args.model_path),
        'configuration': config, 'generation_kwargs': generation_kwargs(config),
        'skin_vocab_mask_enabled': bool(args.skin_vocab_mask),
        'skin_vocab_mask_sha256': hashlib.sha256(args.skin_vocab_mask.read_bytes()).hexdigest() if args.skin_vocab_mask else None,
        'messages': messages,
    }
    if not args.dry_run:
        outputs, versions, config_digest = run_model(args, config, messages)
        payload.update(outputs=outputs, runtime=versions, checkpoint_config_sha256=config_digest)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
