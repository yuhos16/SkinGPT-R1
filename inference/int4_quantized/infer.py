from __future__ import annotations

import argparse
import time
from pathlib import Path

try:
    from .model_utils import (
        DEFAULT_DO_SAMPLE,
        DEFAULT_MODEL_PATH,
        DEFAULT_MAX_NEW_TOKENS,
        DEFAULT_PROMPT,
        DEFAULT_REPETITION_PENALTY,
        DEFAULT_TEMPERATURE,
        DEFAULT_TOP_P,
        QuantizedSkinGPTModel,
        build_single_turn_messages,
    )
except ImportError:
    from model_utils import (
        DEFAULT_DO_SAMPLE,
        DEFAULT_MODEL_PATH,
        DEFAULT_MAX_NEW_TOKENS,
        DEFAULT_PROMPT,
        DEFAULT_REPETITION_PENALTY,
        DEFAULT_TEMPERATURE,
        DEFAULT_TOP_P,
        QuantizedSkinGPTModel,
        build_single_turn_messages,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SkinGPT-R1 INT4 inference")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image_path", type=str, required=True, help="Path to the test image")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt for diagnosis")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--do_sample", action="store_true", default=DEFAULT_DO_SAMPLE)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--repetition_penalty", type=float, default=DEFAULT_REPETITION_PENALTY)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not Path(args.image_path).exists():
        print(f"Error: Image not found at {args.image_path}")
        return

    print("=== [1] Initializing INT4 Quantization ===")
    print("BitsAndBytesConfig will be applied during model loading.")

    print("=== [2] Loading Model and Processor ===")
    start_load = time.time()
    model = QuantizedSkinGPTModel(args.model_path)
    print(f"Model loaded in {time.time() - start_load:.2f} seconds.")

    print("=== [3] Preparing Input ===")
    messages = build_single_turn_messages(args.image_path, args.prompt)

    print("=== [4] Generating Response ===")
    start_infer = time.time()
    output_text = model.generate_response(
        messages,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    print(f"Inference completed in {time.time() - start_infer:.2f} seconds.")
    print("\n================ MODEL OUTPUT ================\n")
    print(output_text)
    print("\n==============================================\n")


if __name__ == "__main__":
    main()
