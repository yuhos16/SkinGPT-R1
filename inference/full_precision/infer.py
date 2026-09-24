from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .model_utils import (
        DEFAULT_MODEL_PATH,
        SkinGPTModel,
        build_single_turn_messages,
        resolve_model_path,
    )
except ImportError:
    from model_utils import (
        DEFAULT_MODEL_PATH,
        SkinGPTModel,
        build_single_turn_messages,
        resolve_model_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SkinGPT-R1 full-precision single inference")
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--prompt",
        type=str,
        default="Please analyze this skin image and provide a diagnosis.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not Path(args.image).exists():
        print(f"Error: Image not found at {args.image}")
        return

    model = SkinGPTModel(resolve_model_path(args.model_path))
    messages = build_single_turn_messages(args.image, args.prompt)

    print(f"\nAnalyzing {args.image}...")
    response = model.generate_response(messages)

    print("-" * 40)
    print("Result:")
    print(response)
    print("-" * 40)


if __name__ == "__main__":
    main()
