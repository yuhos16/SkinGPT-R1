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
    parser = argparse.ArgumentParser(description="SkinGPT-R1 full-precision multi-turn chat")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image", type=str, required=True, help="Path to initial image")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not Path(args.image).exists():
        print(f"Error: Image {args.image} not found.")
        return

    model = SkinGPTModel(resolve_model_path(args.model_path))
    history = build_single_turn_messages(
        args.image,
        "Please analyze this image.",
        system_prompt="You are a professional AI dermatology assistant. Analyze the skin condition carefully.",
    )

    print("\n=== SkinGPT-R1 Chat (Type 'exit' to quit) ===")
    print(f"Image loaded: {args.image}")

    print("\nModel is thinking...", end="", flush=True)
    response = model.generate_response(history)
    print(f"\rAssistant: {response}\n")
    history.append({"role": "assistant", "content": [{"type": "text", "text": response}]})

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue

            history.append({"role": "user", "content": [{"type": "text", "text": user_input}]})

            print("Model is thinking...", end="", flush=True)
            response = model.generate_response(history)
            print(f"\rAssistant: {response}\n")

            history.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
