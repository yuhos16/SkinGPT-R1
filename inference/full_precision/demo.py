from __future__ import annotations

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

IMAGE_PATH = "test_image.jpg"
PROMPT = "Please analyze this skin image and provide a diagnosis."


def main() -> None:
    if not Path(IMAGE_PATH).exists():
        print(f"Warning: Image not found at '{IMAGE_PATH}'. Please edit IMAGE_PATH in demo.py")
        return

    model = SkinGPTModel(resolve_model_path(DEFAULT_MODEL_PATH))
    messages = build_single_turn_messages(IMAGE_PATH, PROMPT)

    print("Processing...")
    output_text = model.generate_response(messages)

    print("\n=== Diagnosis Result ===")
    print(output_text)
    print("========================")


if __name__ == "__main__":
    main()
