import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

# === Configuration ===
MODEL_PATH = "../checkpoint" 
IMAGE_PATH = "test_image.jpg" # Please replace with your actual image path
PROMPT = "You are a professional AI dermatology assistant. Please analyze this skin image and provide a diagnosis."

def main():
    print(f"Loading model from {MODEL_PATH}...")
    
    # 1. Load Model
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Check Image
    import os
    if not os.path.exists(IMAGE_PATH):
        print(f"Warning: Image not found at '{IMAGE_PATH}'. Please edit IMAGE_PATH in demo.py")
        # Create a dummy image for code demonstration purposes if needed, or just return
        return

    # 3. Prepare Inputs
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": IMAGE_PATH},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    print("Processing...")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # 4. Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9
        )

    # 5. Decode
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    print("\n=== Diagnosis Result ===")
    print(output_text[0])
    print("========================")

if __name__ == "__main__":
    main()