
import argparse
from model_utils import SkinGPTModel
import os

def main():
    parser = argparse.ArgumentParser(description="SkinGPT-R1 Single Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    parser.add_argument("--model_path", type=str, default="../checkpoint")
    parser.add_argument("--prompt", type=str, default="Please analyze this skin image and provide a diagnosis.")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return

    # 1. 加载模型 (复用 model_utils)
    # 这样你就不用在这里重复写 transformers 的加载代码了
    bot = SkinGPTModel(args.model_path)

    # 2. 构造单轮消息
    system_prompt = "You are a professional AI dermatology assistant."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": f"{system_prompt}\n\n{args.prompt}"}
            ]
        }
    ]

    # 3. 推理
    print(f"\nAnalyzing {args.image}...")
    response = bot.generate_response(messages)
    
    print("-" * 40)
    print("Result:")
    print(response)
    print("-" * 40)

if __name__ == "__main__":
    main()