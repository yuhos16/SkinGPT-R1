# chat.py
import argparse
import os
from model_utils import SkinGPTModel

def main():
    parser = argparse.ArgumentParser(description="SkinGPT-R1 Multi-turn Chat")
    parser.add_argument("--model_path", type=str, default="../checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to initial image")
    args = parser.parse_args()

    # 初始化模型
    bot = SkinGPTModel(args.model_path)

    # 初始化对话历史
    # 系统提示词
    system_prompt = "You are a professional AI dermatology assistant. Analyze the skin condition carefully."
    
    # 构造第一条包含图片的消息
    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} not found.")
        return

    history = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": f"{system_prompt}\n\nPlease analyze this image."}
            ]
        }
    ]

    print("\n=== SkinGPT-R1 Chat (Type 'exit' to quit) ===")
    print(f"Image loaded: {args.image}")
    
    # 获取第一轮诊断
    print("\nModel is thinking...", end="", flush=True)
    response = bot.generate_response(history)
    print(f"\rAssistant: {response}\n")
    
    # 将助手的回复加入历史
    history.append({"role": "assistant", "content": [{"type": "text", "text": response}]})

    # 进入多轮对话循环
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue

            # 加入用户的新问题
            history.append({"role": "user", "content": [{"type": "text", "text": user_input}]})

            print("Model is thinking...", end="", flush=True)
            response = bot.generate_response(history)
            print(f"\rAssistant: {response}\n")

            # 加入助手的新回复
            history.append({"role": "assistant", "content": [{"type": "text", "text": response}]})

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()