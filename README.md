---
license: cc-by-nc-sa-4.0
language:
- zh
- en
tags:
- medical
---
---

# SkinGPT-R1

**SkinGPT-R1** is a dermatological reasoning vision Language model (VLM).

## ⚠️ Disclaimer

This model is **for research and educational use only**. It is **NOT a substitute for professional medical advice, diagnosis, or treatment**.  

## 🛠️ Environment Setup

To ensure compatibility, we strongly recommend creating a fresh Conda environment.

### 1. Create Conda Environment

Create a new environment named skingpt-r1 with Python 3.10:

```bash
conda create -n skingpt-r1 python=3.10 -y
conda activate skingpt-r1
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### (Optional) For faster inference on NVIDIA GPUs:

```bash
pip install flash-attn --no-build-isolation
```

## 🚀 Usage

### Quick Start

If you just installed the environment and want to check if it works:

Open ***demo.py*** and Change the ***IMAGE_PATH*** variable to your image file.

```bash
python demo.py
```

### Interactive Chat

To have a multi-turn conversation (e.g., asking follow-up questions about the diagnosis) in your terminal:
```bash
python chat.py --image ./test_images/lesion.jpg
```
### FastAPI Backend Deployment

To deploy the model as a backend service (supporting image uploads and session management):

#### Start the Server

```bash
python app.py
```
#### API Workflow
Manage sessions via state_id to support multi-user history.

Upload: POST /v1/upload/{state_id} — Uploads an image for the session.

Chat: POST /v1/predict/{state_id} — Sends text (JSON: {"message": "..."}) and gets a response.

Reset: POST /v1/reset/{state_id} — Clears session history and images.
#### Client Example
```python
import requests

API_URL = "http://localhost:5900"
STATE_ID = "patient_001"

# 1. Upload Image
with open("skin_image.jpg", "rb") as f:
    requests.post(f"{API_URL}/v1/upload/{STATE_ID}", files={"file": f})

# 2. Ask for Diagnosis
response = requests.post(
    f"{API_URL}/v1/predict/{STATE_ID}", 
    json={"message": "Please analyze this image."}
)
print("AI:", response.json()["message"])

# 3. Ask Follow-up
response = requests.post(
    f"{API_URL}/v1/predict/{STATE_ID}", 
    json={"message": "What treatment do you recommend?"}
)
print("AI:", response.json()["message"])
```