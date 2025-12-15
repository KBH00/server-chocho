from fastapi import FastAPI, File, UploadFile
from paddleocr import PaddleOCR
from PIL import Image
import io
import uvicorn
import os
import numpy as np
import base64

from openai import OpenAI

# =========================
# FastAPI App
# =========================
app = FastAPI()

# =========================
# PaddleOCR 초기화
# =========================
ocr = PaddleOCR(
    lang="korean",
    det_db_thresh=0.2,
    det_db_box_thresh=0.4,
    use_angle_cls=True
)

# =========================
# OpenAI Client
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# Root
# =========================
@app.get("/")
async def root():
    return {"message": "OCR & Image2Text Server is running"}

# =========================
# OCR Logic
# =========================
def perform_ocr(image: Image.Image) -> str:
    print(f"이미지 크기: {image.size} 처리 중...")
    image_np = np.array(image)

    result = ocr.ocr(image_np)

    if not result or len(result) == 0:
        return "텍스트를 찾을 수 없습니다."

    data = result[0]
    texts = data["rec_texts"]
    scores = data["rec_scores"]

    result_texts = []
    for text, score in zip(texts, scores):
        result_texts.append(f"{text} (confidence: {score:.2f})")
        print(f"{text} (confidence: {score:.2f})")

    return "\n".join(result_texts) if result_texts else "텍스트를 찾을 수 없습니다."

# =========================
# OCR Endpoint
# =========================
@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    if image.mode != "RGB":
        image = image.convert("RGB")

    result_text = perform_ocr(image)

    return {"text": result_text}

# =========================
# Utils
# =========================
def encode_image_bytes(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

# =========================
# Image2Text Logic
# =========================
def detect_best_object_from_image(image_bytes: bytes) -> str:
    base64_image = encode_image_bytes(image_bytes)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a computer vision assistant."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "이 이미지에 보이는 객체 중 confidence가 가장 높은 객체만 추출해주세요. "
                            "JSON 배열로만 출력해 주세요. "
                            "객체는 name, category, confidence(0~1)를 포함하세요."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content

# =========================
# Image2Text Endpoint
# =========================
@app.post("/image2text")
async def image2text_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = detect_best_object_from_image(image_bytes)

    return {
        "result": result
    }

# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
