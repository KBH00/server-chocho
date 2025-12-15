from fastapi import FastAPI, File, UploadFile
from openai import OpenAI
import base64
import os
import uvicorn

# =========================
# App
# =========================
app = FastAPI()

# =========================
# OpenAI Client
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# Utils
# =========================
def encode_image_bytes(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

# =========================
# OCR (word-level scoring)
# =========================
def vision_ocr_word_score(image_bytes: bytes) -> str:
    base64_image = encode_image_bytes(image_bytes)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an OCR engine for handwriting evaluation. "
                    "You must extract visible text only."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "이 이미지에 보이는 글씨를 OCR 하세요.\n"
                            "문장을 단어 단위로 분리하세요.\n"
                            "각 단어 글씨의 평가 점수를 0~1 사이 score로 추정하세요.\n"
                            "반드시 JSON 배열만 출력하세요.\n\n"
                            "출력 형식:\n"
                            "["
                            "  { \"text\": \"단어\", \"score\": 0.XXX }"
                            "]"
                            "추측하지 말고, 보이는 글자만 사용하세요."
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
# Scan (object detection)
# =========================
def vision_scan_best_object(image_bytes: bytes) -> str:
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
                            "이 이미지에 보이는 객체 중 confidence가 가장 높은 객체만 추출해주세요.\n"
                            "JSON 배열로만 출력해 주세요.\n"
                            "객체의 한글 이름, 영어 이름을 포함하세요.\n"
                            "출력 형식:\n"
                            "["
                            "  { \"kor\": \"단어\", \"eng\": \"word\" }"
                            "]"
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
# Routes
# =========================
@app.get("/")
async def root():
    return {"message": "OpenAI Vision OCR & Scan Server running"}

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = vision_ocr_word_score(image_bytes)
    return { "result": result }

@app.post("/scan")
async def scan_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = vision_scan_best_object(image_bytes)
    return { "result": result }

# =========================
# Entry
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
