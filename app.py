from fastapi import FastAPI, File, UploadFile
from openai import OpenAI
import base64
import os
import uvicorn
import json

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
                    "You are a handwriting evaluation engine. "
                    "Evaluate the overall quality of the handwriting in the image. "
                    "Do not guess unseen content."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "이 이미지는 손글씨 사진입니다.\n\n"
                            "보이는 글씨 전체를 기준으로 handwriting 품질을 평가하세요.\n\n"
                            "평가 기준:\n"
                            "- 글자 형태의 안정성\n"
                            "- 획의 일관성\n"
                            "- 자간 및 균형\n"
                            "- 가독성\n\n"
                            "다음 3개 중 하나로만 grade를 선택하세요:\n"
                            "- Excellent\n"
                            "- Good\n"
                            "- Poor\n\n"
                            "그리고 글씨체를 개선하기 위한 짧고 구체적인 feedback을 작성하세요.\n\n"
                            "반드시 아래 JSON 형식만 출력하세요.\n"
                            "다른 설명은 절대 포함하지 마세요.\n\n"
                            "출력 형식: "
                            "{"
                            "  \"grade\": \"Excellent | Good | Poor\","
                            "  \"feedback\": \"string\""
                            "}"
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

    # OpenAI가 반환한 JSON "문자열"
    result_str = vision_ocr_word_score(image_bytes)
    result_json = json.loads(result_str)

    return {
        "result": result_json
    }


@app.post("/scan")
async def scan_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()

    # OpenAI가 반환한 JSON "문자열"
    result_str = vision_scan_best_object(image_bytes)
    result_json = json.loads(result_str)

    return {
        "result": result_json
    }

# =========================
# Entry
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
