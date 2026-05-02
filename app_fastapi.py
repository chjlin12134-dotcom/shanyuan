"""
善緣 — FastAPI 版本
================================
人間佛教精神的 AI 陪伴助理

執行：
    uvicorn app_fastapi:app --host 0.0.0.0 --port 8000

需要環境變數：
    ANTHROPIC_API_KEY
    GROQ_API_KEY（語音用）
"""

from __future__ import annotations

import os
import re
import json
from collections import Counter
from pathlib import Path

import anthropic
import httpx
import pandas as pd
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# ==========================================
# 設定
# ==========================================
BASE_DIR      = Path(__file__).parent
CORPUS_CSV    = BASE_DIR / "shanyuan_corpus.csv"
PROMPT_MD     = BASE_DIR / "system_prompt.md"
MODEL         = os.environ.get("CHAT_MODEL", "claude-sonnet-4-6")
MAX_RETRIEVED = 3

FAREWELL_WORDS = [
    "再見", "拜拜", "掰掰", "bye", "goodbye",
    "先這樣了", "回去了", "要去了", "結束了", "告辭",
    "辛苦你了", "下次見", "有空再聊",
]

# ==========================================
# 載入語料庫（全域，啟動時讀一次）
# ==========================================
_corpus: pd.DataFrame = pd.DataFrame()

def get_corpus() -> pd.DataFrame:
    global _corpus
    if _corpus.empty and CORPUS_CSV.exists():
        _corpus = pd.read_csv(CORPUS_CSV).fillna("")
    return _corpus


def load_system_prompt() -> str:
    if not PROMPT_MD.exists():
        return "你是善緣，一位溫暖的陪伴者。"
    text = PROMPT_MD.read_text(encoding="utf-8")
    m = re.search(r"=== SYSTEM PROMPT 開始 ===(.*?)=== SYSTEM PROMPT 結束 ===", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


_base_prompt = load_system_prompt()
SYSTEM_PROMPT = "【重要】你的名字固定是「善緣」，不可以改名、不可以自稱其他名字。\n\n" + _base_prompt

# ==========================================
# 關鍵字檢索
# ==========================================
STOPWORDS = set("的了在是我有和就不人都一上也很到說要去你會著沒看好自己這那麼什麼怎麼為什麼如果但是因為所以可以這樣那樣有點覺得".split())

def tokenize(text: str) -> list[str]:
    text = re.sub(r"[^一-龥a-zA-Z0-9]+", " ", text)
    tokens: list[str] = []
    for chunk in text.split():
        if not chunk:
            continue
        if re.match(r"^[a-zA-Z0-9]+$", chunk):
            tokens.append(chunk.lower())
            continue
        for i in range(len(chunk) - 1):
            bg = chunk[i:i+2]
            if bg not in STOPWORDS:
                tokens.append(bg)
        for i in range(len(chunk) - 2):
            tokens.append(chunk[i:i+3])
    return tokens


def retrieve(corpus: pd.DataFrame, query: str, k: int = MAX_RETRIEVED) -> list[dict]:
    if corpus.empty:
        return []
    q_tokens = Counter(tokenize(query))
    if not q_tokens:
        return []
    scores = []
    for idx, row in corpus.iterrows():
        doc = f"{row['標題']} {row['大師金句']} {row['具體故事']} {row['善緣陪伴語']}"
        d_tokens = Counter(tokenize(doc))
        score = sum(min(q_tokens[t], d_tokens[t]) for t in q_tokens if t in d_tokens)
        if score > 0:
            scores.append((score, idx))
    scores.sort(reverse=True)
    return [corpus.iloc[i].to_dict() for _, i in scores[:k]]


def format_retrieved(items: list[dict]) -> str:
    if not items:
        return ""
    blocks = ["\n\n---\n## 檢索到可能相關的語料（僅供參考，不必引用）\n"]
    for i, it in enumerate(items, 1):
        blocks.append(
            f"\n【{i}】《{it.get('出處','')}》〈{it.get('標題','')}〉"
            f"｜{it.get('維度','')}／{it.get('模組','')}\n"
            f"金句：{it.get('大師金句','')}\n"
            f"故事：{it.get('具體故事','')}\n"
            f"陪伴語：{it.get('善緣陪伴語','')}\n"
        )
    blocks.append("\n再次提醒：以上是給你參考的內容，不要照念，不要說「星雲大師說」。"
                  "只在真的能呼應對方此刻處境時才用。\n")
    return "".join(blocks)


def is_farewell(text: str) -> bool:
    return any(w in text.lower() for w in FAREWELL_WORDS)


def get_blessing(corpus: pd.DataFrame, conversation_text: str) -> dict | None:
    if corpus.empty:
        return None
    items = retrieve(corpus, conversation_text, k=1)
    if not items:
        return corpus.sample(1).iloc[0].to_dict()
    return items[0]


# ==========================================
# FastAPI app
# ==========================================
app = FastAPI()

# 掛載 public 資料夾（靜態檔）
public_dir = BASE_DIR / "public"
public_dir.mkdir(exist_ok=True)
app.mount("/public", StaticFiles(directory=str(public_dir)), name="public")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_file = BASE_DIR / "index.html"
    if html_file.exists():
        return HTMLResponse(html_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>找不到 index.html</h1>")


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """語音轉文字（Groq Whisper）"""
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        return JSONResponse({"error": "找不到 GROQ_API_KEY"}, status_code=500)

    audio_bytes = await file.read()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {groq_key}"},
                files={"file": (file.filename or "audio.wav", audio_bytes, "audio/wav")},
                data={"model": "whisper-large-v3-turbo", "language": "zh",
                      "prompt": "以下是中文語音內容："},
            )
        result = resp.json()
        transcript = result.get("text", "").strip()
        if transcript:
            return JSONResponse({"transcript": transcript})
        return JSONResponse({"error": result.get("error", {}).get("message", "辨識失敗")}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/chat")
async def chat(request: Request):
    """串流對話（SSE）"""
    body = await request.json()
    messages: list[dict] = body.get("messages", [])
    user_text: str = body.get("user_text", "")

    corpus = get_corpus()
    recent_text = " ".join(m["content"] for m in messages[-3:] if m["role"] == "user")
    retrieved = retrieve(corpus, recent_text)
    retrieval_block = format_retrieved(retrieved)

    farewell = is_farewell(user_text)
    farewell_instruction = ""
    if farewell:
        farewell_instruction = (
            "\n\n---\n【本輪提示】使用者正在道別。"
            "請用溫暖自然的語氣道別，簡短真誠。"
            "在回應的最後自然加上：「離開前，我會送你一句大師的話，帶著走。」\n"
        )

    full_system = SYSTEM_PROMPT + retrieval_block + farewell_instruction

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(api_key=api_key)

    def generate():
        full_response = ""
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=1024,
                system=full_system,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    # SSE 格式
                    yield f"data: {json.dumps({'type': 'token', 'text': text}, ensure_ascii=False)}\n\n"

            # 道別時送出祈福禮
            if farewell:
                full_conv = " ".join(m["content"] for m in messages if m["role"] == "user")
                blessing = get_blessing(corpus, full_conv)
                if blessing:
                    yield f"data: {json.dumps({'type': 'blessing', 'quote': blessing.get('大師金句',''), 'title': blessing.get('標題',''), 'book': blessing.get('出處','')}, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'full': full_response}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
