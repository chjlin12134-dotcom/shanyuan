"""
善緣 — 人間佛教精神的 AI 陪伴助理
================================
Streamlit 雲端網頁介面

執行：
    streamlit run app.py

需要環境變數：
    ANTHROPIC_API_KEY
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st

# 自動載入同資料夾的 .env（如果存在）
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# ==========================================
# 設定
# ==========================================
BASE_DIR = Path(__file__).parent
CORPUS_CSV = BASE_DIR / "shanyuan_corpus.csv"
PROMPT_MD = BASE_DIR / "system_prompt.md"
MODEL = os.environ.get("CHAT_MODEL", "claude-sonnet-4-6")
MAX_RETRIEVED = 3  # 每輪檢索附上的語料數

# 道別關鍵詞（偵測祈福禮時機）
FAREWELL_WORDS = [
    "再見", "拜拜", "掰掰", "bye", "goodbye", "謝謝", "感謝", "謝了",
    "先這樣", "先這樣了", "回去了", "要去了", "結束了", "告辭",
    "感恩", "辛苦了", "辛苦你了", "下次見", "有空再聊",
]

st.set_page_config(
    page_title="善緣 · 在這裡陪你",
    page_icon="🪷",
    layout="centered",
)

# ==========================================
# 清淡荷花風格 CSS
# ==========================================
st.markdown("""
<style>
    /* Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+TC:wght@300;400;500&display=swap');

    /* 全站字型 */
    html, body, [class*="css"] {
        font-family: 'Noto Serif TC', 'Georgia', serif;
    }

    /* 頁面背景 */
    .stApp {
        background-color: #fdf8f0 !important;
        min-height: 100vh;
    }

    /* 蓮花水印：固定在頁面正中央，pointer-events none 不影響操作 */
    .lotus-bg {
        position: fixed;
        top: 50%;
        left: 60%;
        transform: translate(-50%, -50%);
        width: 820px;
        height: 820px;
        opacity: 0.13;
        pointer-events: none;
        z-index: 0;
    }

    /* 確保內容在蓮花上層 */
    .block-container, [data-testid="stSidebar"] {
        position: relative;
        z-index: 1;
    }

    [data-testid="stHeader"] {
        background: rgba(253,248,240,0.85) !important;
    }

    /* 主內容區：移除 Streamlit 預設頂部空白 */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 1.5rem;
        max-width: min(740px, 92vw);
        background: transparent !important;
    }

    /* 移除 Streamlit 頂部工具列佔位空間 */
    [data-testid="stAppViewContainer"] > section:first-child {
        padding-top: 0 !important;
    }

    /* 隱藏 Streamlit 頂部預設 header 空間 */
    [data-testid="stHeader"] {
        height: 0 !important;
        min-height: 0 !important;
    }

    /* 標題裝飾文字 */
    .deco-line {
        text-align: center;
        color: #c8d8c0;
        font-size: 11px;
        letter-spacing: 0.4em;
        margin-bottom: 1.5rem;
    }

    /* 分隔線 */
    hr {
        border: none;
        border-top: 1px solid rgba(200,185,168,0.4);
        margin: 1rem 0;
    }

    /* 聊天泡泡：溫暖卡片 */
    [data-testid="stChatMessage"] {
        background: rgba(255,252,247,0.92) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(200,216,192,0.5) !important;
        box-shadow: 0 2px 12px rgba(160,180,150,0.12), 0 1px 3px rgba(180,160,140,0.08);
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
    }

    /* 聊天文字 */
    .stChatMessage p {
        font-size: 16px;
        line-height: 1.9;
        color: #3d3d3d;
    }

    /* 輸入框容器 */
    [data-testid="stChatInput"] {
        background: rgba(255,252,247,0.95) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(200,185,168,0.5) !important;
        box-shadow: 0 2px 16px rgba(160,140,120,0.10);
        padding-top: 0.3rem;
    }

    /* 輸入框 */
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        border: none !important;
        font-family: 'Noto Serif TC', serif;
        font-size: 15px;
        color: #3d3d3d;
        min-height: 120px !important;
        padding: 16px 20px !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f3eb 0%, #f2efe8 100%);
        border-right: 1px solid rgba(200,185,168,0.3);
        box-shadow: 2px 0 12px rgba(180,160,140,0.08);
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] small {
        font-size: 13px;
        line-height: 1.85;
        color: #4a4a4a;
    }

    [data-testid="stSidebar"] h4 {
        color: #7a9a7a;
        font-weight: 400;
        letter-spacing: 0.10em;
        font-size: 0.95rem;
    }

    /* 按鈕：圓潤光澤 */
    .stButton > button {
        background: linear-gradient(135deg, #eef4ee 0%, #e4ede4 100%);
        border: 1px solid rgba(168,196,168,0.6);
        border-radius: 20px;
        color: #4a6a4a;
        font-family: 'Noto Serif TC', serif;
        font-size: 13px;
        padding: 0.3rem 1rem;
        transition: all 0.25s ease;
        box-shadow: 0 2px 6px rgba(140,180,140,0.12);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #e0ece0 0%, #d4e8d4 100%);
        border-color: rgba(138,176,138,0.8);
        color: #2a4a2a;
        box-shadow: 0 4px 12px rgba(140,180,140,0.2);
        transform: translateY(-1px);
    }

    /* 清除按鈕（右側小按鈕）*/
    div[data-testid="stHorizontalBlock"] .stButton > button {
        font-size: 12px;
        padding: 0.2rem 0.7rem;
        background: linear-gradient(135deg, #f5ede3 0%, #ede3d5 100%);
        border-color: rgba(200,185,168,0.6);
        color: #8a7a6a;
        box-shadow: 0 1px 4px rgba(180,160,140,0.10);
    }

    /* 祈福禮區塊：質感卡片 */
    .blessing-box {
        background: linear-gradient(135deg,
            rgba(255,255,255,0.7) 0%,
            rgba(238,244,238,0.7) 100%);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(200,216,192,0.5);
        border-radius: 20px;
        padding: 1.5rem 2rem;
        margin: 1.2rem 0;
        text-align: center;
        box-shadow:
            0 8px 32px rgba(140,180,140,0.12),
            0 2px 8px rgba(180,160,140,0.08),
            inset 0 1px 0 rgba(255,255,255,0.8);
        position: relative;
        overflow: hidden;
    }

    .blessing-box::after {
        content: "";
        position: absolute;
        top: -30px; right: -30px;
        width: 100px; height: 100px;
        background: radial-gradient(circle, rgba(200,216,192,0.2) 0%, transparent 70%);
        pointer-events: none;
    }

    .blessing-box::before {
        content: "🪷";
        font-size: 1.6rem;
        display: block;
        margin-bottom: 0.6rem;
        filter: drop-shadow(0 2px 4px rgba(180,140,140,0.2));
    }

    .blessing-title {
        color: #7a9a7a;
        font-size: 11px;
        letter-spacing: 0.3em;
        margin-bottom: 0.8rem;
    }

    .blessing-quote {
        color: #4a5568;
        font-size: 15px;
        line-height: 2;
        font-style: italic;
    }

    .blessing-source {
        color: #9aab9a;
        font-size: 11px;
        margin-top: 0.8rem;
        letter-spacing: 0.05em;
    }

    /* caption 文字 */
    .stCaptionContainer p {
        color: #7a8a7a;
        font-size: 14px;
    }

</style>
""", unsafe_allow_html=True)

# 散落花瓣（相對於內容區定位，不用fixed）
st.markdown("""
<div style="pointer-events:none; position:relative; height:0; overflow:visible;">
  <span style="position:absolute; font-size:38px; opacity:0.09; top:10px; left:10px; transform:rotate(-15deg);">🪷</span>
  <span style="position:absolute; font-size:22px; opacity:0.07; top:30px; right:-20px; transform:rotate(20deg);">🪷</span>
  <span style="position:absolute; font-size:28px; opacity:0.08; top:80px; left:20px; transform:rotate(10deg);">🪷</span>
  <span style="position:absolute; font-size:18px; opacity:0.07; top:60px; right:10px; transform:rotate(-25deg);">🪷</span>
  <span style="position:absolute; font-size:32px; opacity:0.065; top:130px; right:-10px; transform:rotate(8deg);">🪷</span>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 載入 system prompt
# ==========================================
@st.cache_data
def load_system_prompt() -> str:
    if not PROMPT_MD.exists():
        return "你是善緣，一位溫暖的陪伴者。"
    text = PROMPT_MD.read_text(encoding="utf-8")
    m = re.search(r"=== SYSTEM PROMPT 開始 ===(.*?)=== SYSTEM PROMPT 結束 ===", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


# ==========================================
# 載入語料庫
# ==========================================
@st.cache_data
def load_corpus() -> pd.DataFrame:
    if not CORPUS_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(CORPUS_CSV).fillna("")


# ==========================================
# 簡易中文關鍵字檢索
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


# ==========================================
# 祈福禮：從語料庫選出最相關金句
# ==========================================
def get_blessing(corpus: pd.DataFrame, conversation_text: str) -> dict | None:
    """根據對話內容，從語料庫挑出最相關的一句大師金句作為祈福禮。"""
    if corpus.empty:
        return None
    items = retrieve(corpus, conversation_text, k=1)
    if not items:
        # 若無匹配，隨機取一筆
        import random
        row = corpus.sample(1).iloc[0].to_dict()
        return row
    return items[0]


def is_farewell(text: str) -> bool:
    """偵測是否為道別訊息。"""
    text_lower = text.lower().strip()
    return any(w in text_lower for w in FAREWELL_WORDS)


def show_blessing(blessing: dict) -> None:
    """在頁面上顯示祈福禮區塊。"""
    quote = blessing.get("大師金句", "")
    source = blessing.get("標題", "")
    book = blessing.get("出處", "")
    if not quote:
        return
    source_text = f"——〈{source}〉" if source else ""
    if book:
        source_text += f"《{book}》"
    st.markdown(f"""
<div class="blessing-box">
    <div class="blessing-title">✦ 善緣的祈福禮 ✦</div>
    <div class="blessing-quote">「{quote}」</div>
    <div class="blessing-source">{source_text}</div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# Claude API client
# ==========================================
@st.cache_resource
def get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.error("找不到 ANTHROPIC_API_KEY，請在環境變數或 Streamlit secrets 設定")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


# ==========================================
# 載入資源
# ==========================================
corpus = load_corpus()
system_prompt = load_system_prompt()
client = get_client()



# ==========================================
# Sidebar
# ==========================================
with st.sidebar:
    st.markdown(
        "<div style='margin-top:0.5rem;'>"
        "<span style='font-family:\"Noto Serif TC\",serif; font-size:1rem; color:#7a9a7a; letter-spacing:0.1em;'>🪷 善緣</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "這是善緣，深受人間佛教薰陶的在家陪伴者，"
        "在星雲大師智慧啟發下，用真誠、尊重、不評判的心陪伴每一個人。\n\n"
        "不論你有什麼信仰，只要想說說話，善緣都在這裡陪你。\n\n"
        "相信經由陪伴與對話，你能找到自己的答案。"
    )
    st.divider()
    st.markdown(
        "<p style='font-size:10.5px; color:#7a8a7a; line-height:1.5; margin:0;'>"
        "🙏 感謝<b>佛光山人間佛教研究院</b><br>開放大師全集，支持「善緣」。</p>",
        unsafe_allow_html=True,
    )


# ==========================================
# 主頁面標題
# ==========================================
st.markdown("""
<div style="text-align:center; padding: 0.8rem 0 0.3rem 0;">
  <div style="display:flex; align-items:center; justify-content:center; gap:0.6rem; margin-bottom:0.15rem;">
    <svg width="90" height="28" viewBox="0 0 90 28" xmlns="http://www.w3.org/2000/svg" opacity="0.55">
      <path d="M5,24 Q22,4 38,18" stroke="#a8c4a8" stroke-width="1.1" fill="none"/>
      <path d="M0,26 Q18,6 32,20" stroke="#c8d8c0" stroke-width="0.7" fill="none"/>
      <ellipse cx="42" cy="14" rx="4" ry="7" fill="#e8c8c8" opacity="0.65" transform="rotate(-20,42,14)"/>
      <ellipse cx="48" cy="9" rx="4" ry="7" fill="#f0d0d0" opacity="0.6" transform="rotate(10,48,9)"/>
    </svg>
    <span style="font-family:'Noto Serif TC',Georgia,serif; font-size:24px; font-weight:300; letter-spacing:0.25em; color:#4a5568;">🪷 善 緣</span>
    <svg width="90" height="28" viewBox="0 0 90 28" xmlns="http://www.w3.org/2000/svg" opacity="0.55" style="transform:scaleX(-1);">
      <path d="M5,24 Q22,4 38,18" stroke="#a8c4a8" stroke-width="1.1" fill="none"/>
      <path d="M0,26 Q18,6 32,20" stroke="#c8d8c0" stroke-width="0.7" fill="none"/>
      <ellipse cx="42" cy="14" rx="4" ry="7" fill="#e8c8c8" opacity="0.65" transform="rotate(-20,42,14)"/>
      <ellipse cx="48" cy="9" rx="4" ry="7" fill="#f0d0d0" opacity="0.6" transform="rotate(10,48,9)"/>
    </svg>
  </div>
  <div style="font-family:'Noto Serif TC',Georgia,serif; font-size:12px; color:#9aab9a; letter-spacing:0.18em; margin-bottom:0.3rem;">在這裡陪你走一段路</div>
  <div style="height:1px; background:linear-gradient(90deg,transparent,rgba(200,216,192,0.5),transparent); margin:0 auto; width:70%;"></div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 初始化 session state
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "嗨，我是善緣。你今天過得怎麼樣？",
        }
    ]
if "auto_blessing" not in st.session_state:
    st.session_state.auto_blessing = None


# ==========================================
# 顯示對話歷史
# ==========================================
for msg in st.session_state.messages:
    avatar = "🪷" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        # 如果此訊息附有祈福禮，一起顯示
        if msg.get("blessing"):
            show_blessing(msg["blessing"])



# ==========================================
# 語音輸入（Groq Whisper）
# ==========================================
with st.expander("🎙️ 語音輸入", expanded=True):
    lang_choice = st.radio(
        "語言",
        ["繁體中文", "簡體中文"],
        horizontal=True,
        label_visibility="collapsed",
    )
    prompt_hint = "以下是繁體中文語音內容：" if lang_choice == "繁體中文" else "以下是简体中文语音内容："
    audio = st.audio_input("按下錄音，說完再按停止")
    if audio is not None:
        groq_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
        if not groq_key:
            st.warning("找不到 GROQ_API_KEY")
        else:
            with st.spinner("辨識中…"):
                try:
                    import httpx
                    response = httpx.post(
                        "https://api.groq.com/openai/v1/audio/transcriptions",
                        headers={"Authorization": f"Bearer {groq_key}"},
                        files={"file": ("audio.wav", audio.getvalue(), "audio/wav")},
                        data={"model": "whisper-large-v3-turbo", "language": "zh", "prompt": prompt_hint},
                        timeout=30,
                    )
                    result = response.json()
                    transcript = result.get("text", "")
                    if transcript:
                        st.session_state["voice_input"] = transcript
                        st.rerun()
                    elif "error" in result:
                        err = result["error"].get("code", "")
                        if err == "rate_limit_exceeded":
                            st.warning("請稍等幾秒再試一次 🙏")
                        else:
                            st.error(result["error"].get("message", "辨識失敗"))
                except Exception as e:
                    st.error(f"語音辨識失敗：{e}")

# 取出語音輸入（若有）
voice_text = st.session_state.pop("voice_input", None)

# ==========================================
# 使用者輸入
# ==========================================
user_input = voice_text or st.chat_input("想說什麼？")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 偵測道別
    farewell = is_farewell(user_input)

    # 檢索相關語料
    recent_text = " ".join(
        m["content"] for m in st.session_state.messages[-3:] if m["role"] == "user"
    )
    retrieved = retrieve(corpus, recent_text)
    retrieval_block = format_retrieved(retrieved)

    # 道別時加入給善緣的提示，讓回應自然溫暖地收尾，並預告祈福禮
    farewell_instruction = ""
    if farewell:
        farewell_instruction = (
            "\n\n---\n【本輪提示】使用者正在道別。"
            "請用溫暖自然的語氣與他們道別，簡短真誠，像朋友分別時說的最後一句話。"
            "在回應的最後，請自然地加上類似這樣的一句話（可以用自己的語氣說）："
            "「離開前，我會送你一句大師的話，帶著走。」\n"
        )

    full_system = system_prompt + retrieval_block + farewell_instruction

    # 串流回應
    with st.chat_message("assistant", avatar="🪷"):
        placeholder = st.empty()
        full_response = ""
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=1024,
                system=full_system,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"（連線出了點狀況：{e}）"
            placeholder.markdown(full_response)

        # 道別時：依整段對話脈絡，自動選出最貼近的大師金句作為祈福禮
        auto_blessing = None
        if farewell:
            full_conversation = " ".join(
                m["content"] for m in st.session_state.messages
                if m["role"] == "user"
            )
            auto_blessing = get_blessing(corpus, full_conversation)
            if auto_blessing:
                show_blessing(auto_blessing)

    # 儲存訊息（含祈福禮，重新整理時一起顯示）
    msg_entry: dict = {"role": "assistant", "content": full_response}
    if auto_blessing:
        msg_entry["blessing"] = auto_blessing
    st.session_state.messages.append(msg_entry)
