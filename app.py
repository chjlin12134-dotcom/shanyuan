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

    /* 主內容區 */
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2rem;
        max-width: 740px;
        background-color: #fdfaf6;
    }

    /* 頁面底色 */
    .stApp {
        background-color: #fdfaf6;
    }

    /* 標題 */
    h1 {
        font-weight: 300;
        letter-spacing: 0.12em;
        color: #4a5568;
        font-size: 2rem !important;
    }

    /* 副標題 */
    .subtitle {
        color: #9aab9a;
        font-size: 14px;
        margin-top: -8px;
        letter-spacing: 0.08em;
    }

    /* 分隔線 */
    hr {
        border: none;
        border-top: 1px solid #e8e0d5;
        margin: 1rem 0;
    }

    /* 聊天泡泡 - 助理 */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
    div[data-testid="stChatMessage"][class*="assistant"] {
        background-color: #f5f0e8 !important;
        border-radius: 12px;
        border-left: 3px solid #c8d8c0;
        padding: 0.8rem 1rem;
    }

    /* 聊天泡泡 - 使用者 */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
    div[data-testid="stChatMessage"][class*="user"] {
        background-color: #eef4ee !important;
        border-radius: 12px;
        border-left: 3px solid #a8c4a8;
        padding: 0.8rem 1rem;
    }

    /* 聊天文字 */
    .stChatMessage p {
        font-size: 16px;
        line-height: 1.85;
        color: #3d3d3d;
    }

    /* 輸入框 */
    [data-testid="stChatInput"] textarea {
        background-color: #fdf8f2 !important;
        border: 1px solid #d4c9b8 !important;
        border-radius: 12px !important;
        font-family: 'Noto Serif TC', serif;
        font-size: 15px;
        color: #3d3d3d;
        min-height: 120px !important;
        padding: 16px 20px !important;
    }

    /* 輸入框外層容器 */
    [data-testid="stChatInput"] {
        padding-top: 0.5rem;
    }

    /* 清除按鈕對齊右下 */
    div[data-testid="stHorizontalBlock"] .stButton > button {
        font-size: 12px;
        padding: 0.2rem 0.6rem;
        background-color: #f0ebe3;
        border-color: #d4c9b8;
        color: #888;
        border-radius: 16px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8f4ee;
        border-right: 1px solid #e8e0d5;
    }

    [data-testid="stSidebar"] p {
        font-size: 14px;
        line-height: 1.9;
        color: #5a5a5a;
    }

    [data-testid="stSidebar"] h3 {
        color: #7a9a7a;
        font-weight: 400;
        letter-spacing: 0.1em;
    }

    /* 按鈕 */
    .stButton > button {
        background-color: #e8f0e8;
        border: 1px solid #b8d0b8;
        border-radius: 20px;
        color: #4a6a4a;
        font-family: 'Noto Serif TC', serif;
        font-size: 13px;
        padding: 0.3rem 1rem;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #d4e8d4;
        border-color: #8ab08a;
        color: #2a4a2a;
    }

    /* 祈福禮區塊 */
    .blessing-box {
        background: linear-gradient(135deg, #f5f0e8 0%, #eef4ee 100%);
        border: 1px solid #c8d8c0;
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        text-align: center;
        position: relative;
    }

    .blessing-box::before {
        content: "🪷";
        font-size: 1.5rem;
        display: block;
        margin-bottom: 0.5rem;
    }

    .blessing-title {
        color: #7a9a7a;
        font-size: 12px;
        letter-spacing: 0.2em;
        margin-bottom: 0.6rem;
    }

    .blessing-quote {
        color: #4a5568;
        font-size: 15px;
        line-height: 1.9;
        font-style: italic;
    }

    .blessing-source {
        color: #9aab9a;
        font-size: 11px;
        margin-top: 0.6rem;
    }

    /* caption 文字 */
    .stCaptionContainer p {
        color: #9aab9a;
        font-size: 12px;
    }
</style>
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
    st.markdown("#### 🪷 善緣")
    st.caption(
        "我是善緣，一個深受人間佛教薰陶的在家陪伴者，"
        "在星雲大師智慧的啟發下，學習用真誠、尊重、不評判的心陪伴每一個人。\n\n"
        "在我們的對話裡，我會適時分享大師的智慧話語——不是說教，只是點一盞小燈。\n\n"
        "不論你有什麼信仰，或者沒有宗教信仰，只要你想說說話，善緣都在這裡陪你。\n\n"
        "我相信，經由陪伴與對話，你能找到自己的答案。"
    )
    st.divider()
    st.markdown("🙏 **感謝佛光山人間佛教研究院**", unsafe_allow_html=False)
    st.caption("開放星雲大師全集，支持「善緣」專案。")


# ==========================================
# 主頁面標題
# ==========================================
st.markdown("# 🪷 善緣")
st.markdown('<p class="subtitle">在這裡陪你走一段路</p>', unsafe_allow_html=True)
st.markdown("---")


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
# 清除對話（緊貼輸入框上方）
# ==========================================
col_spacer, col_btn = st.columns([5, 1])
with col_btn:
    if st.button("🗑️ 清除", key="clear_main"):
        st.session_state.messages = []
        st.session_state.pop("auto_blessing", None)
        st.rerun()

# ==========================================
# 使用者輸入
# ==========================================
if user_input := st.chat_input("想說什麼？"):
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
