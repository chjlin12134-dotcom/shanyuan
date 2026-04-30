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

st.set_page_config(
    page_title="善緣 · 在這裡陪你",
    page_icon="🪷",
    layout="centered",
)

# ==========================================
# 載入 system prompt（從 .md 抓出標記區段）
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
# 簡易中文關鍵字檢索（不需 embedding）
# ==========================================
STOPWORDS = set("的了在是我有和就不人都一上也很到說要去你會著沒看好自己這那麼什麼怎麼為什麼如果但是因為所以可以這樣那樣有點覺得".split())

def tokenize(text: str) -> list[str]:
    """中文簡易分詞：抓 2–3 字 n-gram。"""
    text = re.sub(r"[^一-龥a-zA-Z0-9]+", " ", text)
    tokens: list[str] = []
    for chunk in text.split():
        if not chunk:
            continue
        # 英文／數字直接收
        if re.match(r"^[a-zA-Z0-9]+$", chunk):
            tokens.append(chunk.lower())
            continue
        # 中文做 bigram + trigram
        for i in range(len(chunk) - 1):
            bg = chunk[i:i+2]
            if bg not in STOPWORDS:
                tokens.append(bg)
        for i in range(len(chunk) - 2):
            tokens.append(chunk[i:i+3])
    return tokens


def retrieve(corpus: pd.DataFrame, query: str, k: int = MAX_RETRIEVED) -> list[dict]:
    """從語料庫挑出最相關的 k 筆。回傳 dict 列表。"""
    if corpus.empty:
        return []
    q_tokens = Counter(tokenize(query))
    if not q_tokens:
        return []

    scores = []
    for idx, row in corpus.iterrows():
        # 把所有相關欄位串起來做匹配
        doc = f"{row['標題']} {row['大師金句']} {row['具體故事']} {row['善緣陪伴語']}"
        d_tokens = Counter(tokenize(doc))
        score = sum(min(q_tokens[t], d_tokens[t]) for t in q_tokens if t in d_tokens)
        if score > 0:
            scores.append((score, idx))

    scores.sort(reverse=True)
    top = scores[:k]
    return [corpus.iloc[i].to_dict() for _, i in top]


def format_retrieved(items: list[dict]) -> str:
    """把檢索到的語料格式化成可附在 system prompt 後面的文字。"""
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
# UI
# ==========================================
st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 720px; }
    .stChatMessage { font-size: 16px; line-height: 1.7; }
    h1 { font-weight: 300; letter-spacing: 0.05em; }
    .subtitle { color: #888; font-size: 14px; margin-top: -10px; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🪷 善緣")
st.markdown('<p class="subtitle">在這裡陪你坐一段路</p>', unsafe_allow_html=True)

corpus = load_corpus()
system_prompt = load_system_prompt()
client = get_client()

with st.sidebar:
    st.markdown("### 🪷 善緣")
    st.markdown(
        "我是善緣，一個在星雲大師智慧啟發下，樂於陪伴您的朋友。\n\n"
        "我不是法師，也不是心理師。但我深受人間佛教的薰陶與培育，"
        "學習用真誠、尊重、不評判的心來陪伴每一個人。\n\n"
        "在我們的對話裡，我會適時就您的情形和您分享大師的智慧話語"
        "——不是說教，只是點一盞小燈。\n\n"
        "不論你有什麼信仰，或者沒有宗教信仰，"
        "只要你想說說話，善緣都在這裡陪你。\n\n"
        "我相信，經由陪伴與對話，你能找到自己的答案。"
    )
    st.divider()
    if not corpus.empty:
        st.caption(f"語料庫：{len(corpus)} 篇")
    else:
        st.caption("⚠️ 語料庫尚未建立，請先執行 extract_corpus.py")
    if st.button("清除對話"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption(
        "🙏 感謝佛光山人間佛教研究院\n"
        "開放星雲大師全集，支持「善緣」專案。"
    )

# 初始化對話歷史
if "messages" not in st.session_state:
    st.session_state.messages = []

    
# 顯示歷史
for msg in st.session_state.messages:
    avatar = "🪷" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# 使用者輸入
if user_input := st.chat_input("想說什麼？"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 檢索相關語料（基於最近 3 輪的內容，讓檢索有上下文）
    recent_text = " ".join(
        m["content"] for m in st.session_state.messages[-3:] if m["role"] == "user"
    )
    retrieved = retrieve(corpus, recent_text)
    retrieval_block = format_retrieved(retrieved)
    full_system = system_prompt + retrieval_block

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

    st.session_state.messages.append({"role": "assistant", "content": full_response})
