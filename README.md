---
title: 善緣
emoji: 🪷
colorFrom: pink
colorTo: purple
sdk: docker
pinned: false
---

# 善緣 · 人間佛教精神的 AI 陪伴助理

> 以星雲大師人間佛教精神為基礎，對西方正向心理學進行在地轉化的 AI 陪伴者。
> 服務對象：一般庶民（不限佛教徒，任何信仰都歡迎）。
> 角色定位：不是法師，不是心理師，是受人間佛教薰陶的在家陪伴者。

---

## 專案結構

```
shanyuan_fulltext_claude v1/
├── extract_corpus.py        # 第一步：語料萃取（讀 Excel → 抓文章 → Claude 結構化萃取 → CSV）
├── system_prompt.md         # 第二步：善緣的人格與對話原則
├── app.py                   # 第三步：Streamlit 對話介面（含 RAG 檢索）
├── requirements.txt         # Python 套件
├── .env.example             # 環境變數範本
├── .gitignore
├── README.md                # 本檔
│
├── 人間萬事.xlsx              # 來源語料 1（989 篇）
├── 語料庫架構與命名規劃.xlsx    # 設計文件
└── shanyuan_corpus.csv      # 萃取後的語料庫（執行 extract_corpus.py 後產生）
```

---

## 安裝

需要 Python 3.10+。

```powershell
# 進專案資料夾
cd "C:\Users\June\AI_project\shanyuan_fulltext_claude v1"

# 建虛擬環境（建議）
python -m venv .venv
.venv\Scripts\activate

# 安裝套件
pip install -r requirements.txt
```

## 設定 API Key

**⚠️ 重要：請先把舊的 API Key 換掉**——你原本 `善緣語料庫_claude.py` 裡的金鑰已經寫在檔案裡，建議到 [Anthropic Console](https://console.anthropic.com/settings/keys) 撤銷舊的、產生新的。

PowerShell 設定環境變數（當次有效）：
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-api03-你的新金鑰"
```

或永久設定（重開終端後仍有效）：
```powershell
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "sk-ant-api03-你的新金鑰", "User")
```

---

## 執行流程

### 第一步：建語料庫（先小樣本測試）

先跑 5 篇驗證流程沒問題：

```powershell
python extract_corpus.py 5
```

確認 `shanyuan_corpus.csv` 內容合理後，再跑全部：

```powershell
python extract_corpus.py
```

特性：
- **斷點續跑**：中斷後重跑會自動跳過已處理的文章
- **進度條**：tqdm 顯示剩餘時間
- **失敗紀錄**：抓不到或萃取失敗的 URL 會記在 `.failed_urls.txt`
- **增量寫入**：每篇處理完立刻寫入 CSV，不會因為當機損失全部進度

成本估算：989 篇 × 每篇約 2K tokens 輸入 + 500 tokens 輸出，用 sonnet 4.6 大約 **USD $10–15**；用 haiku 大約 **USD $1–2**（在 .env 設 `EXTRACT_MODEL=claude-haiku-4-5-20251001` 就能切換）。

### 第二步：本機測試對話介面

```powershell
streamlit run app.py
```

瀏覽器自動開啟 http://localhost:8501，可以直接和善緣對話。

### 第三步：部署上雲

**推薦方案：Streamlit Community Cloud（免費）**

1. 把整個資料夾推到一個新的 GitHub repo（**確認 `.gitignore` 有把 `.env`、API Key、`.processed_urls.txt` 排除**）
2. 到 [share.streamlit.io](https://share.streamlit.io)，用 GitHub 登入
3. 點 "New app"，選你的 repo，main file 填 `app.py`
4. 點 "Advanced settings" → "Secrets"，貼入：
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-api03-你的金鑰"
   ```
5. 點 Deploy，等 2–3 分鐘就有公開網址了

**其他選項：**
- **Hugging Face Spaces**：選 Streamlit template，把檔案傳上去，在 Settings → Secrets 設 API Key
- **Railway／Render**：用 `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

---

## 設計理念與安全

### 為什麼 RAG 用關鍵字而不是 embedding？
- 989 篇規模小，關鍵字檢索夠用、零額外成本、零冷啟動延遲
- 之後語料庫長到上萬篇再考慮 embedding（建議 voyage-3 或 OpenAI text-embedding-3-small）

### 善緣不會做的事
詳見 `system_prompt.md`，重點：
- 不下診斷、不貼標籤、不開處方
- 不傳教、不引用「大師說」
- 不簡化、不比慘、不空話
- 危機處理會直接給台灣的求助資源（1925／1995／1980）

### 隱私
目前的 `app.py` 不存對話紀錄到伺服器，只在使用者瀏覽器的 session 裡。
如果未來要做使用者帳號、對話歷史，需要另外加 DB 與隱私政策。

---

## 待辦／可優化

- [ ] 萃取階段加 batch API 降成本（Anthropic 提供 50% 折扣）
- [ ] 對話介面加「主題探索」入口，讓使用者按維度／模組瀏覽語料
- [ ] 加上多輪對話的長期記憶（用 Claude 的 memory 或外部向量 DB）
- [ ] 寫一份「語料品質審查」腳本，抽樣檢查萃取結果是否符合風格
- [ ] 考慮加上多語版本（簡中、英文）

---

## 版本

- v1.0（2026-04-28）初版：完整萃取→ prompt → web app → 部署 pipeline
