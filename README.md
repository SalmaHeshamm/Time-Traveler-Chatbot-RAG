# ✈️ Time Travel Chatbot — Egyptian Historical Eras

An interactive Arabic-language chatbot that lets users explore ancient Egyptian historical eras through a conversational RAG (Retrieval-Augmented Generation) system, complete with text-to-speech responses.

---

## 🌟 Features

- 🏛️ **Multi-Era Knowledge Base** — Covers four distinct historical periods: Pharaonic, Greek (Ptolemaic), Roman, and Medieval Egypt
- 🤖 **AI-Powered Answers** — Uses Groq LLM via a custom `MultiEraEgyptianRAG` pipeline for accurate, context-aware responses
- 🔊 **Arabic Text-to-Speech** — Converts answers to spoken Arabic audio using `gTTS`
- 🎨 **Modern UI** — Glassmorphism-style design with gradient backgrounds and smooth animations
- 📥 **Audio Download** — Users can download the generated audio as an MP3 file

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/time-travel-chatbot.git
cd time-travel-chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `groq` | LLM API client |
| `gTTS` | Google Text-to-Speech |
| `python-dotenv` | Environment variable loader |
| `langchain` / `faiss-cpu` | RAG pipeline (in `multi.py`) |

### 3. Configure API Keys

**Option A — Local `.env` file:**

```env
GROQ_API_KEY=your_groq_api_key_here
```

**Option B — Streamlit Cloud Secrets (`secrets.toml`):**

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

### 4. Build the Knowledge Base

Before running the app for the first time, generate the vector store:

```bash
python multi.py
```

This will create the `knowledge_base/` directory with embeddings for all four eras.

### 5. Run the App

```bash
streamlit run app.py
```

---

## 🧭 How to Use

1. Open the app and click **🚀 ابدأ الرحلة** (Start the Journey)
2. Select a historical era from the dropdown
3. Type your question in Arabic (e.g., *من هو رمسيس الثاني؟*)
4. Click **🗣️ اسأل واستمع** to get a written + audio answer
5. Use the **📥 تحميل الصوت** button to download the MP3

---

## 🏺 Supported Historical Eras

| Key | Era | Icon |
|---|---|---|
| `pharaonic` | Ancient Pharaonic Egypt | 🏛️ |
| `greek` | Greek / Ptolemaic Period | 🏺 |
| `roman` | Roman Egypt | 🏟️ |
| `medieval` | Medieval Islamic Egypt | 🕌 |

---

## 🔧 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | API key from [console.groq.com](https://console.groq.com) |
| `TOKENIZERS_PARALLELISM` | Auto-set | Set to `false` to avoid HuggingFace tokenizer warnings |

