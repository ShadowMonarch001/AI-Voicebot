# 🎤 My Digital Twin Voice Bot

This is a **voice-enabled chatbot** that speaks and answers **just like me**.  
It uses **RAG (Retrieval-Augmented Generation)** + **LLMs** to mix stored personal knowledge with natural conversation.  
The bot can **listen to your voice**, **understand questions**, and **reply in my style** with both text and speech.  

---

## 🚀 Features
- 🎙️ **Voice Input** using `streamlit-mic-recorder`  
- 📝 **Speech-to-Text** via AssemblyAI  
- 🧠 **RAG (semantic search)** on my personal knowledge base  
- 🤖 **LLM Responses** generated via OpenRouter NVIDIA Nemotron Nano 9B V2
- 🔊 **Voice Output** (Text-to-Speech with gTTS, auto-play in browser)  
- 💬 **Chat History** with context awareness  
- ⚡ **Fallback** to natural conversation if no KB match  

---

## 🏗️ Tech Stack
- [Streamlit](https://streamlit.io/) – Web interface  
- [AssemblyAI](https://www.assemblyai.com/) – Speech-to-text  
- [Sentence Transformers](https://www.sbert.net/) – Embeddings  
- [FAISS](https://faiss.ai/) – Semantic search  
- [OpenRouter API](https://openrouter.ai/) – LLM backend  
- [gTTS](https://pypi.org/project/gTTS/) – Text-to-speech  

---

## 📦 Installation

Clone this repo:

```bash
git clone https://github.com/your-username/my-digital-twin-voice-bot.git
cd my-digital-twin-voice-bot
````

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 API Keys Setup

Create a `.streamlit/secrets.toml` file:

```toml
OPENROUTER_API_KEY = "your_openrouter_api_key"
ASSEMBLYAI_API_KEY = "your_assemblyai_api_key"
```

---

## ▶️ Run the Bot

```bash
streamlit run app.py
```

---

## 📚 My Knowledge Base

The bot includes a **knowledge base of my personal facts** (life story, superpower, growth areas, etc.).
Example snippet from the KB:

```python
KNOWLEDGE_BASE = [
    {
        "id": "misconception",
        "question": "What misconception do your coworkers have about you?",
        "answer": "People think I'm quiet, but I'm deeply collaborative once work begins.I'm fully invested in collaborative problem-solving and team success."
    },
    {
        "id": "superpower",
        "question": "What's your #1 superpower?",
        "answer": "Biological adaptability, like Darwin from X-Men — evolve on the fly in any situation."
    }
]
```

👉 You can extend this with more of my traits, stories, and personal answers.

---

## 🎤 How to Use

1. Click **Start Recording** 🎙️
2. Ask me a question (e.g., *“What’s your story?”*)
3. Bot transcribes your voice 📝
4. It finds the best matching memory (RAG) or answers naturally
5. Bot replies in **my voice + text**

---

## 📊 Demo Limits

* Uses free APIs (OpenRouter + AssemblyAI) with rate limits (there is rate limit on NVIDIA Nemotron Nano 9B V2 and assemblyai)
* Limited to **30 requests/session** in this demo
* If limits hit → wait a bit or retry later

---

## 🛠️ Requirements

See [requirements.txt](requirements.txt):

```txt
streamlit
streamlit-mic-recorder
assemblyai
faiss-cpu
numpy
requests
gTTS
sentence-transformers
scipy
torch
```

---

## 📌 Example Prompts

* "Tell me your life story"
* "What’s your superpower?"
* "What areas are you growing in?"
* "What do people misunderstand about you?"
* "How do you push your limits?"

---

## 📸 Screenshot (UI Preview)

<img width="539" height="821" alt="image" src="https://github.com/user-attachments/assets/c725a35d-b7a7-49f6-af76-0ca2e5cde7b6" />


---
