import io
import json
from datetime import datetime

import assemblyai as aai
import faiss
import numpy as np
import requests
import streamlit as st
from gtts import gTTS
from sentence_transformers import SentenceTransformer
from streamlit_mic_recorder import mic_recorder

# Page config - MUST BE FIRST
st.set_page_config(
    page_title="RAG-Powered Interview Voice Bot",
    page_icon="🎤",
    layout="centered"
)

# Initialize session state IMMEDIATELY after page config
if "messages" not in st.session_state:
    st.session_state.messages = []
if "request_count" not in st.session_state:
    st.session_state.request_count = 0
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None
if "processing_audio" not in st.session_state:
    st.session_state.processing_audio = False
if "last_processed_audio_id" not in st.session_state:
    st.session_state.last_processed_audio_id = None
if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized = False

# Custom CSS with FIXED colors
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        animation: fadeIn 0.5s;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        color: #1565c0;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        color: #4a148c;
    }
    .rag-info {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #856404;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1.1rem;
        border-radius: 8px;
        cursor: pointer;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
    }
    .stats-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Knowledge Base
KNOWLEDGE_BASE = [
    {
        "id": "life_story",
        "question": "What should we know about your life story?",
        "answer": "My life story! Oh, it's gonna get messy because the Protagonist, itself is messed up! I was born in 2003, and the early chapters were suspiciously perfect—all showers of love and innocence until a major loss in the 5th standard hit me hard. Then came the COVID era, where I went completely off-script and spectacularly flubbed my entrance exams by choosing maximum laziness over the grind. Despite that self-sabotage, I managed to pivot: I graduated at 22 with a Computer Science Engineering degree, and I've finally found my passion, diving deep into the cool, chaotic fields of AI/ML and Data Science. My life isn't a clean, optimized algorithm; it's a beautifully messed-up, trial-and-error script that's just now getting to the truly exciting parts!"
    },
    {
        "id": "superpower",
        "question": "What's your #1 superpower?",
        "answer": "My initial thought was the ultimate power fantasy: the ability for my body to instantly generate a curative antibody for any illness, disease, toxin, or venom I encounter. While the supervillain potential (holding the world's health for ransom) is tempting, I'd ultimately choose a more principled path: the complete biological adaptability of Darwin from the X-Men. I wouldn't just be immune; I would evolve on the fly to survive any situation, making me the ultimate engine of rapid, perpetual self-improvement."
    },
    {
        "id": "growth_areas",
        "question": "What are the top 3 areas you'd like to grow in?",
        "answer": "I'm aiming to master the fundamentals first, but with a grand vision: Building Scalable Systems - I want to learn how to design platforms that won't immediately crash when they get real user traffic. I'm tired of building sandcastles; I want to design the whole beach, complete with load-balanced snack shacks! Professional Leadership - I need to move from group project leader to actual team guide—the person who can actually motivate a team without resorting to bribery (yet). Debugging Prowess - My goal is to become a code detective who can hunt down a production bug faster than my friends finds the free snacks from my bag. That means I need serious practice."
    },
    {
        "id": "misconception",
        "question": "What misconception do your coworkers have about you?",
        "answer": "People think I'm quiet, but I'm deeply collaborative once work begins. The initial quietness is just me observing and understanding the team dynamics and project context. Once I'm engaged, I'm fully invested in collaborative problem-solving and team success."
    },
    {
        "id": "boundaries",
        "question": "How do you push your boundaries and limits?",
        "answer": "I push my limits by treating my own education as a system stress test: I consistently sign up for learning projects that are slightly over my head—if I don't feel a mild sense of panic, I'm probably not learning fast enough. This approach keeps me in a constant state of growth and prevents complacency."
    }
]

SIMILARITY_THRESHOLD = 0.60

@st.cache_resource
def initialize_rag_system():
    """Initialize embedding model and FAISS index"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    questions = [item["question"] for item in KNOWLEDGE_BASE]
    answers = [item["answer"] for item in KNOWLEDGE_BASE]
    embeddings = model.encode(questions)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return model, index, questions, answers

def semantic_search(query, model, index, questions, answers, top_k=1):
    """Perform semantic search using RAG"""
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding.astype('float32'), top_k)
    best_idx = indices[0][0]
    similarity_score = 1 / (1 + distances[0][0])
    
    return {
        "question": questions[best_idx],
        "answer": answers[best_idx],
        "similarity": similarity_score,
        "index": best_idx
    }

def get_bot_response(user_input, conversation_history, model, index, questions, answers):
    """Generate response with FASTER model and SMART RAG"""
    try:
        if "OPENROUTER_API_KEY" not in st.secrets:
            return "⚠️ API key not configured. Please add OPENROUTER_API_KEY to Streamlit secrets.", None
        
        rag_result = semantic_search(user_input, model, index, questions, answers)
        
        context_messages = []
        if len(conversation_history) > 0:
            recent_history = conversation_history[-5:]
            for msg in recent_history:
                context_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        system_prompt = """You are Meet Pandya, a 22-year-old Computer Science Engineering graduate born in 2003. Respond naturally and conversationally, as if you ARE Meet Pandya sharing your thoughts, stories, and vibes in first person.

Core personality traits:
- Born in 2003, Computer Science Engineering graduate (age 22)
- Passionate about AI/ML and Data Science
- Self-aware with a great sense of humor
- Authentic and slightly self-deprecating
- Uses wit to discuss serious topics
- Growth-minded and constantly learning

Communication style:
- Natural and conversational
- Mix of professionalism and personality
- Don't be robotic - be human!
- Reference previous conversation when relevant
- Keep responses concise (2-3 paragraphs max)
- For greetings/small talk, respond naturally without forcing facts"""

        if rag_result['similarity'] >= SIMILARITY_THRESHOLD:
            user_prompt = f"""The user asked: "{user_input}"

Retrieved context (similarity: {rag_result['similarity']:.2%}):
Question: {rag_result['question']}
Answer: {rag_result['answer']}

Instructions:
1. Use the retrieved answer as your core information
2. Rephrase it naturally - don't copy-paste
3. Keep it conversational and authentic"""
        else:
            user_prompt = f"""The user said: "{user_input}"

This seems like a general question or greeting.

Instructions:
1. Respond naturally based on your personality
2. If it's a greeting, greet back warmly and offer to discuss your background
3. Maintain the conversational tone"""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(context_messages)
        messages.append({"role": "user", "content": user_prompt})
        
        # FASTER MODEL with shorter timeout
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://streamlit-interview-bot.app",
                "X-Title": "Interview Voice Bot"
            },
            json={
                "model": "nvidia/nemotron-nano-9b-v2:free",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 300
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            bot_response = data["choices"][0]["message"]["content"]
            return bot_response, rag_result if rag_result['similarity'] >= SIMILARITY_THRESHOLD else None
        elif response.status_code == 429:
            error_data = response.json() if response.content else {}
            error_msg = error_data.get("error", {}).get("message", "Rate limit reached")
            return f"⚠️ API Issue: {error_msg}. The free tier has limits. Please wait a moment or check your OpenRouter credits.", None
        else:
            error_data = response.json() if response.content else {}
            error_detail = error_data.get("error", {}).get("message", f"Status {response.status_code}")
            return f"⚠️ API Error: {error_detail}", None
        
    except requests.exceptions.Timeout:
        return "⚠️ Response timeout. The API is slow right now. Please try again.", None
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}", None

def text_to_speech(text):
    """Convert text to speech using gTTS"""
    try:
        if text.startswith("⚠️") or text.startswith("Sorry"):
            return None
        processed_text = text.replace("ML", "M L").replace("AI", "A I")
        tts = gTTS(text=processed_text, lang='en', tld='co.in', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.getvalue()
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None

# Initialize RAG system if not done yet
if not st.session_state.rag_initialized:
    with st.spinner("🧠 Initializing RAG system..."):
        st.session_state.model, st.session_state.index, st.session_state.questions, st.session_state.answers = initialize_rag_system()
    st.session_state.rag_initialized = True

# Header
st.markdown("""
<div class="main-header">
    <h1>🎤 RAG-Powered Interview Voice Bot</h1>
    <p>Semantic Search • Conversation Context • Natural Voice</p>
</div>
""", unsafe_allow_html=True)

# Stats
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="stats-box">
        <h3>{st.session_state.request_count}</h3>
        <p>API Calls</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="stats-box">
        <h3>{len(st.session_state.messages) // 2}</h3>
        <p>Conversations</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="stats-box">
        <h3>{len(KNOWLEDGE_BASE)}</h3>
        <p>Knowledge Items</p>
    </div>
    """, unsafe_allow_html=True)

# Info section
with st.expander("ℹ️ About This Bot"):
    st.markdown("""
    ### 🏗️ Technical Architecture
    
    **Smart RAG Implementation:**
    - Uses 60% similarity threshold
    - Below threshold = Natural LLM conversation
    - Above threshold = RAG-guided responses
    - NVIDIA Nemotron Nano 9B V2 (60s timeout)
    
    ### 📚 Sample Questions:
    - What's your life story?
    - Tell me about your superpower
    - What are your growth areas?
    - What misconception do people have about you?
    - How do you push your boundaries?
    """)

# Voice Input Section
st.markdown("### 🎤 Voice Input")
st.info("🎙️ Click the microphone button below and speak your question")

# Use streamlit-mic-recorder
audio_data = mic_recorder(
    start_prompt="🎙️ Start Recording",
    stop_prompt="⏹️ Stop Recording",
    just_once=False,
    use_container_width=True,
    key='recorder'
)

# Process audio recording with loop prevention
if audio_data and not st.session_state.processing_audio:
    # Create unique ID for this audio recording
    audio_id = str(hash(str(audio_data["bytes"])))
    
    # Only process if this is a NEW recording
    if audio_id != st.session_state.last_processed_audio_id:
        st.session_state.processing_audio = True
        st.session_state.last_processed_audio_id = audio_id
        
        st.success("✅ Recording received! Converting to text...")

        import tempfile
        import time
        
        base_url = "https://api.assemblyai.com"
        headers = {
            "authorization": st.secrets["ASSEMBLYAI_API_KEY"]
        }

        # Save recorded audio as a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_data["bytes"])
            tmp_path = tmp.name

        # Upload audio file to AssemblyAI
        with open(tmp_path, "rb") as f:
            upload_response = requests.post(
                base_url + "/v2/upload",
                headers=headers,
                data=f
            )
        audio_url = upload_response.json()["upload_url"]

        # Create a transcription request
        data = {"audio_url": audio_url, "speech_model": "universal"}
        url = base_url + "/v2/transcript"
        response = requests.post(url, json=data, headers=headers)
        transcript_id = response.json()['id']
        polling_endpoint = base_url + "/v2/transcript/" + transcript_id

        # Poll until transcription is complete
        with st.spinner("📝 Transcribing speech..."):
            while True:
                transcription_result = requests.get(polling_endpoint, headers=headers).json()
                if transcription_result['status'] == 'completed':
                    transcript_text = transcription_result['text']
                    st.success(f"📝 You said: {transcript_text}")
                    
                    # Check request limit BEFORE processing
                    if st.session_state.request_count >= 30:
                        st.error("⚠️ Demo limit reached (30 requests).")
                        st.session_state.processing_audio = False
                        break
                    
                    # Add user message
                    st.session_state.messages.append({"role": "user", "content": transcript_text})
                    
                    # Get bot response
                    with st.spinner("🤔 Thinking..."):
                        response, rag_info = get_bot_response(
                            transcript_text, 
                            st.session_state.messages,
                            st.session_state.model,
                            st.session_state.index,
                            st.session_state.questions,
                            st.session_state.answers
                        )
                        st.session_state.request_count += 1
                    
                    # Generate audio
                    audio_bytes = None
                    if response and not response.startswith("⚠️"):
                        with st.spinner("🔊 Generating voice..."):
                            audio_bytes = text_to_speech(response)
                    
                    # Add bot message
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "rag_info": rag_info,
                        "audio": audio_bytes
                    })
                    
                    # Reset processing flag
                    st.session_state.processing_audio = False
                    
                    # Rerun to display the conversation
                    st.rerun()
                    break
                elif transcription_result['status'] == 'error':
                    st.error(f"Transcription failed: {transcription_result['error']}")
                    st.session_state.processing_audio = False
                    break
                else:
                    time.sleep(3)

st.markdown("---")
st.markdown("### 💬 Chat Interface")

# Display chat history
for idx, message in enumerate(st.session_state.messages):
    with st.container():
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 Assistant:</strong> {message["content"]}
            """, unsafe_allow_html=True)
            
            if "rag_info" in message and message["rag_info"]:
                rag = message["rag_info"]
                st.markdown(f"""
                <div class="rag-info">
                    <strong>🔍 RAG Context Used:</strong> Matched "{rag['question']}" 
                    (Similarity: {rag['similarity']:.1%})
                </div>
                """, unsafe_allow_html=True)
            
            # AUTO-PLAY audio with option to pause
            if "audio" in message and message["audio"]:
                is_latest = (idx == len(st.session_state.messages) - 1)
                st.audio(message["audio"], format='audio/mp3', autoplay=is_latest)

# Chat input
if prompt := st.chat_input("Type your question here..."):
    if st.session_state.request_count >= 30:
        st.error("⚠️ Demo limit reached (30 requests).")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("🤔 Thinking..."):
        response, rag_info = get_bot_response(
            prompt, 
            st.session_state.messages,
            st.session_state.model,
            st.session_state.index,
            st.session_state.questions,
            st.session_state.answers
        )
        st.session_state.request_count += 1
    
    audio_bytes = None
    if response and not response.startswith("⚠️"):
        with st.spinner("🔊 Generating a reply..."):
            audio_bytes = text_to_speech(response)
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "rag_info": rag_info,
        "audio": audio_bytes
    })
    
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>Built for Personal Chat Demo</strong></p>
    <p>🏗️ Smart RAG • ⚡ NVIDIA Nemotron Nano 9B V2 • 🎤 Voice Ready • 🔊 Auto-Play</p>
    <p style="font-size: 0.85rem;">
        Faster model (60s timeout) • Auto-play audio • Mic support<br>
        <em>Note: Using free APIs (OpenRouter & gTTS). If you hit API limits or no voice plays, please wait a moment and try again.</em>
    </p>
</div>
""", unsafe_allow_html=True)
