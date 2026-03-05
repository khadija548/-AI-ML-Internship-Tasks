import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import datetime

# --- 1. CONFIG & BRANDING ---
PLUS = "\u2795"

st.set_page_config(
    page_title="HealthAI | Medical Assistant",
    page_icon=PLUS,
    layout="wide"
)

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button { border-radius: 8px; border: 1px solid #ff4b4b; }
    .stDownloadButton>button { background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# --- 3. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history_list" not in st.session_state:
    st.session_state.chat_history_list = [] # For the sidebar log

# --- 4. SIDEBAR (The Dashboard) ---
with st.sidebar:
    st.title(f"{PLUS} HealthAI Dashboard")
    
    # NEW: Chat History Log
    st.subheader("📜 Chat Sessions")
    if st.session_state.chat_history_list:
        for i, history in enumerate(st.session_state.chat_history_list):
            st.info(f"{i+1}. {history[:25]}...")
    else:
        st.write("No history yet.")

    st.divider()

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button(" Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.session_state.messages:
            transcript = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
            st.download_button("💾 Save", transcript, f"HealthAI_Transcript_{datetime.date.today()}.txt", use_container_width=True)

    st.divider()
    
    # Disclaimer & Tech Stack
    st.error( "**⚠️ Medical Disclaimer**\n This chatbot provides general health information and should not replace professional medical advice.\nAlways consult a qualified healthcare professional for medical advice")
    with st.expander(" Tech Stack"):
        st.write("- **Frontend:** Streamlit")
        st.write("- **Model:** Google FLAN-T5 (LLM)")
        st.write("- **Library:** HuggingFace Transformers")

# --- 5. MAIN UI ---
st.title(f"{PLUS} AI Medical Assistant")
st.markdown("*Empowering health literacy with Large Language Models.*")

# Example Questions
if not st.session_state.messages:
    st.write("### How can I assist you today?")
    examples = ["What are common symptoms of the flu?", "How to treat a minor burn?", "Explain Vitamin D deficiency"]
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        if cols[i].button(ex, use_container_width=True):
            st.session_state.active_query = ex

# Display Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 6. CHAT LOGIC ---
user_input = st.chat_input("Ask a medical or health question...")

if "active_query" in st.session_state:
    user_input = st.session_state.pop("active_query")

if user_input:
    # Save to Session Log Sidebar
    st.session_state.chat_history_list.append(user_input)
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Emergency Trigger
    danger_words = ["kill myself", "suicide", "self harm", "chest pain", "heart attack", "overdose"]
    if any(key in user_input.lower() for key in danger_words):
        response = """
        ### 🆘 Help is Available. You are not alone.
        * **Call or Text 988** (Suicide & Crisis Lifeline). Available 24/7.
        * **Emergency:** Call **911** or go to the nearest ER immediately.
        * **Reach Out:** Please talk to a friend, family member, or doctor right now.
        """
        # FIXED: Show the emergency response immediately in the chat
        with st.chat_message("assistant"):
            st.markdown(response)
    else:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing medical database..."):
                prompt = (f"Provide a thorough, detailed medical explanation for: {user_input}. "
                          f"Structure: Definition, Detailed Symptoms, Causes, and Management.")
                
                inputs = tokenizer(prompt, return_tensors="pt")
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=512, 
                    min_length=150, 
                    repetition_penalty=1.6, 
                    do_sample=True,
                    top_p=0.9
                )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Polishing output
                response = response.replace("Symptoms:", "\n\n**Symptoms:**\n")
                response = response.replace("Causes:", "\n\n**Causes:**\n")
                st.markdown(response)

    # Save to session history so it stays on screen after rerun
    st.session_state.messages.append({"role": "assistant", "content": response})