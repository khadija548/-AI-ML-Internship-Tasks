import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import time

# ---------------------------
# 1. Professional Page Config & Styling
# ---------------------------
st.set_page_config(
    page_title="Nexus AI | Knowledge Assistant",
    page_icon="🌌",
    layout="wide"
)

# Custom CSS for a modern, professional look
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    .stChatMessage {
        border-radius: 15px;
        margin-bottom: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stSpinner > div > div {
        border-top-color: #38bdf8 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------
# 2. Optimized Resource Loading (Caching)
# ---------------------------
@st.cache_resource
def load_resources():
    # Load Model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Load Knowledge Base
    with open("documents/knowledge.txt", "r", encoding="utf-8") as f:
        knowledge = [line.strip() for line in f.readlines() if line.strip()]
    
    # Create Vector Index
    embeddings = model.encode(knowledge)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    return model, knowledge, index

model, knowledge, index = load_resources()

# ---------------------------
# 3. Sidebar - Pro Features
# ---------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712139.png", width=80)
    st.title("Nexus Control")
    st.info("Currently indexing: **{}** documents".format(len(knowledge)))
    
    st.divider()
    confidence_threshold = st.slider("Match Accuracy Threshold", 0.0, 2.0, 1.2)
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------
# 4. Chat Interface Logic
# ---------------------------
st.title("🌌 Nexus AI")
st.caption("Advanced RAG Engine v2.0 • Powered by FAISS")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages with professional avatars
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🌌"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("How can I assist you today?"):
    
    # Add User Message
    st.chat_message("user", avatar="👤").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="🌌"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()
        
        with st.spinner("Synthesizing context..."):
            # RAG Retrieval
            query_embedding = model.encode([prompt])
            distances, indices = index.search(np.array(query_embedding), k=3)
            
            # Filtering by confidence threshold
            valid_indices = [idx for i, idx in enumerate(indices[0]) if distances[0][i] < confidence_threshold]
            
            if not valid_indices:
                full_response = "I'm sorry, I couldn't find specific information in my knowledge base to answer that accurately."
            else:
                # Simulated streaming for professional feel
                context_chunks = [knowledge[i] for i in valid_indices]
                full_response = " ".join(context_chunks)
            
            # Professional "Typewriter" effect
            displayed_text = ""
            for char in full_response:
                displayed_text += char
                response_placeholder.markdown(displayed_text + "▌")
                time.sleep(0.01) # Adjust speed here
            
            response_placeholder.markdown(full_response)
            
            # Show "Sources" in a professional expander
            if valid_indices:
                with st.expander("🔍 View Sources"):
                    for i, idx in enumerate(valid_indices):
                        st.write(f"**Source {i+1}:** {knowledge[idx]}")

    st.session_state.messages.append({"role": "assistant", "content": full_response})