# Context-Aware RAG Chatbot

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** that answers user questions using a custom knowledge base.

The chatbot retrieves relevant information using vector similarity search and provides responses through a conversational interface.

---

## Project Overview

Traditional chatbots rely only on pre-trained models.  
This project improves accuracy by retrieving relevant information from a **knowledge base** before generating responses.

The chatbot uses **semantic embeddings and vector search** to find the most relevant information for each question.

---

## Technologies Used

- Python
- Streamlit
- Sentence Transformers
- FAISS (Vector Similarity Search)
- Retrieval-Augmented Generation (RAG)

---

## Features

- Semantic search using embeddings
- Vector database retrieval
- Chat-style interface
- Custom knowledge base
- Real-time responses

---

## How It Works

1. The knowledge base is stored in a text document.
2. Each sentence is converted into vector embeddings.
3. The FAISS vector index stores these embeddings.
4. When a user asks a question:
   - The question is converted into an embedding
   - FAISS searches for the most similar knowledge entry
   - The chatbot returns the best answer.

---

## Project Structure
