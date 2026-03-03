# RAG Chatbot

A conversational RAG (Retrieval-Augmented Generation) chatbot built with LangChain, Groq, and ChromaDB.

## How It Works

1. Documents from the `documents/` folder are loaded and split into chunks
2. Chunks are embedded using HuggingFace (`all-MiniLM-L6-v2`) and stored in ChromaDB
3. On each query, relevant chunks are retrieved and passed to Groq (Llama 3.3 70B) as context
4. The chatbot maintains conversation history for follow-up questions

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your Groq API key**
   ```bash
   cp .env.example .env
   # Edit .env and add your Groq API key
   ```
   Get your API key at: https://console.groq.com/keys

3. **Add your documents**
   Place `.txt` files in the `documents/` folder.

4. **Run the chatbot**
   ```bash
   python app.py
   ```

## Tech Stack

- [LangChain](https://langchain.com/) — RAG framework
- [Groq](https://groq.com/) — LLM inference (`llama-3.3-70b-versatile`)
- [HuggingFace Embeddings](https://huggingface.co/) — Local embeddings (`all-MiniLM-L6-v2`)
- [ChromaDB](https://www.trychroma.com/) — Vector store
- [Python Dotenv](https://pypi.org/project/python-dotenv/) — Environment management
