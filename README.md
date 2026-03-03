# RAG Chatbot

A conversational RAG (Retrieval-Augmented Generation) chatbot built with LangChain and ChromaDB.

## How It Works

1. Documents from the `documents/` folder are loaded and split into chunks
2. Chunks are embedded using OpenAI Embeddings and stored in ChromaDB
3. On each query, relevant chunks are retrieved and passed to GPT-3.5 as context
4. The chatbot maintains conversation history for follow-up questions

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your OpenAI API key**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

3. **Add your documents**
   Place `.txt` files in the `documents/` folder.

4. **Run the chatbot**
   ```bash
   python app.py
   ```

## Tech Stack

- [LangChain](https://langchain.com/) — RAG framework
- [ChromaDB](https://www.trychroma.com/) — Vector store
- [OpenAI](https://openai.com/) — Embeddings & LLM (GPT-3.5-turbo)
- [Python Dotenv](https://pypi.org/project/python-dotenv/) — Environment management
