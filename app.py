from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()


def load_documents():
    loader = DirectoryLoader("documents/", glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)


def build_chain():
    print("Loading documents and building vector store...")
    chunks = load_documents()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    chain = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)
    return chain


def main():
    print("=" * 40)
    print("    RAG Chatbot with Groq + LangChain")
    print("=" * 40)
    print("Type 'quit' to exit.\n")

    chain = build_chain()
    print("Ready! Ask me anything based on the documents.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if not query:
            continue
        result = chain({"question": query})
        print(f"Bot: {result['answer']}\n")


if __name__ == "__main__":
    main()
