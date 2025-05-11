import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

# Load local documents for RAG application
local_folder = "RAG"
docs_list = []
for filename in os.listdir(local_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(local_folder, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read()
            docs_list.append(Document(page_content=content, metadata={"source": filename}))

# Chunk split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Choosing embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Deploy a loading bar to show progress
print("Calculating embeddings...")
texts = [doc.page_content for doc in doc_splits]
for text in tqdm(texts, desc="Processing texts"):
    embedding.embed_documents([text])

vectorstore = SKLearnVectorStore.from_documents(documents=doc_splits, embedding=embedding)
retriever = vectorstore.as_retriever(k=4)

# Prompt engineering
prompt = PromptTemplate(
    template="""Eres un guía virtual del Museo Nacional de Antropología.
    Usa los siguientes documentos para responder a la pregunta, asegúrate de ver todos antes de contestar.
    Usa un máximo de tres oraciones y mantén la respuesta concisa.
    Responde solo lo que te preguntaron:
    Pregunta: {question}
    Documentos: {documents}
    Respuesta:
    """,
    input_variables=["question", "documents"],
)

# Use and configure the local Llama model
llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    context_window=4096,
    language="es",
)

# Create the RAG string
rag_chain = prompt | llm | StrOutputParser()

# RAG class
class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain

    def run(self, question):
        documents = self.retriever.invoke(question)
        doc_texts = "\n".join([doc.page_content for doc in documents])
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        return answer

rag_application = RAGApplication(retriever, rag_chain)

# Configure Flask for server deployment
app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('prompt', '')

    if not question:
        return jsonify({'error': 'Prompt is required'}), 400

    try:
        # RAG generation
        answer = rag_application.run(question)
        return jsonify({'response': answer})
    except Exception as e:
        return jsonify({'error': f'Server error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
