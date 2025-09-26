import gradio as gr
from retriever import answer
from ingest import ingest_pdfs, ingest_urls

# Preload data into ChromaDB
ingest_pdfs([
    "data/Demo/RAGInfo1.pdf",
    "data/Demo/RAGInfo2.pdf",
    "data/Demo/RAGInfo3.pdf",
    "data/Demo/MLInfo1.pdf",
    "data/Demo/AIInfo1.pdf",
])
ingest_urls([
    "https://atos.net/wp-content/uploads/2024/08/atos-retrieval-augmented-generation-ai-whitepaper.pdf",
    "https://www.tutorialspoint.com/machine_learning/machine_learning_tutorial.pdf",
    "https://www.oajaiml.com/uploads/archivepdf/63501191.pdf",
    "https://medium.com/@Hiadore/building-a-simple-rag-question-answering-system-from-your-own-pdf-for-free-no-framework-used-part-bd60bf370c52",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://jsonplaceholder.typicode.com/todos/1",
    "https://huggingface.co/api/models/bert-base-uncased"
])

def rag_demo(query):
    result = answer(query)
    return result["answer"]

demo = gr.Interface(
    fn=rag_demo,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question..."),
    outputs=gr.Textbox(lines=15, label="Answer", placeholder="The answer will appear here..."),
    title="Clarifile Demo",
    description="A lightweight RAG system for PDFs and URLs, powered by ChromaDB + Llama 3.1."
)

if __name__ == "__main__":
    demo.launch(share=True)
