import gradio as gr
from retriever import answer

def rag_demo(query):
    result = answer(query)
    return result["answer"]

demo = gr.Interface(
    fn=rag_demo,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question..."),
    outputs="text",
    title="📄 Clarifile Demo",
    description="A lightweight RAG system for PDFs and URLs, powered by ChromaDB + Llama 3.1."
)

if __name__ == "__main__":
    demo.launch()
