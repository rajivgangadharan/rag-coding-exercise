import gradio as gr
import requests

API_URL = "http://localhost:8001"  # Change this if deployed elsewhere


def search_documents(query, top_k):
    payload = {"query": query, "top_k": int(top_k)}
    try:
        response = requests.post(f"{API_URL}/search", json=payload)
        response.raise_for_status()
        results = response.json()
        output = ""
        for i, doc in enumerate(results):
            output += f"### Result {i+1}:\n"
            output += f"**Content:** {doc['content'][:500]}...\n\n"
            output += f"**Metadata:** {doc['metadata']}\n"
            output += f"**Score:** {doc['score']:.4f}\n\n---\n\n"
        return output
    except Exception as e:
        return f"❌ Error: {str(e)}"


def upload_pdf(file):
    try:
        with open(file.name, "rb") as f:
            files = {"file": (file.name, f, "application/pdf")}
            response = requests.post(f"{API_URL}/documents/upload", files=files)
            response.raise_for_status()
            return f"✅ Uploaded: {file.name}"
    except Exception as e:
        return f"❌ Error uploading file: {str(e)}"


with gr.Blocks(title="🧠 Document Search Assistant") as demo:
    gr.Markdown("# 🧠 Document Search Interface")
    gr.Markdown("Submit a query to retrieve the most relevant documents.")

    with gr.Row():
        query = gr.Textbox(label="Query", placeholder="Enter your search query")
        top_k = gr.Slider(label="Top K", minimum=1, maximum=10, value=3, step=1)

    search_btn = gr.Button("🔍 Search")
    output_box = gr.Markdown()

    search_btn.click(fn=search_documents, inputs=[query, top_k], outputs=output_box)

    gr.Markdown("## 📄 Upload a PDF for indexing")
    uploader = gr.File(label="Upload PDF", file_types=[".pdf"])
    upload_output = gr.Textbox(label="Upload Status", interactive=False)

    uploader.change(fn=upload_pdf, inputs=uploader, outputs=upload_output)

if __name__ == "__main__":
    demo.launch()
