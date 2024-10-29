import gradio as gr
from gradio_log import Log

from constants import LOG_FILE
from logging_setup import setup_logger
from ollama_utils import get_ollama_model_list
from summarization import summarize_pdf

# Set up logger
logger = setup_logger(LOG_FILE)

# Create Gradio App
with gr.Blocks() as demo:
    gr.Markdown("### [PDF Summarization Tool](https://github.com/samfallahian/large_pdf_summarization)")
    with gr.Row():
        pdf_upload = gr.File(
            label="Upload PDF File", type="filepath", height=100, file_types=['.pdf']
        )
    with gr.Row():
        summary_output = gr.Textbox(label="Status", value="")
    with gr.Row():
        summary_output_m = gr.Markdown("## Your summary will appear here.")
    with gr.Row():
        with gr.Column(scale=1):
            summarization_type = gr.Radio(
                ["Map-Reduce", "Iterative Refinement"],
                value="Map-Reduce",
                label="Select Summarization Type",
                info="Choose summarization method.",
            )
            ollama_model = gr.Dropdown(
                get_ollama_model_list(),
                label="Ollama Model",
                info="Select Ollama Model",
            )
        with gr.Column(scale=1):
            chunk_size = gr.Number(
                label="Document Chunk Size", value=1000, precision=0
            )
            max_token = gr.Number(
                label="Maximum Token Length", value=1000, precision=0
            )
    summarize_button = gr.Button("Summarize")

    summarize_button.click(
        fn=summarize_pdf,
        inputs=[
            pdf_upload,
            summarization_type,
            ollama_model,
            chunk_size,
            max_token,
        ],
        outputs=[summary_output_m, summary_output],
    )

    Log(LOG_FILE, dark=True)

if __name__ == "__main__":
    demo.launch(show_error=True)