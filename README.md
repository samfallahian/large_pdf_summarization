# Large PDF Summarization with Open-Source LLMs

This project provides an efficient solution for summarizing large PDF files using open-source LLMs, powered by [Ollama](https://ollama.com/), [Langchain](https://www.langchain.com/), [LangGraph](https://github.com/langchain-ai/langgraph) for robust RAG orchestration. The solution leverages [Gradio 5](https://www.gradio.app/) for a user-friendly interface and [gradio-log](https://github.com/louis-she/gradio-log) for process logging.

## Features

- **Map-Reduce Summarization**: Parallelized processing for handling lengthy texts.
- **Iterative Refinement**: Enhanced accuracy through iterative summarization steps.
- **Local Model Serving**: Uses Ollama to deploy models locally on your infrastructure.

## Requirements

1. **Install Ollama**  
   - Follow instructions at [Ollama](https://ollama.com/)

2. **Clone the Repository**
   ```bash
   git clone git@github.com:samfallahian/large_pdf_summarization.git
   
3. **Set Up Python Environment**
   ```bash
   conda create --name large_pdf_summarization python=3.11
   conda activate large_pdf_summarization
   
4. **Install PyTorch in venv**  
   - Refer to [Pytorch](https://pytorch.org/get-started/locally/)

3. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   

## Configuration

You can set your local server connection values and other constants in constants.py to customize your setup according to your server and environment needs.

## Running the Project

1. **Navigate to the source code folder:**  
   ```bash
   cd large_pdf_summarization
   
2. **Run the application and follow the link:**  
   ```bash
   python main.py
   
3. **Run the application:**  
   ```bash
   python main.py
