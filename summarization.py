import logging
from pathlib import Path
from typing import List, Literal, TypedDict

from langchain.chains.combine_documents.reduce import split_list_of_docs
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_ollama.chat_models import ChatOllama
from langchain_text_splitters import CharacterTextSplitter
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langchain_community.document_loaders import PyPDFLoader
from openai import base_url

from constants import LOG_FILE, OLLAMA_BASE_URL
from logging_setup import setup_logger

logger = logging.getLogger("gradio_log")

class OverallState(TypedDict):
    contents: List[str]
    summaries: List[str]
    collapsed_summaries: List[Document]
    final_summary: str

class SummaryState(TypedDict):
    content: str

class State(TypedDict):
    contents: List[str]
    index: int
    summary: str

async def summarize_pdf(pdf_file, summarization_type, ollama_model, chunk_size, max_token):
    if pdf_file is None:
        return "Please select a file before proceeding.", ""
    if ollama_model is None:
        return "Please select a model before proceeding.", ""
    llm = ChatOllama(model=ollama_model, base_url=OLLAMA_BASE_URL, temperature=0)
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=0
    )
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()
    logger.info("PDF file has successfully loaded.")
    Path(LOG_FILE).touch()
    split_docs = text_splitter.split_documents(docs)
    logger.info(f"Generated {len(split_docs)} documents.")
    Path(LOG_FILE).touch()

    if summarization_type == "Map-Reduce":
        summary, status = await map_reduce_summarization(split_docs, llm, max_token)
    else:
        summary, status = await iterative_refinement_summarization(split_docs, llm)

    return summary, status

async def map_reduce_summarization(split_docs, llm, max_token):
    logger.info("Starting summarization using Map-Reduce ...")
    Path(LOG_FILE).touch()
    map_prompt = PromptTemplate.from_template("Write a concise summary of the following:\n\n{context}")
    reduce_prompt = PromptTemplate.from_template("""
    The following is a set of summaries:
    {docs}
    Take these and distill it into a final, consolidated summary
    of the main themes.
    """)
    map_chain = map_prompt | llm | StrOutputParser()
    reduce_chain = reduce_prompt | llm | StrOutputParser()

    def length_function(documents: List[Document]) -> int:
        return sum(len(doc.page_content.split()) for doc in documents)

    async def generate_summary(state: SummaryState):
        response = await map_chain.ainvoke({"context": state["content"]})
        return {"summaries": [response]}

    def map_summaries(state: OverallState):
        return [
            Send("generate_summary", {"content": content}) for content in state["contents"]
        ]

    def collect_summaries(state: OverallState):
        return {
            "collapsed_summaries": [Document(page_content=summary) for summary in state["summaries"]]
        }

    async def collapse_summaries(state: OverallState):
        doc_lists = split_list_of_docs(
            state["collapsed_summaries"], length_function, max_token
        )
        results = []
        for doc_list in doc_lists:
            docs_content = "\n".join([doc.page_content for doc in doc_list])
            response = await reduce_chain.ainvoke({"docs": docs_content})
            results.append(Document(page_content=response))
        return {"collapsed_summaries": results}

    def should_collapse(state: OverallState) -> Literal["collapse_summaries", "generate_final_summary"]:
        num_tokens = length_function(state["collapsed_summaries"])
        if num_tokens > max_token:
            return "collapse_summaries"
        else:
            return "generate_final_summary"

    async def generate_final_summary(state: OverallState):
        docs_content = "\n".join([doc.page_content for doc in state["collapsed_summaries"]])
        response = await reduce_chain.ainvoke({"docs": docs_content})
        return {"final_summary": response}

    graph = StateGraph(OverallState)
    graph.add_node("generate_summary", generate_summary)
    graph.add_node("collect_summaries", collect_summaries)
    graph.add_node("collapse_summaries", collapse_summaries)
    graph.add_node("generate_final_summary", generate_final_summary)

    graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
    graph.add_edge("generate_summary", "collect_summaries")
    graph.add_conditional_edges("collect_summaries", should_collapse)
    graph.add_conditional_edges("collapse_summaries", should_collapse)
    graph.add_edge("generate_final_summary", END)

    app = graph.compile()

    summary = ""
    async for step in app.astream(
        {"contents": [doc.page_content for doc in split_docs]},
        {"recursion_limit": 10},
    ):
        logger.debug(list(step.keys()))
        Path(LOG_FILE).touch()
        if "generate_final_summary" in step:
            summary = step["generate_final_summary"]["final_summary"]

    return summary, "Process has been completed."

async def iterative_refinement_summarization(docs, llm):
    logger.info("Starting summarization using Iterative Refinement ...")
    Path(LOG_FILE).touch()
    summarize_prompt = ChatPromptTemplate(
        [("human", "Write a concise summary of the following: {context}")]
    )
    initial_summary_chain = summarize_prompt | llm | StrOutputParser()

    refine_template = """
    Produce a final summary.

    Existing summary up to this point:
    {existing_answer}

    New context:
    ------------
    {context}
    ------------

    Given the new context, refine the original summary.
    """
    refine_prompt = ChatPromptTemplate([("human", refine_template)])
    refine_summary_chain = refine_prompt | llm | StrOutputParser()

    async def generate_initial_summary(state: State, config: RunnableConfig):
        summary = await initial_summary_chain.ainvoke(
            state["contents"][0],
            config,
        )
        return {"summary": summary, "index": 1}

    async def refine_summary(state: State, config: RunnableConfig):
        content = state["contents"][state["index"]]
        summary = await refine_summary_chain.ainvoke(
            {"existing_answer": state["summary"], "context": content},
            config,
        )
        return {"summary": summary, "index": state["index"] + 1}

    def should_refine(state: State) -> Literal["refine_summary", END]:
        if state["index"] >= len(state["contents"]):
            return END
        else:
            return "refine_summary"

    graph = StateGraph(State)
    graph.add_node("generate_initial_summary", generate_initial_summary)
    graph.add_node("refine_summary", refine_summary)
    graph.add_edge(START, "generate_initial_summary")
    graph.add_conditional_edges("generate_initial_summary", should_refine)
    graph.add_conditional_edges("refine_summary", should_refine)

    app = graph.compile()

    summary = ""
    async for step in app.astream(
        {"contents": [doc.page_content for doc in docs]},
        stream_mode="values",
    ):
        if summary_step := step.get("summary"):
            summary = summary_step
            logger.debug("Updated summary:")
            logger.debug(summary)
            Path(LOG_FILE).touch()

    return summary, "Process has been completed."