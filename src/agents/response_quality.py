# Ensuring there is the source and date of the information
# Flow: not good enough... ask for clarification / say there is no information

import os
import json
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph
from langchain_anthropic import ChatAnthropic
from langchain.schema import AIMessage
from src.agents.rag import RAGState 
from langgraph.graph import StateGraph, END, START

COMM_GUIDELINES = json.load(open('data/comms.json', 'rb'))["comms"]

class ConfigError(Exception):
    """An exception class for configuration errors."""
    def __init__(self, message):
        super().__init__(message)

def response_quality_node(state: RAGState):
    """Evaluates the quality of the response and either shows it to the user or asks for further clarification."""
    # Extract info from RAGOutput
    text_output = state["response"].text
    completeness_score = state["response"].metadata.completeness_score
    confidence_score = state["response"].metadata.confidence_score

    # Review confidence score
    if completeness_score < 0.8 and confidence_score < 0.8:
        return """I need more context to answer your question. Could you provide more information about your situation?"""
    if completeness_score < 0.8:
        return f"""This is a quick overview: {text_output}.\n Do you need more detail? """
    if confidence_score < 0.8:
        return """"""

    # Review alignment with RedCross tone using Claude
    llm = ChatAnthropic(
        model="claude-3-5-haiku-20241022",
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    if not COMM_GUIDELINES:
        raise ConfigError("There are no communication guidelines provided in the data folder.")

    system_prompt = f"""You are a response quality assistant for a Red Cross virtual assistant.
    You need to check whether or not the assistant output follows the INCLUSIVE LANGUAGE GUIDELINE.
    To do this, you need to see if any of the terms that need to be avoided are present and, if needed, replace them with one of the preferred terms provided for each category.
    If no terms that need to be avoided are present in the text, you should output the assistant query as you received it.
    
    ABOUT THE INCLUSIVE LANGUAGE GUIDELINE:
    'Avoid' are words that should not be in your output and 'Preferred Terms' are the words you should use if you encounter
    a word to be avoided. These are available for you in pairs, starting with the term to search for and Avoid, and ending
    with the Preferred Terms that you need to replace. You should choose only one of the preferred terms depending on the context.
    '.etc' denotes any other words that are similar to the ones previously written in the same enumeration. The 'Reasoning' is included
    to help with distinguishing between instances where the avoided words might be used with a different meaning.

    INCLUSIVE LANGUAGE GUIDELINE:
    {COMM_GUIDELINES}
    """
    # Get structured analysis from LLM
    if not text_output:
        raise ValueError("There was no response to be reviwed from the RAG agent")

    reviewed_output = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": f"Query: {text_output}"}
        ]
    )

    result = {
        "response": reviewed_output,
        "confidence_score": confidence_score,
        "completeness_score": completeness_score,
        "approved": True,
    }
    print("reviewed result:", result)
    return result

# Create graph
workflow = StateGraph(RAGState)

workflow.add_node("response_quality", response_quality_node)
workflow.add_edge(START, "response_quality")

graph = workflow.compile()
