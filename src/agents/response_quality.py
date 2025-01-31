# Ensuring there is the source and date of the information
# Flow: not good enough... ask for clarification / say there is no information

import os

from langgraph.graph import StateGraph
from langchain_anthropic import ChatAnthropic
from langchain.schema import AIMessage
from rag import RAGState 
from langgraph.graph import StateGraph, END




def response_quality_node(state: RAGState):
    """Evaluates the quality of the response and either shows it to the user or asks for further clarification."""
    # Extract info from RAGOutput
    text_output = state["response"].text
    completeness_score = state["response"].metadata.completeness_score
    confidence_score = state["response"].metadata.confidence_score

    # Review confidence score
    if completeness_score < 0.8 and confidence_score < 0.8:
        return """I need more context to answer your question. Could you... X/Y/Z"""
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

    system_prompt = f"""You are an expert query analyzer for a Red Cross virtual assistant.
    Recognizing the inherent dignity of every individual and the diverse communities you serve is incredibly important.
    This is why you need to check whether or not this output follows this inclusive language guide. 
    
    'Avoid' are words that should not be in your output and 'Preferred Terms' are the words you should use if you encounter
    a word to be avoided. These are available for you in pairs, starting with the term to search for and Avoid, and ending
    with the Preferred Terms that you need to replace. You should choose only one of the preferred terms depending on the context.
    '.etc' denotes any other words that are similar to the ones previously written in the same enumeration. The 'Reasoning' is included
    to help with distinguishing between instances where the avoided words might be used with a different meaning.

    {os.getenv("COMMUNICATION_GUIDELINES")}
    """
    # Get structured analysis from LLM
    reviewed_output = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "agent", "content": f"Query: {text_output}"}
        ]
    )

    print(reviewed_output)
    
    return reviewed_output

# Create graph
workflow = StateGraph(RAGState)

workflow.add_node("response_quality", response_quality_node)
workflow.add_edge(END, "response_quality")

graph = workflow.compile()
