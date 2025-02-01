# Tutorial available on Streamlit:
# https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps
from dotenv import load_dotenv
load_dotenv()

from typing import List, Dict
import os
import streamlit as st

from typing import Annotated, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
# from slowapi import Limiter
# from slowapi.util import get_remote_address
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from src.agents import (
    query_understanding,
    rag,
    response_quality
)

st.title("Helpful Information as Aid")

# Define overall graph state
class ConversationState(TypedDict):
    """State for the entire conversation graph"""
    messages: Annotated[list, add_messages]  # Conversation history
    query: str  # Current user query
    location: Optional[str]  # Optional location context
    analysis: Optional[query_understanding.QueryAnalysis]
    # Analysis from query understanding, which has all the context we need
    # To keep consistency in conversation (language, emotional state, extracted_entities,
    # domains, query_type, etc.)
    initial_response: Optional[rag.RAGOutput]  # Response from RAG
    final_response: Optional[str]  # Quality review feedback


def build_conversation_graph():
    """Build the main conversation workflow graph"""

    # Initialize graph with ConversationState
    workflow = StateGraph(ConversationState)

    # workflow.set_entry_point("handle_greeting")
    # Like David said, to have the bot start the conversation

    # Add all agent nodes
    workflow.add_node("query_understanding", query_understanding.query_understanding_node)
    workflow.add_node("rag", rag.rag_node)
    workflow.add_node("response_quality", response_quality.response_quality_node)

    # Add simple routing nodes
    def await_clarification_node(state):
        """Returns clarification request with topic options"""
        print("current state:", state)
        message = state["messages"][-1].content
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": message  # Pass through clarification message
                }
            ]
        }

    def emergency_node(state):
        """Returns emergency contact information"""
        whatsapp_number = "environment variable very secret"
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"""  
                        This seems urgent and like you need immediate assistance. Please contact the Red Cross directly at this number {whatsapp_number}  
                        to get help immediately.  
                        For any medical emergency please contact 112.  
                        """
                }
            ]
        }

    workflow.add_node("await_clarification", await_clarification_node)
    workflow.add_node("emergency", emergency_node)

    # Define routing logic which is all based on query understanding output
    def route_by_query_type(state):
        """
        Routes to appropriate node based on query analysis
        In query_understanding QueryAnalysis.query_type Literal["clear", "needs_clarification", "emergency"]
        """
        analysis = state["analysis"]

        # Route based on query type
        if analysis["query_type"] == "clear":
            return "rag"
        elif analysis["query_type"] == "emergency":
            return "emergency"
        else:
            return "await_clarification"

    # Add edges
    workflow.add_edge(START, "query_understanding")

    # Add conditional edges from query understanding
    workflow.add_conditional_edges(
        "query_understanding",
        route_by_query_type,
        {
            "rag": "rag",
            "emergency": "emergency",
            "await_clarification": "await_clarification"
        }
    )

    # Connect RAG to response quality - normal flow
    workflow.add_edge("rag", "response_quality")

    # Somewhere here would be the base agent

    # All other nodes go to END - our multiple endings
    workflow.add_edge("emergency", END)
    workflow.add_edge("await_clarification", END)
    workflow.add_edge("response_quality", END)

    return workflow.compile()

class ChatInput(BaseModel):
    message: str
    history: List[Dict[str, str | None]] = None
    location: Optional[str] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str


# Initialize conversation graph
conversation_graph = build_conversation_graph()

def chat(chat_input: ChatInput) -> ChatResponse:
    """Handle chat requests"""

    # Initialize state for this conversation turn
    initial_state = {
        "messages": chat_input.history,  # Conversation history
        "query": chat_input.message,
        "location": chat_input.location,
        "analysis": None,  # For query understanding output
        "initial_response": None,  # For RAG output
        "final_response": None  # For response quality output
    }

    print(f"Initial state: {initial_state}")

    try:
        # Process through agent graph
        result = conversation_graph.invoke(initial_state)
        print("result", result)

        # Extract final response
        if "final_response" in result:
            response_text = result["final_response"]["text"]
        else:
            # Fallback to last message if no final_response
            response_text = result["messages"][-1].content
            print("response_text", response_text)

        return ChatResponse(response=response_text)

    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if "messages" not in st.session_state:
    st.session_state.messages = []

location = st.text_input("Your location (optional):", key="location_input")


if prompt := st.chat_input("How can I help you today?"):

    current_input = ChatInput(
        message=prompt,
        history=st.session_state.messages,
        location=location if location else None
    )
    st.session_state.messages.append({"role": "user", "content": prompt})
    result = chat(chat_input=current_input)
    st.session_state.messages.append({"role": "assistant", "content": result.response})

# Function to display messages in the chat
def display_chat_messages() -> None:
    """Display the conversation history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Display chat messages from history
display_chat_messages()
