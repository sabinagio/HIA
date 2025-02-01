from typing import Annotated, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from src.agents import (
    query_understanding,
    rag,
    response_quality
)

# Load environment variables
load_dotenv()


# Define overall graph state - now more similar to the example
class ConversationState(TypedDict):
    """State for the entire conversation graph"""
    messages: Annotated[list, add_messages]  # Conversation history
    query: Optional[str] # Current user query
    location: Optional[str] # Optional location context
    analysis: Optional[dict] # Optional[query_understanding.QueryAnalysis]
    # Analysis from query understanding, which has all the context we need
    # To keep consistency in conversation (language, emotional state, extracted_entities,
    # domains, query_type, etc.)
    response: Optional[rag.RAGOutput]
    quality_review: Optional[str]


def build_conversation_graph():
    """Build the main conversation workflow graph"""
    workflow = StateGraph(ConversationState)

    # Add agent nodes
    workflow.add_node("query_understanding", query_understanding.query_understanding_node)
    workflow.add_node("rag", rag.rag_node)
    workflow.add_node("response_quality", response_quality.response_quality_node)

    # Simplified routing nodes with chat-like responses
    def await_clarification_node(state):
        message = state["messages"][-1]["content"] # message already defined in qu agent
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": message
                }
            ]
        }

    def emergency_node(state):
        whatsapp_number = "environment variable very secret"
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        f"""  
                        This seems urgent and like you need immediate assistance. Please contact the Red Cross directly at this number {whatsapp_number}  
                        to get help immediately.  
                        For any medical emergency please contact 112.  
                        """
                    )
                }
            ]
        }

    workflow.add_node("await_clarification", await_clarification_node)
    workflow.add_node("emergency", emergency_node)

    # Routing logic
    def route_by_query_type(state):
        analysis = state["analysis"]
        if analysis.query_type == "clear":
            return "rag"
        elif analysis.query_type == "emergency":
            return "emergency"
        else:
            return "await_clarification"

    # Add edges
    workflow.add_edge(START, "query_understanding")
    workflow.add_conditional_edges(
        "query_understanding",
        route_by_query_type,
        {
            "rag": "rag",
            "emergency": "emergency",
            "await_clarification": "await_clarification"
        }
    )
    workflow.add_edge("rag", "response_quality")
    workflow.add_edge("emergency", END)
    workflow.add_edge("await_clarification", END)
    workflow.add_edge("response_quality", END)

    return workflow.compile()


# Initialize conversation graph
graph_app = build_conversation_graph()


def stream_graph_updates(user_input: str, location: Optional[str] = None):
    """Stream updates from the graph in a chat-like format"""
    state = {
        "messages": [{"role": "user", "content": user_input}],
        "query": user_input,
        "location": location
    }

    for event in graph_app.stream(state):
        for value in event.values():
            if "messages" in value and value["messages"]:
                print("Assistant:", value["messages"][-1]["content"])


if __name__ == "__main__":
    print("Red Cross Virtual Assistant (type 'quit' to exit)")

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Thank you for using the Red Cross Virtual Assistant. Goodbye!")
                break

            stream_graph_updates(user_input)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again or contact support if the issue persists.")