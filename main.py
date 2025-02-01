from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from src.agents import query_understanding, feedback, rag, emergency_response, context_management, response_quality

app = FastAPI()


class ChatInput(BaseModel):
    message: str
    location: Optional[str] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    emergency: bool = False
    needs_clarification: bool = False
    clarification_options: Optional[list] = None

# State schema which is the Base State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Conversation history
    query: str  # Current query
    location: Optional[str]
    session_id: Optional[str]
    current_response: Optional[str]
    metadata: Optional[dict]  # Store metadata about current conversation
    chat_active: bool  # Track if conversation is still active

# Initialize memory saver
memory = MemorySaver()

def await_clarification_node(state: AgentState):
    """No need for agent script, simply directs flow to END with clarification message"""
    return Command(
        goto=END,
        update=state  # We want to keep the state for the conversation history
    )
    # how to return message?

def emergency_node(state: AgentState):
    """No need for agent script, simply answers with the whatsapp number = 112 for critical emergency"""
    return Command(
        goto=END,
        update=state
    )
    # how to return message?

# Build the agent network
def build_agent_network():
    workflow = StateGraph(AgentState)

    # Add all agents
    workflow.add_node("query_understanding", query_understanding.query_understanding_node)
    workflow.add_node("await_clarification", await_clarification_node)
    workflow.add_node("rag", rag.rag_node)
    workflow.add_node("response_quality", response_quality.response_quality_node)

    # Define complete flow
    workflow.add_edge(START, "query_understanding")
    workflow.add_conditional_edges(
        "query_understanding",
        lambda state: "rag" if state.get("query_type") == "clear" # need to add query_type somewhere
        else "await_clarification",
    )
    # add routing to emergency too
    workflow.add_edge("rag", "response_quality")
    workflow.add_edge("response_quality", END)

    return workflow.compile()


agent_network = build_agent_network()


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_input: ChatInput):
    try:
        # Initialize state
        state = {
            "messages": [], # Empty conversation history
            "query": chat_input.message, # Original user input
            "location": chat_input.location, # Initial location if provided
            "session_id": chat_input.session_id, # To track the conversation
        }
        print(f"state: {state}")

        # Process through agent network
        result = agent_network.invoke(state)
        print(f"result: {result}")

        # Format response
        response = ChatResponse(
            response=result["response"],
            emergency=result.get("is_emergency", False),
            needs_clarification=result.get("needs_clarification", False),
            clarification_options=result.get("clarification_options", None)
        )
        print(f"response: {response}")

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)