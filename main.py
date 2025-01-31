from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from langgraph.graph import StateGraph, START
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


# Build the agent network
def build_agent_network():
    workflow = StateGraph()

    # Add all agents - to be defined after brainstorming
    workflow.add_node("query_understanding", query_understanding.query_understanding_node)
    # workflow.add_node("context_management", context_management.context_management_node)
    workflow.add_node("rag", rag.rag_node)
    # workflow.add_node("emergency_response", emergency_response.emergency_response_node)
    # workflow.add_node("response_quality", response_quality.response_quality_node)
    # workflow.add_node("feedback", feedback.feedback_node)

    # Add edges
    workflow.add_edge(START, "query_understanding")
    # Then other edges when new agents are done

    return workflow.compile()


agent_network = build_agent_network()


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_input: ChatInput):
    try:
        # Initialize state
        state = {
            "messages": [],
            "query": chat_input.message,
            "location": chat_input.location,
            "session_id": chat_input.session_id
        }

        # Process through agent network
        result = agent_network.invoke(state)

        # Format response
        response = ChatResponse(
            response=result["response"],
            emergency=result.get("is_emergency", False),
            needs_clarification=result.get("needs_clarification", False),
            clarification_options=result.get("clarification_options")
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)