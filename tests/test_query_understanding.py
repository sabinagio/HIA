import pytest
from src.agents.query_understanding import query_understanding_node, AgentState

@pytest.fixture
def base_state():
    return {
        "messages": [],
        "query": "",
        "location": None
    }

def test_clear_food_assistance_query():
    state = AgentState(
        messages=[],
        query="Where can I get food assistance for my family of 4 in Amsterdam?",
        location="Amsterdam"
    )

    result = query_understanding_node(state)
    print(result)

    # Since query is clear, we expect two Commands (for context and RAG)
    assert len(result) == 2

    # Check Context Management update
    context_cmd = next(cmd for cmd in result if cmd.goto == "context_management")
    assert context_cmd.update["user_context"]["domain"] == "food"
    assert context_cmd.update["user_context"]["location"] == "Amsterdam"
    assert context_cmd.update["user_context"]["entities"]["family_size"] == 4

    # Check RAG update
    rag_cmd = next(cmd for cmd in result if cmd.goto == "rag")
    assert "food" in rag_cmd.update["query_context"]["original_query"].lower()
    assert rag_cmd.update["query_context"]["domain"] == "food"


def test_emergency_medical_query():
    state = AgentState(
        messages=[],
        query="I'm having chest pains and difficulty breathing",
        location="Rotterdam"
    )

    result = query_understanding_node(state)

    # For emergency, expect single Command to emergency response
    assert result.goto == "emergency_response"
    assert result.update["analysis"]["query_type"] == "emergency"
    assert result.update["analysis"]["domain"] == "health"
    assert result.update["analysis"]["confidence"] >= 0.9


def test_unclear_query_needs_clarification():
    state = AgentState(
        messages=[],
        query="I need help",
        location=None
    )

    result = query_understanding_node(state)

    # For unclear queries, expect clarification Command
    assert result.goto == "await_clarification"
    assert "analysis" in result.update
    assert "messages" in result.update

    # Check that there are multiple clarification options
    assert len(result.update["analysis"]["clarification_options"]) >= 2

    # Check that options are actually presented - multiple lines for options
    assert len(result.update["messages"][-1]["content"].split("\n")) > 1


def test_multilingual_query():
    state = AgentState(
        messages=[],
        query="Waar kan ik voedselhulp krijgen?",  # Dutch: Where can I get food assistance?
        location="Amsterdam"
    )

    result = query_understanding_node(state)

    assert len(result) == 2  # Expect context and RAG updates

    context_cmd = next(cmd for cmd in result if cmd.goto == "context_management")
    assert context_cmd.update["user_context"]["language"] == "dutch"
    assert context_cmd.update["user_context"]["domain"] == "food"


def test_emotional_distress_not_emergency():
    state = AgentState(
        messages=[],
        query="I'm worried about feeding my children next week",
        location=None
    )

    result = query_understanding_node(state)

    context_cmd = next(cmd for cmd in result if cmd.goto == "context_management")

    # Check for any negative/concerned emotional states
    emotional_state = context_cmd.update["user_context"]["emotional_state"].lower()
    assert any(emotion in emotional_state for emotion in [
        "worried",
        "anxious",
        "concerned",
        "distressed",
        "troubled",
        "uneasy"
    ]), f"Emotional state '{emotional_state}' should indicate concern"

    # Check that query domain is correctly identified
    assert context_cmd.update["user_context"]["domain"] == "food"
    # Should not be routed to emergency despite emotional content
    assert any(cmd.goto == "rag" for cmd in result)

    # Check that children are identified in entities
    entities = context_cmd.update["user_context"]["entities"]
    print(entities)
    has_children_mention = any(
        "children" in str(key).lower() or "children" in str(value).lower()
        for key, value in entities.items()
    )
    assert has_children_mention, f"Should identify presence of children in entities. Got: {entities}"

