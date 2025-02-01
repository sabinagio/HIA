import pytest
from fastapi.testclient import TestClient
from main import app, build_agent_network, ChatInput
from unittest.mock import Mock, patch

client = TestClient(app)


@pytest.fixture
def mock_agent_network():
    with patch('main.build_agent_network') as mock:
        network = Mock()
        mock.return_value = network
        yield network


@pytest.fixture
def sample_chat_input():
    return {
        "message": "I need food assistance",
        "location": "Amsterdam",
        "session_id": "test-123"
    }

def test_basic_chat_flow():
    """Test basic flow: input query -> main -> output response"""

    # Input query
    chat_input = {
        "message": "I need food assistance",
        "location": "Amsterdam",
        "session_id": "test-123"
    }
    print(f"\nSending input: {chat_input}")

    # Send request to endpoint
    response = client.post("/chat", json=chat_input)
    print(f"Response status code: {response.status_code}")
    print(f"Response content: {response.content}")

    # Basic checks
    if response.status_code == 200:
        data = response.json()
        print(f"Parsed response data: {data}")
        assert "response" in data
        assert isinstance(data["response"], str)
    else:
        print(f"Error response: {response.json()}")
        pytest.fail(f"Request failed with status {response.status_code}")


def test_chat_endpoint_error_handling(mock_agent_network, sample_chat_input):
    # Test error handling
    mock_agent_network.invoke.side_effect = Exception("Test error")

    response = client.post("/chat", json=sample_chat_input)
    assert response.status_code == 500
    print(response.json())


def test_invalid_input():
    # Test missing required field
    response = client.post("/chat", json={"location": "Amsterdam"})
    assert response.status_code == 422


def test_chat_input_validation():
    # Test input model validation
    valid_input = ChatInput(
        message="Test message",
        location="Test location",
        session_id="test-123"
    )
    assert valid_input.message == "Test message"
    assert valid_input.location == "Test location"
    assert valid_input.session_id == "test-123"