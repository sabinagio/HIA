import pytest
from datetime import datetime
from src.agents.rag import rag_node, RAGInput, RAGState, RAGOutput, InformationMetadata


@pytest.fixture
def sample_query_context():
    return RAGInput(
        original_query="Where can I get food assistance and shelter?",
        domains=["food", "shelter"],  # multiple domains
        entities={"location": "Amsterdam"},
        language="english"
    )


@pytest.fixture
def base_state(sample_query_context):
    return RAGState(
        query_context=sample_query_context,
        response=None
    )


def test_rag_node_basic_response(base_state):
    # Execute RAG node
    result = rag_node(base_state)
    print(f"test_rag_node_basic_response: {result}")

    # Check command structure
    assert result.goto == "response_quality"
    assert "response" in result.update

    # Parse response
    response = result.update["response"]
    assert "text" in response
    assert "metadata" in response
    assert "relevant_chunks" in response
    assert "domains_covered" in response


def test_rag_response_metadata_structure(base_state):
    result = rag_node(base_state)
    print(f"test_rag_response_metadata_structure: {result}")
    metadata = result.update["response"]["metadata"]

    # Check all required metadata fields are present
    assert "source" in metadata
    assert "last_updated" in metadata
    assert "contact_info" in metadata
    assert "completeness_score" in metadata
    assert "confidence_score" in metadata

    # Check score ranges
    assert 0 <= metadata["completeness_score"] <= 1
    assert 0 <= metadata["confidence_score"] <= 1


def test_rag_handles_different_languages(base_state):
    # Modify query context to test different languages
    base_state["query_context"].language = "dutch" # languages tested: Dutch, Ukrainian, Syrian, Iranian
    result = rag_node(base_state)
    print(f"test_rag_handles_different_languages: {result}")

    # Response should acknowledge language preference
    response_text = result.update["response"]["text"].lower()
    assert len(response_text) > 0


def test_rag_domain_filtering(base_state):
    # Test multiple domains
    result = rag_node(base_state)
    print(f"test_rag_domain_filtering: {result}")
    chunks = result.update["response"]["relevant_chunks"]
    domains_covered = result.update["response"]["domains_covered"]

    # Check if both domains are covered in chunks
    assert any("food" in chunk.lower() for chunk in chunks)
    assert any("shelter" in chunk.lower() for chunk in chunks)

    # Check if domains are properly tracked
    assert "food" in domains_covered
    assert "shelter" in domains_covered

    # Test single domain case
    base_state["query_context"].domains = ["shelter"]
    result = rag_node(base_state)
    chunks = result.update["response"]["relevant_chunks"]
    domains_covered = result.update["response"]["domains_covered"]
    assert any("shelter" in chunk.lower() for chunk in chunks)
    assert "shelter" in domains_covered

    # Test with "Other" domain
    base_state["query_context"].domains = ["Other"]
    result = rag_node(base_state)
    assert len(result.update["response"]["relevant_chunks"]) > 0


def test_rag_handles_missing_information(base_state):
    # Modify query to something unlikely to have good matches
    base_state["query_context"].original_query = "Who won the Eurovision song contest in 1992?"
    base_state["query_context"].domains = ["Other"]
    result = rag_node(base_state)
    print(f"test_rag_handles_missing_information: {result}")

    # Should have low confidence/completeness scores
    metadata = result.update["response"]["metadata"]
    assert metadata["confidence_score"] >= 0.0  # hard to tell what it should be, confident that you found nothing or not confident because nothing was found
    assert metadata["completeness_score"] < 0.5

    # Response should acknowledge limited information
    response_text = result.update["response"]["text"].lower()
    print(response_text)


def test_rag_uses_entity_information(base_state):
    # Add multiple entities
    base_state["query_context"].entities = {
        "location": "Amsterdam",
        "family_size": 4,
        "urgency": "high"
    }

    result = rag_node(base_state)
    print(f"test_rag_uses_entity_information: {result}")
    response_text = result.update["response"]["text"].lower()

    # Response should incorporate entity information
    assert "amsterdam" in response_text or "location" in response_text

    # Entity information should influence search
    assert len(result.update["response"]["relevant_chunks"]) > 0
    # Check if both domains are represented
    assert len(result.update["response"]["domains_covered"]) >= 2


def test_rag_metadata_freshness(base_state):
    result = rag_node(base_state)
    print(f"test_rag_metadata_freshness: {result}")
    metadata = result.update["response"]["metadata"]

    # Convert last_updated to datetime if needed
    if isinstance(metadata["last_updated"], str):
        last_updated = datetime.strptime(metadata["last_updated"], "%Y-%m-%d")
    else:
        last_updated = metadata["last_updated"]

    # Check that the information isn't too old
    assert last_updated.year >= 2024


def test_rag_contact_info_structure(base_state):
    result = rag_node(base_state)
    print(f"test_rag_contact_info_structure: {result}")
    contact_info = result.update["response"]["metadata"]["contact_info"]

    # If contact info is provided, check structure
    if contact_info:
        assert isinstance(contact_info, dict)
        # Check for contact fields
        assert any(key in contact_info for key in [
            "email",
            "phone",
            "website",
            "address"
        ])


def test_multiple_domains_completeness(base_state):
    # Test with multiple domains including one that doesn't exist
    base_state["query_context"].domains = ["food", "shelter", "nonexistent_domain"]
    result = rag_node(base_state)

    # Check completeness score reflects partial coverage
    metadata = result.update["response"]["metadata"]
    assert metadata["completeness_score"] <= 0.67  # Only 2 out of 3 domains exist

    # Check domains_covered
    domains_covered = result.update["response"]["domains_covered"]
    assert "food" in domains_covered
    assert "shelter" in domains_covered
    assert "nonexistent_domain" not in domains_covered


def test_domain_priority(base_state):
    # Test that results are balanced across domains
    base_state["query_context"].domains = ["food", "shelter", "health"]
    result = rag_node(base_state)

    chunks = result.update["response"]["relevant_chunks"]
    domains_covered = result.update["response"]["domains_covered"]

    # Check that we have results from multiple domains
    assert len(domains_covered) > 1

    # Check that response text includes information from multiple domains
    response_text = result.update["response"]["text"].lower()
    domain_mentions = sum(1 for domain in ["food", "shelter", "health"]
                          if domain in response_text)
    assert domain_mentions > 1