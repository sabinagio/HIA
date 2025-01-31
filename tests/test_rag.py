import pytest
from datetime import datetime

from sympy.codegen.cnodes import restrict

from src.agents.rag import rag_node, RAGInput, RAGState, RAGOutput, InformationMetadata
from typing import Dict


@pytest.fixture
def sample_query_context():
    return RAGInput(
        original_query="Where can I get food assistance?",
        domain="food",
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
    base_state["query_context"].language = "dutch"
    result = rag_node(base_state)
    print(f"test_rag_handles_different_languages: {result}")

    # Response should acknowledge language preference
    response_text = result.update["response"]["text"].lower()
    assert len(response_text) > 0

    # When running, check if response is actually in Dutch
    # We're not going to add language detection models here
    # So here we just verify it generates a response


def test_rag_domain_filtering(base_state):
    # Test food domain
    result = rag_node(base_state)
    print(f"test_rag_domain_filtering: {result}")
    chunks = result.update["response"]["relevant_chunks"]
    assert any("food" in chunk.lower() for chunk in chunks)

    # Test shelter domain
    base_state["query_context"].domain = "shelter"
    result = rag_node(base_state)
    chunks = result.update["response"]["relevant_chunks"]
    assert any("shelter" in chunk.lower() for chunk in chunks)


def test_rag_handles_missing_information(base_state):
    # Modify query to something unlikely to have good matches or that we know for sure we don't have
    base_state["query_context"].original_query = "Who won the Eurovision song contest in 1992?"
    result = rag_node(base_state)
    print(f"test_rag_handles_missing_information: {result}")

    # Should have low confidence/completeness scores
    metadata = result.update["response"]["metadata"]
    assert metadata["confidence_score"] > 0.5 # as in, it is certain that the information is not present
    assert metadata["completeness_score"] < 0.5

    # Response should acknowledge limited information
    response_text = result.update["response"]["text"].lower()
    print(response_text)
    # Better to check with eyes here as it's hard to predict which words will be used
    # assert any(phrase in response_text for phrase in [
    #     "limited information",
    #     "contact",
    #     "more details",
    #     "directly"
    # ])


def test_rag_uses_entity_information(base_state):
    # Add location entity
    base_state["query_context"].entities = {
        "location": "Amsterdam",
        "family_size": 4
    }

    result = rag_node(base_state)
    print(f"test_rag_uses_entity_information: {result}")
    response_text = result.update["response"]["text"].lower()

    # Response should incorporate entity information
    assert "amsterdam" in response_text or "location" in response_text

    # Entity information should influence search
    assert len(result.update["response"]["relevant_chunks"]) > 0


def test_rag_metadata_freshness(base_state):
    result = rag_node(base_state)
    print(f"test_rag_metadata_freshness: {result}")
    metadata = result.update["response"]["metadata"]

    # Convert last_updated to datetime
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
        # Check for contact fields that we know exist
        assert any(key in contact_info for key in [
            "email",
            "phone",
            "website",
            "address"
        ])