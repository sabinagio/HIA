from typing import List, Dict, Optional
from chromadb.errors import InvalidCollectionException
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions
from langchain_anthropic import ChatAnthropic
from langgraph.types import Command
import json
from src.utils.llm_utils import get_api_key
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
api_key = get_api_key()

# Input/Output schemas
class RAGInput(BaseModel):
    """Expected input from query understanding agent"""
    original_query: str = Field(description="Original user query")
    domains: List[str] = Field(description="List of domains relevant to the query (e.g., ['food', 'shelter'])")
    entities: Dict = Field(description="Extracted entities from query")
    language: str = Field(description="Language of the query")


class InformationMetadata(BaseModel):
    """Metadata about the retrieved information"""
    source: str = Field(description="Source of the information (e.g., document name, URL)")
    last_updated: datetime = Field(description="When the information was last updated")
    contact_info: Optional[Dict] = Field(description="Relevant point of contact information")
    completeness_score: float = Field(
        description="Score 0-1 indicating completeness of available information",
        ge=0.0,
        le=1.0
    )
    confidence_score: float = Field(
        description="Score 0-1 indicating confidence in the information",
        ge=0.0,
        le=1.0
    )


class RAGOutput(BaseModel):
    """Output from RAG agent"""
    text: str = Field(description="Generated response text")
    metadata: InformationMetadata = Field(description="Metadata about the information")
    relevant_chunks: List[str] = Field(description="Retrieved relevant text chunks")
    domains_covered: List[str] = Field(description="List of domains for which information was found")


class RAGState(TypedDict):
    """State for RAG Agent"""
    query_context: RAGInput
    response: Optional[RAGOutput]


def initialize_vectorstore():
    """Initialize and return Chroma vectorstore with embeddings"""
    # Initialize persistent Chroma client
    client = chromadb.PersistentClient(path="./chroma_db")

    # client.delete_collection("test_collection")

    # Create or get existing collection
    try:
        collection = client.get_collection(
            name="test_collection",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
        print("Collection obtained.")

        return collection

    except (ValueError, InvalidCollectionException):  # Collection doesn't exist
        collection = client.create_collection(
            name="test_collection",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
        print("Collection created.")

        # For testing/development, create a simple vectorstore with some sample data
        documents = [
            "Food assistance is available at the Red Cross office on Mainstreet in Amsterdam. Open Monday-Friday 9-5.",
            "Emergency shelter services can be accessed 24/7 at our downtown location in Amsterdam.",
            "Financial aid applications are processed within 5-7 business days.",
            "For immediate medical assistance, please call emergency services at 112.",
        ]

        # Add metadata to each document
        metadatas = [
            {
                "source": "RC Food Services Guide",
                "last_updated": "2024-01-15",
                "contact": json.dumps({"email": "food@redcross.org", "phone": "555-0123"}),
                "domain": "food"
            },
            {
                "source": "RC Shelter Guide",
                "last_updated": "2024-01-20",
                "contact": json.dumps({"phone": "555-0124"}),
                "domain": "shelter"
            },
            {
                "source": "RC Financial Aid Guide",
                "last_updated": "2024-01-10",
                "contact": json.dumps({"email": "finance@redcross.org"}),
                "domain": "financial"
            },
            {
                "source": "RC Emergency Guide",
                "last_updated": "2024-01-01",
                "contact": json.dumps({"phone": "112"}),
                "domain": "health"
            }
        ]

        # Add documents to collection
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=[f"doc_{i}" for i in range(len(documents))]
        )

        return collection


def rag_node(state: RAGState):
    """
    RAG agent that retrieves relevant information and generates a response with metadata.
    """
    collection = initialize_vectorstore()
    print("Initialized vectorstore")
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0
    )

    query_context = state["query_context"]
    print(f"Obtained query context: {query_context}")

    # Build enhanced query incorporating all domains
    domains_str = ", ".join(query_context["domains"])
    enhanced_query = f"""
    Domains: {domains_str}
    Query: {query_context["original_query"]}
    Entities: {query_context["entities"]}
    """
    print("Defined enhanced query: {}".format(enhanced_query))

    # Define target number of results we want
    target_results_per_domain = 3 # k in the search
    total_target_results = len(query_context["domains"]) * target_results_per_domain

    # Collect results for all domains
    all_documents = []
    all_metadatas = []
    all_distances = []
    domains_covered = set()

    # Query for each domain
    for domain in query_context["domains"]:
        if domain.lower() != "other":
            # Query with domain filter
            results = collection.query(
                query_texts=[enhanced_query],
                n_results=target_results_per_domain,
                where={"domain": domain.lower()},
                include=["documents", "metadatas", "distances"]
            )
        else:
            # For "Other", try to get more results since we're searching broadly
            results = collection.query(
                query_texts=[enhanced_query],
                n_results=total_target_results,  # Try to get more results for general queries
                include=["documents", "metadatas", "distances"]
            )

        if results['documents'][0]:  # If we got any results
            all_documents.extend(results['documents'][0])
            all_metadatas.extend(results['metadatas'][0])
            all_distances.extend(results['distances'][0])
            domains_covered.add(domain)

    # Prepare consolidated results
    query_results = {
        'documents': all_documents,
        'metadatas': all_metadatas,
        'distances': all_distances
    }
    print(f"Consolidated query results: {query_results}")

    # # Don't calculate if no results
    # if not all_distances:
    #     confidence_score = 0.0
    #     completeness_score = 0.0
    # else:
    #     # Normalize distances to [0, 1] range to avoid negative values
    #     normalized_distances = [min(1.0, max(0.0, d)) for d in all_distances]
    #     avg_distance = sum(normalized_distances) / len(normalized_distances)
    #     confidence_score = max(0.0, min(1.0, 1.0 - avg_distance))
    #
    #     # Calculate completeness based on number and relevance of results
    #     if "Other" in query_context["domains"]:
    #         # For "Other" queries, base completeness on how many relevant docs we found
    #         # compared to what we asked for, and their average relevance
    #         # (because otherwise topic is Other and completeness is 0.5)
    #         results_ratio = len(all_documents) / total_target_results
    #         relevance_weight = max(0.0, min(1.0, 1.0 - avg_distance))  # bound
    #         final_completeness_score = max(0.0, min(1.0, results_ratio * relevance_weight))
    #     else:
    #         # For domain-specific queries, consider both domain coverage and documents retrieved
    #         domain_coverage = len(domains_covered) / len(query_context["domains"])
    #         results_coverage = len(all_documents) / total_target_results
    #         final_completeness_score = min(1.0, (domain_coverage + results_coverage) / 2)

    # Prepare document context for LLM
    document_context = "\n".join(all_documents)

    # Ask LLM to assess the quality of the information completeness to get a more evolved score
    completeness_prompt = f"""
    Based on the following query and retrieved information, rate how complete and comprehensive 
    the available information is on a scale from 0 to 1, where:
    - 1.0 means all aspects of the query are fully addressed with detailed, relevant information
    - 0.0 means no relevant information was found

    Query: {query_context["original_query"]}

    Retrieved Information:
    {document_context}

    Provide only a number between 0 and 1, no explanation.
    """
    # try - except in case the llm does not return a number
    # try:
    #     llm_response = llm.invoke(completeness_prompt).content.strip()
    #     llm_completeness = float(llm_response)
    #     llm_completeness = max(0.0, min(1.0, llm_completeness)) # bound
    #     final_completeness_score = llm_completeness
    #     # final_completeness_score = max(0.0, min(1.0, (completeness_score + llm_completeness) / 2))
    # except (ValueError, Exception) as e:
    #     print(f"LLM completeness error: {e}")
    #     final_completeness_score = 0.0


    context = f"""
    Retrieved information:
    {document_context}

    Additional context:
    Domains: {domains_str}
    Location: {query_context["entities"].get('location', 'Not specified')}
    Other relevant information: {', '.join(f'{k}: {v}' for k, v in query_context["entities"].items() if k != 'location')}
    """

    system_prompt = f"""You are a helpful Red Cross assistant. Generate a response based on the retrieved information.
    Language to use: {query_context["language"]}

    Retrieved information:
    {context}

    Important: 
    1. Address all requested domains: {domains_str}
    2. Only include verifiable information from the sources
    3. If information for any domain is incomplete, acknowledge this and suggest contacting Red Cross directly
    4. Structure your response to clearly separate information for different domains"""

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query_context["original_query"]}
    ])

    # Get most recent metadata
    if not all_metadatas:
        latest_metadata = {}
        contact_info = {}
        last_updated = datetime.now()
    else:
        latest_idx = max(range(len(all_metadatas)),
                         key=lambda i: all_metadatas[i]["last_updated"])
        latest_metadata = all_metadatas[latest_idx]
        contact_info = json.loads(latest_metadata.get("contact", "{}"))
        last_updated = datetime.strptime(latest_metadata["last_updated"], "%Y-%m-%d")

    # Prepare output
    output = RAGOutput(
        text=response.content,
        metadata=InformationMetadata(
            source=", ".join(set(m["source"] for m in all_metadatas)),
            last_updated=last_updated,
            contact_info=contact_info,
            completeness_score= 1,#final_completeness_score,
            confidence_score= 1,#confidence_score,
        ),
        relevant_chunks=all_documents,
        domains_covered=list(domains_covered)
    )
    print(f"Prepared output: {output}")

    return Command(
        goto="response_quality",
        update={
            "response": output.model_dump()
        }
    )