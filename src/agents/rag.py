from typing import List, Dict, Optional

from chromadb.errors import InvalidCollectionException
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from datetime import datetime
# from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.utils import embedding_functions
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter # for when we have actual data
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START
from langgraph.types import Command
import json
from src.utils.llm_utils import get_api_key

api_key = get_api_key()

# Input/Output schemas
class RAGInput(BaseModel):
    """Expected input from query understanding agent"""
    original_query: str = Field(description="Original user query")
    domain: str = Field(description="Domain of the query (e.g., food, shelter)")
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

    # Get query context
    query_context = state["query_context"]
    print(f"Obtained query context: {query_context}")

    # Search collection with enhanced query
    enhanced_query = f"""
    Domain: {query_context.domain}
    Query: {query_context.original_query}
    Entities: {query_context.entities}
    """
    # Domain: {query_context["domain"]}
    # Query: {query_context["original_query"]}
    # Entities: {query_context["entities"]}
    print("Defined enhanced query: {}".format(enhanced_query))

    # Query collection
    results = collection.query(
        query_texts=[enhanced_query],
        n_results=3,
        where={"domain": query_context.domain} if query_context.domain != "other" else None,
        include=["documents", "metadatas", "distances",]
    )
    print("Obtained results: {}".format(results))

    query_results = {
        'documents': results['documents'][0],  # All documents for our query
        'metadatas': results['metadatas'][0],  # All metadata for our query
        'distances': results['distances'][0]  # All distances for our query
    }
    print(f"Query results: {query_results}")

    # Now query_results contains all docs/metadata/distances for our single query
    docs = query_results['documents']
    metadatas = query_results['metadatas']
    distances = query_results['distances']

    # Calculate confidence and completeness scores
    avg_score = sum(distances) / len(distances) if distances else 0
    confidence_score = min(1.0, avg_score / 0.8)  # Normalize to 0-1
    completeness_score = min(1.0, len(docs) / 3)  # Based on getting 3 relevant results

    # Prepare document context for LLM
    document_context = "\n".join(docs)

    # Prepare query input context
    context = f"""
    Retrieved information:
    {document_context}

    Additional context:
    Location: {query_context.entities.get('location', 'Not specified')}
    Other relevant information: {', '.join(f'{k}: {v}' for k, v in query_context.entities.items() if k != 'location')} # once we get to know the data we can improve this
    """

    system_prompt = f"""You are a helpful Red Cross assistant. Generate a response based on the retrieved information.
    Language to use: {query_context.language}

    Retrieved information:
    {context}

    Important: Only include verifiable information from the sources. If information is incomplete,
    acknowledge this and suggest contacting Red Cross directly for more details."""

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query_context.original_query}
    ])
    # print(f"Obtained LLM response: {response}")

    # Get most recent metadata
    latest_idx = max(range(len(metadatas)),
                    key=lambda i: metadatas[i]["last_updated"])
    latest_metadata = metadatas[latest_idx]
    print(f"Latest metadata: {latest_metadata}")
    contact_info = json.loads(latest_metadata.get("contact", "{}"))  # load the contact json
    print("Obtained contact info: {}".format(contact_info))

    # Prepare output
    output = RAGOutput(
        text=response.content,
        metadata=InformationMetadata(
            source=", ".join(m["source"] for m in metadatas),
            last_updated=datetime.strptime(latest_metadata["last_updated"], "%Y-%m-%d"),
            contact_info=contact_info,
            completeness_score=completeness_score,
            confidence_score=confidence_score,
        ),
        relevant_chunks=docs
    )
    print(f"Prepared output:{output}")

    # Return update and route to response quality agent
    return Command(
        goto="response_quality",
        update={
            "response": output.model_dump()
        }
    )



# Create graph
workflow = StateGraph(RAGState)
workflow.add_node("rag", rag_node)
workflow.add_edge(START, "rag")
graph = workflow.compile()