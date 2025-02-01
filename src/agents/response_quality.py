from typing import List, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command
from langchain_anthropic import ChatAnthropic
from src.agents.rag import InformationMetadata
import json
import os

# Configuration
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
COMPLETENESS_THRESHOLD = float(os.getenv("COMPLETENESS_THRESHOLD", "0.7"))

COMM_GUIDELINES = json.load(open('data/comms.json', 'rb'))["comms"]

class ConfigError(Exception):
    """An exception class for configuration errors."""
    def __init__(self, message):
        super().__init__(message)

class ResponseQualityInput(BaseModel):
    """Expected input from RAG agent"""
    text: str = Field(description="Generated response text")
    metadata: InformationMetadata = Field(description="Metadata about the information")
    relevant_chunks: List[str] = Field(description="Retrieved relevant text chunks")
    domains_covered: List[str] = Field(description="List of domains for which information was found")


class ResponseQualityOutput(BaseModel):
    """Output from quality check"""
    text: str = Field(description="Final response text")
    original_text: str = Field(description="Original response before modifications")
    metadata: InformationMetadata = Field(description="Original metadata")
    modifications_made: List[str] = Field(description="List of changes made to response", default_factory=list)


class ResponseQualityState(BaseModel):
    """State for Response Quality Agent"""
    initial_response: ResponseQualityInput
    final_response: Optional[ResponseQualityOutput] = None


def response_quality_node(state: dict) -> Command:
    """
    Evaluates response quality and returns improved version with quality context.
    Returns Command object with next node and state updates.
    """
    try:
        if "initial_response" not in state:
            raise ValueError("No initial response in state")

        input_data = ResponseQualityInput(**state["initial_response"])

        modifications = []
        response_text = input_data.text

        # Add caveats based on confidence/completeness
        completeness_score = input_data.metadata.completeness_score
        confidence_score = input_data.metadata.confidence_score

        if completeness_score < COMPLETENESS_THRESHOLD:
            response_text = (
                f"Based on the currently available information:\n\n{response_text}\n\n"
                "Note: There may be additional resources available. "
                "Would you like more specific information about any particular aspect?"
            )
            modifications.append("Added completeness caveat")

        if confidence_score < CONFIDENCE_THRESHOLD:
            response_text += (
                "\n\nFor the most up-to-date and complete information, "
                "we recommend contacting your local Red Cross office directly."
            )
            modifications.append("Added confidence caveat")

        # Review alignment with RedCross tone using Claude
        try:

            llm = ChatAnthropic(
                model="claude-3-5-haiku-20241022",
                temperature=0,
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )

            if not COMM_GUIDELINES:
                raise ConfigError("There are no communication guidelines provided in the data folder.")

            system_prompt = f"""You are a response quality assistant for a Red Cross virtual assistant.
                You need to check whether or not the assistant output follows the INCLUSIVE LANGUAGE GUIDELINE.
                To do this, you need to see if any of the terms that need to be avoided are present and, if needed, replace them with one of the preferred terms provided for each category.
                If no terms that need to be avoided are present in the text, you should output the assistant query as you received it.

                ABOUT THE INCLUSIVE LANGUAGE GUIDELINE:
                'Avoid' are words that should not be in your output and 'Preferred Terms' are the words you should use if you encounter
                a word to be avoided. These are available for you in pairs, starting with the term to search for and Avoid, and ending
                with the Preferred Terms that you need to replace. You should choose only one of the preferred terms depending on the context.
                '.etc' denotes any other words that are similar to the ones previously written in the same enumeration. The 'Reasoning' is included
                to help with distinguishing between instances where the avoided words might be used with a different meaning.

                INCLUSIVE LANGUAGE GUIDELINE:
                {COMM_GUIDELINES}
                """

            reviewed_text = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": response_text}
            ])

            if reviewed_text.content != response_text:
                response_text = reviewed_text.content
                modifications.append("Applied inclusive language guidelines")

        except Exception as e:
            print(f"Warning: Failed to check inclusive language: {e}")

        # Prepare final output
        output = ResponseQualityOutput(
            text=response_text,
            original_text=input_data.text,
            metadata=input_data.metadata,
            modifications_made=modifications
        )

        # Return Command with state update
        return Command(
            goto=END,
            update={
                "final_response": output.model_dump()
            }
        )

    except Exception as e:
        print(f"Error in response quality node: {e}")
        # Return original response with error caveat
        if "initial_response" in state and isinstance(state["initial_response"], dict):
            original_text = state["initial_response"].get("text", "")
            original_metadata = state["initial_response"].get("metadata", {})
        else:
            original_text = ""
            original_metadata = {}

        error_output = ResponseQualityOutput(
            text=f"{original_text}\n\nNote: Some of the information could not be completely verified. Please verify information with your local Red Cross office.",
            original_text=original_text,
            metadata=original_metadata,
            modifications_made=["Added error caveat"]
        )

        return Command(
            goto=END,
            update={
                "final_response": error_output.model_dump(),
                "error": str(e)
            }
        )
