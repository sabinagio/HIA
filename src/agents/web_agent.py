from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_anthropic import ChatAnthropic
from langgraph.types import Command

load_dotenv()

llm = ChatAnthropic(
    model="claude-3-5-haiku-20241022",
    temperature=0
)

def web_search(query: str) -> str:
    wrapper = DuckDuckGoSearchAPIWrapper(region="nl-nl", max_results=2) # time='y' limit to past year (m, d, w)
    search_tool = DuckDuckGoSearchResults(api_wrapper=wrapper)
    results = search_tool.run(query)
    return results

def prompt_search(query_context: dict) -> dict:
    """
    Uses query understanding output to perform targeted web searches.

    Args:
        query_context: {
            "original_query": str,
            "domains": list[str],  # e.g. ["food", "shelter"]
            "entities": dict,      # e.g. {"location": "Amsterdam", "date": "2024"}
            "language": str        # e.g. "english"
        }
    """

    # Map domains to relevant websites - otherwise could get things from newspapers "the red cross did this last year"
    domain_sites = {
        "food": ["voedselbank.nl", "voedselbankennederland.nl"],
        "shelter": ["deregenboog.org", "opvang.nl"],
        "healthcare": ["ggd.nl", "zorgverzekeringslijn.nl"],
        "domestic_violence": ["veiligthuis.nl", "blijfgroep.nl"],
        "education": ["amsterdam.nl/onderwijs"],
        "refugees": ["vluchtelingenwerk.nl", "refugeehelp.nl"]
    }

    # Get relevant sites based on query domains
    relevant_sites = ["rodekruis.nl"]  # Always include Red Cross
    location = query_context["entities"].get("location", "Netherlands")
    # if location.lower() == "amsterdam":
    #     relevant_sites.append("amsterdam.nl")
    # elif location.lower() == "rotterdam":
    #     relevant_sites.append("rotterdam.nl")
    # elif location.lower() == "eindhoven":
    #     relevant_sites.append("eindhoven.nl")
    # elif location.lower() == "utrecht":
    #     relevant_sites.append("utrecht.nl")
    # elif location.lower() == "groningen":
    #     relevant_sites.append("groningen.nl")
    # elif location.lower() == "den haag":
    #     relevant_sites.append("denhaag.nl")

    # Add domain-specific sites
    for domain in query_context["domains"]:
        if domain in domain_sites:
            relevant_sites.extend(domain_sites[domain])

    # Create search query builder prompt
    system_prompt = """
    Create a simple search query (maximum 10 words) in English or Dutch that will find help information.
    Return ONLY the search query, no explanation or strategy.
    """

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""
            Original query: {query_context['original_query']}
            Location: {query_context['entities'].get('location', 'Netherlands')}
            Domains: {query_context['domains']}
            Language: {query_context['language']}
        """}
    ])

    # Build final search query
    search_query = response.content
    print(search_query)
    domain_priority = " OR ".join(f"site:{site}" for site in relevant_sites)
    full_query = f"{search_query} {domain_priority}"

    # Perform search
    results = web_search(full_query)

    return {"web_response": results}

def search_summary(query_context: dict, search_result: dict) -> dict:

    system_prompt = """
    You are a Red Cross assistant who helps people in need in the Netherlands.
    
    Information given:
        - question(s) the vulnerable person is asking
        - relevant search results
        - vulnerable person's preferred language from question and context
        - Optional: vulnerable person's city
    
    Based on that information, you need to summarise the search results to address
    the vulnerable person's immediate need.
    Use kind and inclusive language, respecting the person's dignity.
    Give an answer that is as clear and useful as possible, the vulnerable person
    is in a stressful situation and needs to be helped and reassured.
    Answer in the preferred language of the vulnerable person.
    Include links to the source articles in your answer to foster trust.
    """

    llm_response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""
            Original query: {query_context['original_query']}
            Location: {query_context['entities'].get('location', 'Netherlands')}
            Domains: {query_context['domains']}
            Language: {query_context['language']}
        """}
    ])

    response = llm_response.content

    return {"web_agent_response": response}

def web_agent(query_context: dict):
    search_result = prompt_search(query_context)
    web_agent_response = search_summary(query_context, search_result)
    return Command(
        goto="response_quality",
        update={
            "initial_response": web_agent_response
        }
    )

if __name__ == "__main__":
    test_context = {
        "original_query": "Where can I get food assistance and shelter?",
        "domains": ["food", "shelter"],
        "entities": {"location": "Amsterdam"},
        "language": "english"
    }

    json_output = web_agent(test_context)
    print(json_output)