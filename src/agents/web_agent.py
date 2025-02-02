from dotenv import load_dotenv
from typing import List, Dict
import re
from urllib.parse import urlparse
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_anthropic import ChatAnthropic
from langgraph.types import Command
import time

load_dotenv()

llm = ChatAnthropic(
    model="claude-3-5-haiku-20241022",
    temperature=0
)


def extract_urls_from_text(text: str) -> List[str]:
    """Extract all URLs from text using regex."""
    # Pattern matches both http(s):// URLs and www. URLs
    url_pattern = r'(?:https?://)?(?:www\.)?[\w\-]+\.[\w\-]+(?:\.\w+)?(?:/[\w\-\./?%&=]*)?'

    urls = re.findall(url_pattern, text)
    # Clean and normalize URLs
    clean_urls = []
    for url in urls:
        # Add https:// if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        # Parse URL to get domain
        parsed = urlparse(url)
        clean_urls.append(parsed.netloc)

    return list(set([u.lower() for u in clean_urls]))

def get_contact_info(initial_results: str) -> List[Dict]:
    """Perform a second search to get the contact information."""

    contact_results = []
    contact_wrapper = DuckDuckGoSearchAPIWrapper(region="nl-nl", max_results=1)
    contact_search_tool = DuckDuckGoSearchResults(api_wrapper=contact_wrapper)

    web_domains = extract_urls_from_text(initial_results)
    pattern = re.compile(r'^(?!.*\d).*$')
    web_domains = [d for d in web_domains if pattern.match(d)]

    for web_domain in web_domains:
        # Create simple contact-specific search query
        contact_query = f"{web_domain} contact"

        try:
            contact_info = contact_search_tool.run(contact_query)
            contact_results.append({
                "domain": web_domain,
                "contact_info": contact_info
            })
            time.sleep(0.01)
        except Exception as e:
            print(f"Error getting contact info for {web_domain}: {e}")

    return contact_results

def web_search(query: str) -> dict:
    wrapper = DuckDuckGoSearchAPIWrapper(region="nl-nl", max_results=2) # time='y' limit to past year (m, d, w)
    search_tool = DuckDuckGoSearchResults(api_wrapper=wrapper)
    results = search_tool.run(query)
    # Get contact information for found sources
    contact_info = get_contact_info(results)
    return {
        "web_response": {
            "results": results,
            "contact_details": contact_info,
            "query_used": query
        }
    }

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

    return results

def search_summary(query_context: dict, search_result: dict) -> dict:

    summary_system_prompt = """
    You are a Red Cross assistant who helps people in need in the Netherlands.
    
    Information given:
        - question(s) the vulnerable person is asking
        - relevant search results
        - vulnerable person's preferred language from question and context
        - Optional: vulnerable person's city
    
    Based on that information, you need to summarise the search results to address the vulnerable person's immediate need.
    Use kind and inclusive language, respecting the person's dignity while still being professional,
    and don't call the user "friend"  or "brother/sister". 
    This is a chat so you can address the user as "You/you".
    Give an answer that is as clear and useful as possible with all the necessary practical information.
    The user is a vulnerable person who is in a stressful situation and needs to be helped and reassured.
    Answer in the preferred language of the vulnerable person.
    Include links to the source articles in your answer to foster trust.
    Don't provide statistics or non-practical information.
    """

    llm_summary_response = llm.invoke([
        {"role": "system", "content": summary_system_prompt},
        {"role": "user", "content": f"""
            Original query: {query_context['original_query']}
            Location: {query_context['entities'].get('location', 'Netherlands')}
            Domains: {query_context['domains']}
            Language: {query_context['language']}
            Web Search Results: {search_result["web_response"]["results"]}
            Contact Information found: {search_result["web_response"]["contact_details"]}
        """}
    ])

    summary_response = llm_summary_response.content

    return {"web_agent_response": summary_response}

def web_agent_node(state: dict):
    query_context = state.get("query_context")
    search_result = prompt_search(query_context)
    web_agent_response = search_summary(query_context, search_result)
    print(web_agent_response)
    return Command(
        goto="response_quality",
        update={
            "web_agent_response": web_agent_response
        }
    )

if __name__ == "__main__":
    test_context = {
        "original_query": "Where can I get a doctor for my sick child?",
        "domains": ["Health & Wellbeing"],
        "entities": {"location": "Amsterdam"},
        "language": "ukrainian"
    }

    web_agent_output = web_agent_node(test_context)