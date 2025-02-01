# Helpful Information as Aid - HFG7

## Agents

### Query Understanding Agent

### RAG agent
It receives structured **input from the query understanding agent** (rag_update => RAGInput)\
**Uses Chroma** as the vector store with test data (can be replaced with production data) \
**Enhances the search query with domain and entity information** (for the domain that's even more helpful if we cluster
topics and add a topic tag to the metadata of the Vector DB - and then we use this topic list to ask the
query understanding agent to classify input) \
**Calculates confidence and completeness scores** based on search results

Generates a response that includes:

- The actual response text
- Metadata about sources and freshness
- Contact information where available
- Confidence and completeness scores

Routes the **output to the response quality agent**

## Utils & Schemas

## Tests

```
# start all tests
pytest tests/

# start a specific test
python -m pytest tests/test_query_understanding.py::test_clear_food_assistance_query -v
python -m pytest tests/test_rage.py::test_rag_handles_different_languages -v
```
