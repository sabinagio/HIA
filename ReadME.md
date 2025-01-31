# Helpful Information as Aid - HFG7

## Installation

1. To use the `src` modules, make sure to run this command in your conda environment:
```bash
pip install -e .
```

2. To load and re-load `src` modules into Jupyter notebooks, use the following commands at the top of the notebook:
```python
%load_ext autoreload
%autoreload 2
```
This will allow you to reload the Python scripts without restarting the Jupyter kernel.

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
```
