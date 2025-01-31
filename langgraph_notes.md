**StateGraph:** 
This is a structure that represents the flow of work between agents. 
Like a roadmap showing how different agents should work together. 
Each agent represents a node in the graph, and the connections between them are edges.

**Nodes:** 
A node represents an agent that can perform specific tasks.
We have nodes for:
Query understanding (analyzes user input)
RAG (retrieves relevant info)
Response Quality (Refine response)
Web Agent could give directions etc

Each node can receive commands, process them, and potentially generate new commands for other nodes.
Edges: define the valid paths between nodes. 
For example, the edge from "understanding" to "RAG" means that after the query understanding agent 
completes its work, it can pass control to the rag agent.

**Commands:** 
These are messages that agents use to communicate. A Command includes:
action: what needs to be done
parameters: any data needed for the action
source_agent: who sent the command
target_agent: who should receive it

The workflow typically goes like this:
A user query enters the system.
The understanding agent analyzes it.
It creates a command for the rag agent.
The rag agent retrieves the relevant info and generates an initial response and commands the response agent.
The response quality agent finalize the answer.