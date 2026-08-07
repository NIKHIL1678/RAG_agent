from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# Define the State specific to the RAG Subgraph
class RAG(TypedDict):
    messages : list
    query : str
    result : str
    error : str    

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# --- THE FIX: Using a custom @tool instead of create_retriever_tool ---
@tool
def custom_document_search(query: str) -> str:
    """Search and return information from user-provided URLs and documents. Always use this to answer questions about uploaded files."""
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 4, 'fetch_k': 20})
        docs = retriever.invoke(query)
        
        if not docs:
            return f"No relevant information found in the custom documents for the query: '{query}'."
            
        # Format the retrieved chunks into a single string for the LLM
        formatted_results = f"Document Search Results for '{query}':\n\n"
        for i, doc in enumerate(docs, 1):
            formatted_results += f"--- Snippet {i} ---\n{doc.page_content}\n\n"
            
        return formatted_results
    except Exception as e:
        return f"Error retrieving documents from database: {str(e)}"

# Bind the custom tool to the RAG Agent LLM
RAG_TOOLS = [custom_document_search]
rag_agent_llm = llm.bind_tools(RAG_TOOLS)

def RAG_Agent_Node(state: RAG) -> RAG:
    System_input = """You are a RAG (Retrieval-Augmented Generation) Agent. 
    Use your custom_document_search tool to find contextual information from the database to answer user queries accurately."""
    
    system_message = SystemMessage(content=System_input)
    current_messages = state.get('messages', [])
    
    if not current_messages:
        messages_for_llm = [system_message, HumanMessage(content=state.get('query', ''))]
    else:
        messages_for_llm = [system_message] + current_messages
        
    response = rag_agent_llm.invoke(messages_for_llm)
    
    state['messages'] = current_messages + [response]
    state['result'] = response.content if response.content else "Tool call generated."
    return state

def RAG_Agent_TOOLNode(state: RAG) -> RAG:
    messages = state.get('messages', [])
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            # Execute our new custom tool
            if tool_name == "custom_document_search":
                response = custom_document_search.invoke(tool_args)
            else:
                response = f"Tool {tool_name} not found."
                
            tool_message = ToolMessage(content=str(response), tool_call_id=tool_call["id"])
            messages.append(tool_message)
            
    state['messages'] = messages
    return state

def routing_by_RAG_Agent(state: RAG) -> str:
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "RAG_Agent_TOOLNode"
    return "Supervisor"

# Build and Compile the RAG Subgraph
RAG_graph = StateGraph(RAG)
RAG_graph.add_node("RAG_Agent", RAG_Agent_Node)
RAG_graph.add_node("RAG_Agent_TOOLNode", RAG_Agent_TOOLNode)

RAG_graph.add_edge(START, "RAG_Agent")
RAG_graph.add_conditional_edges(
    "RAG_Agent", 
    routing_by_RAG_Agent, 
    {
        "RAG_Agent_TOOLNode": "RAG_Agent_TOOLNode",
        "Supervisor": END 
    }
)
RAG_graph.add_edge("RAG_Agent_TOOLNode", "RAG_Agent")

compiled_rag_graph = RAG_graph.compile()