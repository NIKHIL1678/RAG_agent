from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent
from langchain.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from ddgs import DDGS
from langchain_groq import ChatGroq
import requests
import re
import os
import time
from functools import lru_cache
from dotenv import load_dotenv

# --- NEW IMPORTS ---
from ingestion import ingest_document
from rag_agent import compiled_rag_graph

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

class Supervisor(TypedDict):
    messages : list
    last_agent : str
    last_result : str
    user_input : str
 
class API(TypedDict):
    messages : list
    query : str
    result : str
    error : str  
    url : str  
    
class Wikipedia(TypedDict):
    messages : list
    query : str
    result : str
    error : str  
    url : str  
 
""" All are Agents Nodes:"""

def Supervisor_Agent(state: Supervisor) -> Supervisor:
    current_messages = state.get('messages', [])
    
    # --- 1. SMART SHORT-CIRCUIT ---
    if current_messages and isinstance(current_messages[-1], AIMessage):
        last_content = current_messages[-1].content.strip().lower()
        # UPDATED: Added rag and ingest to the short-circuit list
        if last_content not in ["api", "wikipedia", "rag", "ingest"]:
            return state
            
    # --- 2. BUILD THE CLEAN CONTEXT ---
    # UPDATED: Added RAG and Ingest conditions
    System_input = """You are a supervisor agent managing specialized worker agents:
    1. API_Agent: Use strictly for booking flights, booking hotels, and finding tourist places.
    2. Wikipedia_Agent: Use strictly for searching factual data, current events, or history on the internet.
    3. RAG_Agent: Use strictly for answering questions about the user's uploaded files or custom databases.
    4. Ingest_Agent: Use strictly when the user provides a URL or file path and asks you to ingest, upload, or process it.
    
    Rules for your response:
    - If it's a simple greeting ("Hi", "Hello"), reply directly. Do NOT write code.
    - If the user query requires the API_Agent, reply with exactly "API".
    - If the user query requires the Wikipedia_Agent, reply with exactly "Wikipedia".
    - If the user query requires the RAG_Agent, reply with exactly "RAG".
    - If the user query requires ingesting a file/url, reply with exactly "Ingest".
    
    Look at the user's latest query below and decide."""

    messages_for_llm = [SystemMessage(content=System_input)]
    
    for msg in current_messages:
        if isinstance(msg, (HumanMessage, AIMessage, ToolMessage)):
            messages_for_llm.append(msg)
            
    user_input = state.get('user_input', '')
    if not current_messages or (current_messages[-1].content != user_input):
        messages_for_llm.append(HumanMessage(content=user_input))
        
    # --- 3. CALL LLM ---
    llm_response = llm.invoke(messages_for_llm)
    clean_content = llm_response.content.replace("`", "").strip()
    
    state['messages'] = current_messages + [AIMessage(content=clean_content)]
    state['last_result'] = clean_content
    
    return state

def routing_by_Supervisor(state: Supervisor) -> str:
    messages = state.get('messages', [])
    if not messages:
        return END
        
    last_message_content = messages[-1].content.strip().lower()
    
    # UPDATED: Added new routes
    if last_message_content == "api":
        return "API_Graph_Node"
    elif last_message_content == "wikipedia":
        return "Wikipedia_Graph_Node"
    elif last_message_content == "rag":
        return "RAG_Graph_Node"
    elif last_message_content == "ingest":
        return "Ingestion_Node"
    else:
        return END

# --- NEW: LINEAR INGESTION NODE ---
def Ingestion_Node(state: Supervisor) -> Supervisor:
    messages = state.get('messages', [])
    user_input = state.get('user_input', '')
    
    # 1. Check for URL
    url_match = re.search(r'(https?://[^\s]+)', user_input)
    
    # 2. Check for Windows file path (handles spaces in folder names)
    file_match = re.search(r'([a-zA-Z]:\\[^<>:"/|?*]+\.[a-zA-Z0-9]+)', user_input)
    
    if url_match:
        url = url_match.group(1)
        print(f"Ingestion Node: Detected URL '{url}'. Starting ingestion...")
        result_text = ingest_document(source_path=url, source_type="url")
        
    elif file_match:
        file_path = file_match.group(1).strip()
        print(f"Ingestion Node: Detected File Path '{file_path}'. Starting ingestion...")
        
        if file_path.lower().endswith('.pdf'):
            source_type = "pdf"
        else:
            source_type = "text"
            
        result_text = ingest_document(source_path=file_path, source_type=source_type)
        
    else:
        result_text = "I couldn't find a valid URL or file path. Please provide a web link or a full local path."
        
    messages.append(AIMessage(content=result_text))
    state['messages'] = messages
    state['last_result'] = result_text
    return state
    
def API_Agent(state: API) -> API:
    System_input = """You are an API Agent. You have access to EXACTLY THREE tools, and no others:
    - get_flight(origin, destination, date, passengers): SEARCHES real flight offers. Does not book or purchase anything.
    - get_hotel(location, check_in, check_out, guests): SEARCHES real hotel offers. Does not book or purchase anything.
    - get_tourist_places(location, local_time): SEARCHES real tourist attractions near a location.

    CRITICAL RULES:
    - These tools only SEARCH/RETRIEVE information. There is NO booking, purchasing, or reservation tool available.
    - NEVER call a tool with any name other than get_flight, get_hotel, or get_tourist_places (e.g. do NOT call "book_flight", "book_hotel", "reserve_flight", or anything similar -- those tools do not exist and calling them will error).
    - If the user says "book it", "confirm", "yes please book", or similar, do NOT call a tool. Instead, reply in plain text that you can search and compare options but can't complete an actual purchase, and point them to the airline/hotel's own site or a travel agent to finalize the booking.
    - If you need more information to run a search (e.g. missing dates, origin, or number of travelers), ask the user before calling a tool."""
    
    system_message = SystemMessage(content=System_input)
    current_messages = state.get('messages', [])
    if not current_messages:
        messages_for_llm = [system_message, HumanMessage(content=state.get('query', ''))]
    else:
        messages_for_llm = [system_message] + current_messages

    try:
        response = api_agent_llm.invoke(messages_for_llm)
    except Exception as e:
        # Most commonly a hallucinated/invalid tool call rejected by the
        # provider (e.g. "book_flight" instead of the real "get_flight").
        # Recover gracefully instead of crashing the whole conversation loop.
        fallback_text = (
            "I ran into an error trying to process that request "
            f"(details: {e}). I can search for flights, hotels, or tourist "
            "places, but I can't complete an actual purchase/booking. "
            "Could you rephrase what you'd like me to search for?"
        )
        response = AIMessage(content=fallback_text)

    state['messages'] = current_messages + [response]
    state['result'] = response.content if response.content else "Tool call generated."
    return state

def API_Agent_TOOLNode(state: API) -> API:
    messages = state.get('messages', [])
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            # NOTE: tool_name matches the @tool function name (get_hotel /
            # get_flight / get_tourist_places), not "book_hotel"/"book_flight".
            if tool_name == "get_hotel":
                response = get_hotel.invoke(tool_args)
            elif tool_name == "get_tourist_places":
                response = get_tourist_places.invoke(tool_args)
            elif tool_name == "get_flight":
                response = get_flight.invoke(tool_args)
            else:
                response = f"Tool {tool_name} not found."
                
            tool_message = ToolMessage(content=str(response), tool_call_id=tool_call["id"])
            messages.append(tool_message)
            
    state['messages'] = messages
    return state
    
def Wikipedia_Agent_TOOLNode(state: Wikipedia) -> Wikipedia:
    messages = state.get('messages', [])
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if tool_name == "search_factual_data":
                response = search_factual_data.invoke(tool_args)
            else:
                response = f"Tool {tool_name} not found."
                
            tool_message = ToolMessage(content=str(response), tool_call_id=tool_call["id"])
            messages.append(tool_message)
            
    state['messages'] = messages
    return state   

def routing_by_API_Agent(state: API) -> str:
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "API_Agent_TOOLNode"
    else:
        return "Supervisor"
    
def routing_by_Wikipedia_Agent(state: Wikipedia) -> str:
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "Wikipedia_Agent_TOOLNode"
    else:
        return "Supervisor"    
        
def Wikipedia_Agent(state: Wikipedia) -> Wikipedia:
    System_input = """You are a Search Agent. You have access to a DuckDuckGo search tool to find strictly factual data.
    Execute searches to answer the user's query accurately based on external facts.
    CRITICAL INSTRUCTION: You must trigger the tool natively via the tool-calling API. 
    DO NOT write out raw text tags like <function=...>. 
    DO NOT hallucinate arguments like 'max_results'. Just provide the 'query'."""
    
    system_message = SystemMessage(content=System_input)
    current_messages = state.get('messages', [])
    if not current_messages:
        messages_for_llm = [system_message, HumanMessage(content=state.get('query', ''))]
    else:
        messages_for_llm = [system_message] + current_messages
        
    response = wikipedia_agent_llm.invoke(messages_for_llm)
    state['messages'] = current_messages + [response]
    state['result'] = response.content if response.content else "Tool call generated."
    return state

# ===========================================================================
# --- REAL TOOL IMPLEMENTATIONS ---
#
# get_tourist_places: fully free, no API key (OpenStreetMap Nominatim +
#   Wikipedia GeoSearch).
#
# get_flight / get_hotel: use Duffel (https://duffel.com), a free self-service
#   sandbox -- single access token, no separate OAuth step.
#   Sign up free at https://app.duffel.com/join (no credit card), then in
#   your dashboard: make sure you're in "Developer test mode" -> Developers ->
#   Access tokens -> New token. Test tokens start with "duffel_test_".
#   Put it in your .env (never hardcode it or paste it in chat/commits):
#     DUFFEL_ACCESS_TOKEN=duffel_test_xxxxxxxx
#   Test mode returns real, working responses from Duffel's own sandbox
#   airline ("Duffel Airways") and a sandbox test hotel, so prices/schedules
#   are not real-world, but the API flow and response shapes are.
# ===========================================================================

DUFFEL_BASE = "https://api.duffel.com"


def _duffel_headers() -> dict:
    token = os.environ.get("DUFFEL_ACCESS_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing DUFFEL_ACCESS_TOKEN env var. "
            "Get a free test token at https://app.duffel.com/join"
        )
    return {
        "Authorization": f"Bearer {token}",
        "Duffel-Version": "v2",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


@lru_cache(maxsize=256)
def _place_iata_code(place_name: str) -> str:
    """Resolve a free-text city/airport name to an IATA code via Duffel Places."""
    resp = requests.get(
        f"{DUFFEL_BASE}/places/suggestions",
        headers=_duffel_headers(),
        params={"query": place_name},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("data", [])
    if not results:
        raise ValueError(f"Could not resolve IATA code for '{place_name}'")
    # Prefer a city-level match (broader), otherwise take the first airport.
    for r in results:
        if r.get("type") == "city":
            return r["iata_code"]
    return results[0]["iata_code"]


@tool
def get_flight(origin: str, destination: str, date: str, passengers: int = 1) -> str:
    """Searches real flight offers between two cities on a given date for a number of passengers."""
    try:
        origin_code = _place_iata_code(origin)
        dest_code = _place_iata_code(destination)

        body = {
            "data": {
                "slices": [
                    {"origin": origin_code, "destination": dest_code, "departure_date": date}
                ],
                "passengers": [{"type": "adult"} for _ in range(passengers)],
                "cabin_class": "economy",
            }
        }

        resp = requests.post(
            f"{DUFFEL_BASE}/air/offer_requests",
            headers=_duffel_headers(),
            params={"return_offers": "true"},
            json=body,
            timeout=30,
        )
        resp.raise_for_status()
        offers = resp.json().get("data", {}).get("offers", [])

        if not offers:
            return f"No flights found from {origin} to {destination} on {date}."

        # cheapest first
        offers.sort(key=lambda o: float(o["total_amount"]))

        lines = [f"Flights from {origin} ({origin_code}) to {destination} ({dest_code}) on {date}:"]
        for offer in offers[:5]:
            price = offer["total_amount"]
            currency = offer["total_currency"]
            slice0 = offer["slices"][0]
            segments = slice0["segments"]
            carrier = segments[0]["operating_carrier"]["name"]
            stops = len(segments) - 1
            stop_text = "nonstop" if stops == 0 else f"{stops} stop(s)"
            lines.append(f"- {carrier} | {stop_text} | {price} {currency}")
        return "\n".join(lines)

    except Exception as e:
        return f"Flight search failed: {e}"


@tool
def get_hotel(location: str, check_in: str, check_out: str, guests: int = 1) -> str:
    """Searches real hotel offers in a city for given check-in/check-out dates and guest count."""
    try:
        # Geocode the location (free, no key) to feed Duffel Stays' lat/lng search.
        geo_resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": location, "format": "json", "limit": 1},
            headers={"User-Agent": "multi-agent-travel-bot/1.0"},
            timeout=15,
        )
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
        if not geo_data:
            return f"Could not find coordinates for '{location}'."
        lat = float(geo_data[0]["lat"])
        lng = float(geo_data[0]["lon"])

        body = {
            "data": {
                "location": {
                    "radius": 5,
                    "geographic_coordinates": {"latitude": lat, "longitude": lng},
                },
                "check_in_date": check_in,
                "check_out_date": check_out,
                "guests": [{"type": "adult"} for _ in range(guests)],
            }
        }

        resp = requests.post(
            f"{DUFFEL_BASE}/stays/search",
            headers=_duffel_headers(),
            json=body,
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json().get("data", {}).get("results", [])

        if not results:
            return f"No available hotel offers in {location} for those dates."

        results.sort(key=lambda r: float(r.get("cheapest_rate_total_amount") or "inf"))

        lines = [f"Hotels in {location} ({check_in} to {check_out}, {guests} guest(s)):"]
        for item in results[:5]:
            name = item["accommodation"]["name"]
            price = item.get("cheapest_rate_total_amount")
            currency = item.get("cheapest_rate_currency")
            if price:
                lines.append(f"- {name}: {price} {currency} total")
            else:
                lines.append(f"- {name}: price unavailable")
        return "\n".join(lines)

    except Exception as e:
        return f"Hotel search failed: {e}"


@tool
def get_tourist_places(location: str, local_time: str = None) -> str:
    """Retrieves real tourist attractions near a given location using free, keyless APIs."""
    try:
        geo_resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": location, "format": "json", "limit": 1},
            headers={"User-Agent": "multi-agent-travel-bot/1.0"},
            timeout=15,
        )
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
        if not geo_data:
            return f"Could not find coordinates for '{location}'."

        lat = geo_data[0]["lat"]
        lon = geo_data[0]["lon"]

        wiki_resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "geosearch",
                "gscoord": f"{lat}|{lon}",
                "gsradius": 10000,
                "gslimit": 10,
                "format": "json",
            },
            headers={"User-Agent": "multi-agent-travel-bot/1.0"},
            timeout=15,
        )
        wiki_resp.raise_for_status()
        places = wiki_resp.json().get("query", {}).get("geosearch", [])

        if not places:
            return f"No notable tourist places found near {location}."

        lines = [f"Tourist places near {location}:"]
        for p in places:
            lines.append(f"- {p['title']} ({p['dist']:.0f}m away)")
        return "\n".join(lines)

    except Exception as e:
        return f"Tourist place search failed: {e}"


@tool
def search_factual_data(query: str) -> str:
    """Searches the web for factual data."""
    max_results = 5
    try:
        with DDGS(timeout=10) as ddgs:
            results = list(ddgs.text(query, safesearch='moderate', max_results=max_results))
        if not results:
            return f"No factual data found for the query: '{query}'."
        formatted_results = f"Factual Search Results for '{query}':\n\n"
        for i, res in enumerate(results, 1):
            formatted_results += f"{i}. Title: {res.get('title', 'N/A')}\n"
            formatted_results += f"   Fact Snippet: {res.get('body', 'N/A')}\n"
            formatted_results += f"   Source URL: {res.get('href', 'N/A')}\n\n"
        return formatted_results
    except Exception as e:
        return f"Web Search Error: {str(e)}"    
    
API_TOOLS = [ get_hotel, get_tourist_places, get_flight ]
WikiPedia_Tool = [ search_factual_data ] 

api_agent_llm = llm.bind_tools(API_TOOLS)
wikipedia_agent_llm = llm.bind_tools(WikiPedia_Tool)

API_graph = StateGraph(API)
API_graph.add_node("API_Agent", API_Agent)
API_graph.add_node("API_Agent_TOOLNode", API_Agent_TOOLNode) 
API_graph.add_edge(START, "API_Agent")
API_graph.add_conditional_edges("API_Agent", routing_by_API_Agent, {"API_Agent_TOOLNode": "API_Agent_TOOLNode", "Supervisor": END})
API_graph.add_edge("API_Agent_TOOLNode", "API_Agent")

Wikipedia_graph = StateGraph(Wikipedia)
Wikipedia_graph.add_node("Wikipedia_Agent", Wikipedia_Agent)
Wikipedia_graph.add_node("Wikipedia_Agent_TOOLNode", Wikipedia_Agent_TOOLNode)
Wikipedia_graph.add_edge(START, "Wikipedia_Agent")
Wikipedia_graph.add_conditional_edges("Wikipedia_Agent", routing_by_Wikipedia_Agent, {"Wikipedia_Agent_TOOLNode": "Wikipedia_Agent_TOOLNode", "Supervisor": END})
Wikipedia_graph.add_edge("Wikipedia_Agent_TOOLNode", "Wikipedia_Agent")

compiled_api_graph = API_graph.compile()
compiled_wikipedia_graph = Wikipedia_graph.compile()

# --- INITIALIZE MAIN GRAPH ---
main_graph = StateGraph(Supervisor)

main_graph.add_node("Supervisor_Agent", Supervisor_Agent)
main_graph.add_node("API_Graph_Node", compiled_api_graph)
main_graph.add_node("Wikipedia_Graph_Node", compiled_wikipedia_graph)
main_graph.add_node("RAG_Graph_Node", compiled_rag_graph) # ADDED
main_graph.add_node("Ingestion_Node", Ingestion_Node)     # ADDED

main_graph.add_edge(START, "Supervisor_Agent")

main_graph.add_conditional_edges(
    "Supervisor_Agent",
    routing_by_Supervisor,
    {
        "API_Graph_Node": "API_Graph_Node",
        "Wikipedia_Graph_Node": "Wikipedia_Graph_Node",
        "RAG_Graph_Node": "RAG_Graph_Node",         # ADDED
        "Ingestion_Node": "Ingestion_Node",         # ADDED
        END: END
    }
)

main_graph.add_edge("API_Graph_Node", "Supervisor_Agent")
main_graph.add_edge("Wikipedia_Graph_Node", "Supervisor_Agent")
main_graph.add_edge("RAG_Graph_Node", "Supervisor_Agent")   # ADDED
main_graph.add_edge("Ingestion_Node", "Supervisor_Agent")   # ADDED

app = main_graph.compile()

# --- Execution Loop ---
print("==================================================")
print("Multi-Agent System Initialized. Ready to chat!")
print("Type 'exit' or 'quit' to stop.")
print("==================================================")

chat_history = []

while True:
    user_query = input("\nYou: ")
    if user_query.lower() in ['exit', 'quit']:
        break
        
    chat_history.append(HumanMessage(content=user_query))
    
    initial_state = {
        "user_input": user_query,
        "messages": chat_history,
        "last_agent": "",
        "last_result": ""
    }
    
    print("\nAgents are thinking...\n")
    try:
        final_state = app.invoke(initial_state)
        final_message = final_state['messages'][-1].content
        print(f"Assistant: {final_message}")
        
        chat_history = final_state['messages']
    except Exception as e:
        print(f"Error: {e}")