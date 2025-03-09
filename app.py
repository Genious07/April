import os
import re
import asyncio
import datetime
import random
import logging
import faiss
import numpy as np
import httpx
from fastapi import FastAPI, Request, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import unquote, urlparse, parse_qs
from groq import Groq
from playwright.async_api import async_playwright

# ---------------------------
# SET UP LOGGING & ENVIRONMENT
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
load_dotenv()

# ---------------------------
# OPTIONAL IN-MEMORY CACHES
# ---------------------------
youtube_transcript_cache = {}  # Cache transcripts by video ID
web_content_cache = {}         # Cache web content by URL

# ---------------------------
# FASTAPI APP INITIALIZATION
# ---------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# GLOBAL ERROR HANDLER
# ---------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error, please try again later."}
    )

# ---------------------------
# DATABASE SETUP USING MOTOR
# ---------------------------
def get_database():
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    return client["stelle_db"]

db = get_database()
chats_collection = db["chats"]
memory_collection = db["long_term_memory"]

# ---------------------------
# FAISS INDEX SETUP
# ---------------------------
index = faiss.IndexFlatL2(1536)
memory_map = {}  # Maps user_id to FAISS index

async def load_faiss():
    try:
        async for mem in memory_collection.find():
            vector = np.array(mem["vector"], dtype="float32").reshape(1, -1)
            idx = index.ntotal
            index.add(vector)
            memory_map[mem["user_id"]] = idx
    except Exception as e:
        logging.error(f"Error loading FAISS index: {e}", exc_info=True)

@app.on_event("startup")
async def startup_event():
    await load_faiss()

# ---------------------------
# UTILITY FUNCTIONS
# ---------------------------
def get_current_datetime() -> str:
    now = datetime.datetime.now()
    return now.strftime("%B %d, %Y, %I:%M %p")

def filter_think_messages(messages: list) -> list:
    filtered = []
    for msg in messages:
        content = msg.get("content", "")
        cleaned_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if cleaned_content:
            new_msg = msg.copy()
            new_msg["content"] = cleaned_content
            filtered.append(new_msg)
    return filtered

# ---------------------------
# Pydantic MODELS
# ---------------------------
class GenerateRequest(BaseModel):
    user_id: str
    session_id: str
    prompt: str

class GenerateResponse(BaseModel):
    response: str

class BrowseRequest(BaseModel):
    query: str

class BrowseResponse(BaseModel):
    result: str

# ---------------------------
# ASYNC HTTP REQUEST HELPER
# ---------------------------
async def async_get(url: str, timeout: int = 20) -> httpx.Response:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        response = await client.get(url, timeout=timeout)
    return response

# ---------------------------
# HEADLESS BROWSER EXTRACTION USING PLAYWRIGHT
# ---------------------------
async def extract_dynamic_content(url: str) -> str:
    """Fetch rendered HTML using Playwright."""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until='networkidle')
            # Reduced extra wait time to 0.5 seconds for efficiency
            await asyncio.sleep(0.5)
            content = await page.content()
            await browser.close()
            logging.info(f"Dynamic content extracted from {url}")
            return content
    except Exception as e:
        logging.error(f"Error in extract_dynamic_content for {url}: {e}", exc_info=True)
        return f"Error extracting dynamic content from {url}."

# ---------------------------
# WEB EXTRACTION FUNCTIONS
# ---------------------------
async def extract_web_content_async(url: str) -> str:
    """
    Extract web content using a headless browser first; fall back to static GET.
    Implements in-memory caching to speed up repeated requests.
    """
    if url in web_content_cache:
        logging.info(f"Using cached content for {url}")
        return web_content_cache[url]

    # Try dynamic extraction using Playwright.
    html = await extract_dynamic_content(url)
    if "Error extracting dynamic content" not in html:
        soup = BeautifulSoup(html, 'html.parser')
    else:
        response = await async_get(url, timeout=20)
        soup = BeautifulSoup(response.text, 'html.parser')
    for script in soup(["script", "style"]):
        script.extract()
    lines = [line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip()]
    content = "\n".join(lines)
    web_content_cache[url] = content
    logging.info(f"Extracted {len(content)} characters from {url}")
    return content

def extract_actual_url(duckduckgo_url: str) -> str:
    if duckduckgo_url.startswith("/l/?"):
        parsed_url = urlparse(duckduckgo_url)
        return unquote(parse_qs(parsed_url.query).get("uddg", [""])[0])
    return duckduckgo_url

# ---------------------------
# YOUTUBE TRANSCRIPT EXTRACTION FUNCTIONS
# ---------------------------
def extract_youtube_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    return match.group(1) if match else None

def _extract_youtube_info(url: str) -> str:
    try:
        video_id = extract_youtube_video_id(url)
        if not video_id:
            return "Could not extract video ID from URL."
        # Use cached transcript if available
        if video_id in youtube_transcript_cache:
            logging.info(f"Using cached transcript for video {video_id}")
            return youtube_transcript_cache[video_id]
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([d["text"] for d in transcript_list])
        youtube_transcript_cache[video_id] = transcript
        logging.info(f"Extracted YouTube transcript for video {video_id}: {transcript[:200]}...")
        return transcript
    except Exception as e:
        logging.error(f"Error in _extract_youtube_info: {e}", exc_info=True)
        return "Error extracting YouTube transcript."

async def extract_youtube_info_async(url: str) -> str:
    return await asyncio.to_thread(_extract_youtube_info, url)

# ---------------------------
# GROQ CALL FUNCTIONS
# ---------------------------
async def content_for_website(content: str) -> str:
    prompt = (
        f"Summarize the following content concisely:\n\n{content}\n\n"
        "Then, perform a chain-of-thought analysis by:\n"
        "1. Listing key themes and main ideas.\n"
        "2. Describing the content’s structure and purpose.\n"
        "3. Evaluating clarity and detail.\n"
        "4. Providing a final, concise summary.\n"
        "Include your reasoning and final summary."
    )
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY_CONTENT"))
        response = await asyncio.to_thread(
            client.chat.completions.create,
            messages=[
                {"role": "system", "content": "You are an expert website content analyser"},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
            max_tokens=700,
            temperature=0.8
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error in content_for_website: {e}", exc_info=True)
        return "Error generating website content summary."

async def detailed_explanation(content: str) -> str:
    prompt = (
        "You are an expert analysis assistant. Begin with a chain-of-thought analysis of the content by identifying and listing key themes, challenges, and critical points. "
        "Then, produce a detailed explanation using clear headings and bullet points covering all essential aspects comprehensively.\n\n"
        "Content:\n\n" + content
    )
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY_EXPLANATION"))
        response = await asyncio.to_thread(
            client.chat.completions.create,
            messages=[
                {"role": "system", "content": "You are an expert explanation and critical thinking assistant"},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
            max_tokens=700,
            temperature=0.8
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error in detailed_explanation: {e}", exc_info=True)
        return "Error generating detailed explanation."

async def classify_prompt(prompt: str) -> str:
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY_CLASSIFY"))
        response = await asyncio.to_thread(
            client.chat.completions.create,
            messages=[
                {"role": "system", "content": "You are a prompt analyzer. If this query needs internet research and real time data reply with 'research', otherwise reply with 'no research'."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192"
        )
        reply = response.choices[0].message.content.strip().lower()
        logging.info(f"Classify Prompt Response: {reply}")
        return reply
    except Exception as e:
        logging.error(f"Error in classify_prompt: {e}", exc_info=True)
        return "no research"

async def free_search(query: str, limit: int = 10) -> str:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/90.0.4430.85 Safari/537.36"
        }
        url = "https://html.duckduckgo.com/html"
        data = {"q": query}
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.post(url, data=data, headers=headers, timeout=20)
        response.raise_for_status()
        logging.info(f"DuckDuckGo response length: {len(response.text)}")
        await asyncio.sleep(0.2)
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        # Use more robust extraction with multiple selectors
        for result in soup.find_all("div", class_="result")[:limit]:
            link_tag = result.find("a", class_="result__a")
            if not link_tag:
                continue
            title = link_tag.get_text()
            link = extract_actual_url(link_tag.get("href"))
            snippet = ""
            snippet_tag = result.find("a", class_="result__snippet")
            if snippet_tag:
                snippet = snippet_tag.get_text()
            else:
                snippet_div = result.find("div", class_="result__snippet")
                if snippet_div:
                    snippet = snippet_div.get_text()
            if not snippet:
                snippet = result.get_text(strip=True)[:200]
            results.append(f"Title: {title}\nLink: {link}\nSnippet: {snippet}\n")
        final_results = "\n".join(results) if results else "No relevant search results found."
        logging.info(f"Free search extracted content: {final_results[:200]}...")
        return final_results
    except httpx.RequestError as e:
        logging.error(f"HTTPX RequestError in free_search: {e}", exc_info=True)
        return "Error during search request."
    except Exception as e:
        logging.error(f"Error in free_search: {e}", exc_info=True)
        return "Unexpected error during search."

async def browse_and_generate(user_query: str) -> str:
    current_date = get_current_datetime()
    query_with_date = f"{user_query.strip()} Today’s date/time is: {current_date}"
    logging.info(f"Browse and Generate Query: {query_with_date}")
    try:
        if re.search(r'https?://[^\s]+', user_query):
            if "youtube.com" in user_query or "youtu.be" in user_query:
                raw_content = await extract_youtube_info_async(user_query)
            else:
                raw_content = await extract_web_content_async(user_query)
        else:
            raw_content = await free_search(user_query)
        
        logging.info(f"Raw content extracted: {raw_content[:300]}...")
        llm_prompt = (
            f"User Query: {query_with_date}\n\n"
            f"Extracted Content:\n{raw_content}\n\n"
            "Based on the above, provide a concise, insightful, and context-aware response."
        )
        client = Groq(api_key=os.environ.get("GROQ_API_KEY_BROWSE"))
        response = await asyncio.to_thread(
            client.chat.completions.create,
            messages=[
                {"role": "system", "content": (
                    "You are Stelle's assistant, an advanced AI skilled in analyzing raw content and generating clear, concise, human-like responses. "
                    "Analyze the provided information and generate a response that addresses the query comprehensively."
                )},
                {"role": "user", "content": llm_prompt}
            ],
            model="llama3-70b-8192",
            max_tokens=1500,
            temperature=0.6
        )
        final_response = response.choices[0].message.content.strip()
        logging.info(f"LLM response from browse: {final_response[:300]}...")
        return final_response
    except Exception as e:
        logging.error(f"Error in browse_and_generate: {e}", exc_info=True)
        return "Error processing browse and generate request."

# ---------------------------
# NEW HELPER FUNCTIONS FOR RAG CRAWLING
# ---------------------------
def extract_urls_from_search_results(search_results: str) -> list:
    pattern = r"Link:\s*(https?://\S+)"
    return re.findall(pattern, search_results)

async def crawl_and_extract(query: str, max_pages: int = 5) -> str:
    modified_query = query  # Use the query as is
    search_results = await free_search(modified_query)
    urls = extract_urls_from_search_results(search_results)
    urls = urls[:max_pages]
    if not urls:
        return "No URLs found for crawling."
    
    for url in urls:
        logging.info(f"URL to be crawled: {url}")
    
    tasks = [asyncio.create_task(extract_web_content_async(url)) for url in urls]
    pages_contents = await asyncio.gather(*tasks, return_exceptions=True)
    aggregated = "\n\n".join([content for content in pages_contents if isinstance(content, str)])
    
    if len(aggregated) > 4000:
        aggregated = await content_for_website(aggregated)
    
    return aggregated

# ---------------------------
# LONG-TERM SUMMARY FUNCTIONS
# ---------------------------
async def efficient_summarize(previous_summary: str, new_messages: list, user_id: str, max_summary_length: int = 500) -> str:
    user_queries = "\n".join([msg["content"] for msg in new_messages if msg["role"] == "user"])
    context_text = f"User ID: {user_id}\n"
    if previous_summary:
        context_text += f"Previous Summary:\n{previous_summary}\n\n"
    context_text += f"New User Queries:\n{user_queries}"
    
    summary_prompt = (
        f"Based on the following user queries, generate a concise summary (max {max_summary_length} characters) that captures the user's interests, concerns, and communication style. "
        f"This summary will be used to personalize future interactions.\n\n{context_text}"
    )
    try:
        client_summary = Groq(api_key=os.environ.get("GROQ_API_KEY_MEMORY_SUMMARY"))
        response = await asyncio.to_thread(
            client_summary.chat.completions.create,
            messages=[
                {"role": "system", "content": "You are an AI that creates personalized conversation summaries based solely on user queries."},
                {"role": "user", "content": summary_prompt}
            ],
            model="llama3-70b-8192",
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error in efficient_summarize: {e}", exc_info=True)
        return previous_summary if previous_summary else "Summary unavailable."

async def store_long_term_memory(user_id: str, session_id: str, new_messages: list):
    try:
        mem_entry = await memory_collection.find_one({"user_id": user_id})
        previous_summary = mem_entry["summary"] if mem_entry and "summary" in mem_entry else ""
        new_summary = await efficient_summarize(previous_summary, new_messages, user_id)
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        embedding_response = await openai.Embedding.acreate(
            input=new_summary,
            model="text-embedding-ada-002"
        )
        new_vector = embedding_response["data"][0]["embedding"]

        if mem_entry:
            await memory_collection.update_one(
                {"user_id": user_id},
                {"$set": {
                    "summary": new_summary,
                    "session_id": session_id,
                    "vector": new_vector,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc)
                }}
            )
            update_faiss_memory(user_id, new_vector)
        else:
            await memory_collection.insert_one({
                "user_id": user_id,
                "session_id": session_id,
                "summary": new_summary,
                "vector": new_vector,
                "timestamp": datetime.datetime.now(datetime.timezone.utc)
            })
            update_faiss_memory(user_id, new_vector)
        logging.info(f"✅ Long-term memory updated for user {user_id}")
    except Exception as e:
        logging.error(f"Error storing long-term memory: {e}", exc_info=True)

def update_faiss_memory(user_id, new_vector):
    new_vector = np.array(new_vector, dtype="float32").reshape(1, -1)
    if user_id in memory_map:
        index.remove_ids(np.array([memory_map[user_id]], dtype="int64"))
    idx = index.ntotal
    index.add(new_vector)
    memory_map[user_id] = idx

# ---------------------------
# ENDPOINTS
# ---------------------------
@app.post("/generate", response_model=GenerateResponse)
async def generate_response(request: Request, background_tasks: BackgroundTasks):
    try:
        current_date = get_current_datetime()
        data = await request.json()
        req = GenerateRequest(**data)
        user_id = req.user_id
        session_id = req.session_id
        user_message = req.prompt

        if not user_id or not session_id or not user_message:
            raise HTTPException(status_code=400, detail="Invalid request")

        extracted_text = ""
        if re.search(r'https?://[^\s]+', user_message):
            url_match = re.search(r'(https?://[^\s]+)', user_message)
            if url_match:
                url = url_match.group(0)
                logging.info(f"🔗 Detected URL: {url}")
                if "youtube.com" in url or "youtu.be" in url:
                    extracted_text = await extract_youtube_info_async(url)
                    extracted_text = await detailed_explanation(extracted_text)
                else:
                    extracted_text = await extract_web_content_async(url)
                    extracted_text = await content_for_website(extracted_text)
        
        modified_prompt = (f"Analyze the user query: {user_message} and the provided content: {extracted_text}. "
                           "Read the full context and provide a detailed, informative explanation addressing the query."
                           ) if extracted_text else user_message

        research_needed = await classify_prompt(user_message)
        if research_needed == "research":
            research_results = await browse_and_generate(user_message)
            logging.info(f"Research Results: {research_results[:300]}...")
            if research_results:
                modified_prompt += f"\n*Internal prompt* Additional Research Data: {research_results}"

        chat_entry = await chats_collection.find_one({"user_id": user_id, "session_id": session_id})
        if chat_entry and "messages" in chat_entry:
            filtered_messages = filter_think_messages(chat_entry["messages"])
            chat_history = filtered_messages[-2:]
        else:
            chat_history = []

        long_term_memory = ""
        mem_entry = await memory_collection.find_one({"user_id": user_id})
        if mem_entry and "summary" in mem_entry:
            long_term_memory = mem_entry["summary"]

        messages = [{
            "role": "system",
            "content": (
                "You are Stelle, a highly strategic and empathetic AI assistant. Your mission is to provide thoughtful guidance by blending conventional and unconventional solutions. "
                "For strategic questions, use an internal chain-of-thought analysis without revealing it. For general conversations, respond naturally. "
                "Internal prompt - Today’s date/time is: " + current_date
            )
        }]
        if long_term_memory:
            messages.append({"role": "system", "content": f"Long-term memory: {long_term_memory}"})
        messages += chat_history
        messages.append({"role": "user", "content": modified_prompt})
        logging.info(f"Final Messages Sent to LLM: {messages}")

        generate_api_keys = [
            os.environ.get("GROQ_API_KEY_GENERATE_1"),
            os.environ.get("GROQ_API_KEY_GENERATE_2"),
        ]
        selected_key = random.choice(generate_api_keys)
        client_generate = Groq(api_key=selected_key)
        
        response = await asyncio.to_thread(
            client_generate.chat.completions.create,
            messages=messages,
            model="deepseek-r1-distill-llama-70b",
            max_completion_tokens=4000,
            temperature=0.7,
        )
        reply_content = response.choices[0].message.content.strip()

        new_messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": reply_content}
        ]
        if chat_entry:
            await chats_collection.update_one(
                {"user_id": user_id, "session_id": session_id},
                {"$push": {"messages": {"$each": new_messages}},
                 "$set": {"last_updated": datetime.datetime.now(datetime.timezone.utc)}}
            )
        else:
            await chats_collection.insert_one({
                "user_id": user_id,
                "session_id": session_id,
                "messages": new_messages,
                "last_updated": datetime.datetime.now(datetime.timezone.utc)
            })

        if chat_entry and len(chat_entry.get("messages", [])) >= 10:
            background_tasks.add_task(store_long_term_memory, user_id, session_id, chat_entry["messages"][-10:])

        return GenerateResponse(response=reply_content)
    except Exception as e:
        logging.error(f"Error in generate_response endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while processing your request.")

@app.get("/chat-history")
async def get_chat_history(user_id: str = Query(...), session_id: str = Query(...)):
    try:
        chat_entry = await chats_collection.find_one({"user_id": user_id, "session_id": session_id}, {"messages": 1})
        if chat_entry and "messages" in chat_entry:
            filtered_messages = filter_think_messages(chat_entry["messages"])
            return {"messages": filtered_messages}
        else:
            return {"messages": []}
    except Exception as e:
        logging.error(f"Error in get_chat_history endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred retrieving chat history.")

@app.post("/browse", response_model=BrowseResponse)
async def browse_internet(request: Request):
    try:
        current_date = get_current_datetime()
        data = await request.json()
        req = BrowseRequest(**data)
        original_query = req.query.strip()
        if not re.search(r'https?://[^\s]+', original_query):
            query = original_query + " Today’s date/time is: " + current_date
        else:
            query = original_query
        
        if re.search(r'https?://[^\s]+', query):
            if "youtube.com" in query or "youtu.be" in query:
                raw_content = await extract_youtube_info_async(query)
            else:
                raw_content = await extract_web_content_async(query)
        else:
            raw_content = await free_search(query)
        
        logging.info(f"Browsing raw content: {raw_content[:300]}...")
        llm_prompt = (
            f"User Query: {query}\n\n"
            f"Extracted Content:\n{raw_content}\n\n"
            "Based on the above, please provide a detailed, insightful, and context-aware response that directly addresses the user's query."
        )
        client_browse = Groq(api_key=os.environ.get("GROQ_API_KEY_BROWSE_ENDPOINT"))
        llm_response = await asyncio.to_thread(
            client_browse.chat.completions.create,
            messages=[
                {"role": "system", "content": (
                    "You are Stelle, an advanced AI assistant skilled in analyzing raw content and generating clear responses. "
                    "Analyze the provided content and generate a detailed answer for the query."
                )},
                {"role": "user", "content": llm_prompt}
            ],
            model="deepseek-r1-distill-qwen-32b",
            max_tokens=1500,
            temperature=0.7,
        )
        final_output = llm_response.choices[0].message.content.strip()
        logging.info(f"LLM response from /browse endpoint: {final_output[:300]}...")
        return BrowseResponse(result=final_output)
    except Exception as e:
        logging.error(f"Error in browse_internet endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred during browsing.")

@app.post("/crawl", response_model=BrowseResponse)
async def crawl_and_generate_endpoint(request: Request):
    """
    Integrates the RAG pipeline with a real-time web crawler.
    It accepts a user query, performs a web crawl to fetch and aggregate multiple pages,
    and then uses the aggregated content to generate a context-rich answer.
    """
    try:
        current_date = get_current_datetime()
        data = await request.json()
        req = BrowseRequest(**data)
        query = req.query.strip()
        
        aggregated_content = await crawl_and_extract(query, max_pages=5)
        logging.info(f"Crawled and aggregated content: {aggregated_content[:300]}...")
        
        llm_prompt = (
            f"User Query: {query}\n"
            f"Current Date/Time: {current_date}\n\n"
            f"Aggregated Web Content:\n{aggregated_content}\n\n"
            "Based on the above, provide a detailed and insightful answer that integrates the real-time information from multiple sources."
        )
        
        client_crawl = Groq(api_key=os.environ.get("GROQ_API_KEY_CRAWL"))
        llm_response = await asyncio.to_thread(
            client_crawl.chat.completions.create,
            messages=[
                {"role": "system", "content": "You are a RAG system that uses crawled web content to provide up-to-date answers."},
                {"role": "user", "content": llm_prompt}
            ],
            model="llama3-70b-8192",
            max_tokens=1500,
            temperature=0.7,
        )
        final_output = llm_response.choices[0].message.content.strip()
        logging.info(f"Final output from /crawl endpoint: {final_output[:300]}...")
        return BrowseResponse(result=final_output)
    except Exception as e:
        logging.error(f"Error in crawl_and_generate_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during the crawling process.")

# ---------------------------
# RUN THE APPLICATION
# ---------------------------
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
