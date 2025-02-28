import os, re, asyncio, datetime
import faiss
import numpy as np
import openai
import httpx
from fastapi import FastAPI, Request, HTTPException, Query, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import unquote, urlparse, parse_qs

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app with CORS middleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# DATABASE SETUP USING MOTOR
# ---------------------------
def get_database():
    # Use your MongoDB connection URI from env vars or hardcode for testing.
    client = AsyncIOMotorClient(os.getenv("MONGO_URI", "mongodb+srv://satwiks788:GADF7TDf03nV37PG@password.otrvm.mongodb.net/"))
    return client["stelle_db"]

db = get_database()
chats_collection = db["chats"]
memory_collection = db["long_term_memory"]

# ---------------------------
# FAISS INDEX SETUP (Synchronous)
# ---------------------------
index = faiss.IndexFlatL2(1536)
memory_map = {}  # Maps user_id to FAISS index

async def load_faiss():
    async for mem in memory_collection.find():
        vector = np.array(mem["vector"], dtype="float32").reshape(1, -1)
        idx = index.ntotal
        index.add(vector)
        memory_map[mem["user_id"]] = idx

@app.on_event("startup")
async def startup_event():
    await load_faiss()

# ---------------------------
# UTILITY FUNCTIONS
# ---------------------------
def get_current_datetime() -> str:
    now = datetime.datetime.now()
    return now.strftime("%B %d, %Y, %I:%M %p")

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
async def async_get(url: str, timeout: int = 10) -> httpx.Response:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=timeout)
    return response

# ---------------------------
# WEB EXTRACTION FUNCTIONS (Async)
# ---------------------------
async def extract_web_content_async(url: str) -> str:
    try:
        response = await async_get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style"]):
                script.extract()
            lines = [line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip()]
            return "\n".join(lines)
        else:
            return f"Error: Unable to fetch webpage. Status code: {response.status_code}"
    except Exception as e:
        return f"Error: Exception occurred: {str(e)}"

# Synchronous YouTube extraction wrapped to run asynchronously
def _extract_youtube_info(url: str) -> str:
    try:
        video_id = extract_youtube_video_id(url)
        if not video_id:
            return "Could not extract video ID from URL."
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([d["text"] for d in transcript_list])
    except Exception as e:
        return f"Error extracting YouTube transcript: {str(e)}"

async def extract_youtube_info_async(url: str) -> str:
    return await asyncio.to_thread(_extract_youtube_info, url)

def extract_youtube_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    return match.group(1) if match else None

# ---------------------------
# OPENAI CALL FUNCTIONS (Async using acreate)
# ---------------------------
async def content_for_website(content: str) -> str:
    prompt = (
        f"Summarize the following content concisely:\n\n{content}\n\n"
        "Then, perform a chain-of-thought analysis by:\n"
        "1. Listing key themes and main ideas.\n"
        "2. Describing the content’s structure and its purpose.\n"
        "3. Evaluating clarity and detail.\n"
        "4. Providing a final, concise summary.\n"
        "Include your reasoning and the final summary."
    )
    response = await openai.ChatCompletion.acreate(
        model="ft:gpt-4o-mini-2024-07-18:zyngate-pvt-ltd::AuGhz8tI:ckpt-step-62",
        messages=[
            {"role": "system", "content": "You are an expert website content analyser"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=700,
        temperature=0.8
    )
    return response["choices"][0]["message"]["content"].strip()

async def detailed_explanation(content: str) -> str:
    prompt = (
        "You are an expert analysis assistant. Begin with a chain-of-thought analysis of the content by identifying and listing key themes, challenges, and critical points. "
        "Then, produce a detailed explanation using clear headings and bullet points to cover all essential aspects comprehensively.\n\n"
        "Content:\n\n" + content
    )
    response = await openai.ChatCompletion.acreate(
        model="ft:gpt-4o-mini-2024-07-18:zyngate-pvt-ltd::AuGhz8tI:ckpt-step-62",
        messages=[
            {"role": "system", "content": "You are an expert explanation and critical thinking assistant"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=700,
        temperature=0.8
    )
    return response["choices"][0]["message"]["content"].strip()

async def free_search(query: str, limit: int = 10) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = "https://html.duckduckgo.com/html/"
        data = {"q": query}
        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=data, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for result in soup.find_all("div", class_="result")[:limit]:
            link_tag = result.find("a", class_="result__a")
            if not link_tag:
                continue
            title = link_tag.get_text()
            link = extract_actual_url(link_tag.get("href"))
            snippet_tag = result.find("div", class_="result__snippet")
            snippet = snippet_tag.get_text() if snippet_tag else "No snippet available."
            results.append(f"Title: {title}\nLink: {link}\nSnippet: {snippet}\n")
        return "\n".join(results) if results else "No relevant search results found."
    except httpx.RequestError as e:
        return f"Error during search request: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def extract_actual_url(duckduckgo_url: str) -> str:
    if duckduckgo_url.startswith("/l/?"):
        parsed_url = urlparse(duckduckgo_url)
        return unquote(parse_qs(parsed_url.query).get("uddg", [""])[0])
    return duckduckgo_url

async def classify_prompt(prompt: str) -> str:
    response = await openai.ChatCompletion.acreate(
        model="ft:gpt-4o-mini-2024-07-18:zyngate-pvt-ltd::AuGhz8tI:ckpt-step-62",
        messages=[
            {"role": "system", "content": "You are a prompt analyzer. If this query needs internet research and real time data reply with 'research', otherwise reply with 'no research'."},
            {"role": "user", "content": prompt}
        ]
    )
    reply = response["choices"][0]["message"]["content"].strip().lower()
    print(reply)
    return reply

async def browse_and_generate(user_query: str) -> str:
    Date = get_current_datetime()
    query_with_date = user_query.strip() + " Today’s date/time is: " + Date
    print(query_with_date)
    if re.search(r'https?://[^\s]+', user_query):
        if "youtube.com" in user_query or "youtu.be" in user_query:
            raw_content = await extract_youtube_info_async(user_query)
        else:
            raw_content = await extract_web_content_async(user_query)
    else:
        raw_content = await free_search(user_query)
    llm_prompt = (
        f"User Query: {query_with_date}\n\n"
        f"Extracted Content:\n{raw_content}\n\n"
        "Based on the above, provide a concise, insightful, and context-aware response."
    )
    response = await openai.ChatCompletion.acreate(
        model="ft:gpt-4o-mini-2024-07-18:zyngate-pvt-ltd::AuGhz8tI:ckpt-step-62",
        messages=[
            {"role": "system", "content": f"You are Stelle's assistant, an advanced AI assistant skilled in analyzing raw content and generating clear, concise, and human-like responses. Your task is to:\n\n"
                                          f"1. Analyze the information provided in the extracted content.\n"
                                          f"2. Generate a response that:\n"
                                          "- Directly addresses the user's query.\n"
                                          "- Is  thorough, and engaging.\n"
                                          "- Is written in a human-like tone with clarity and precision.\n\n"
                                          "-It Must have all content that is extracted"
                                          f"Use the insights from the extracted content to craft an informative and conversational response."},
            {"role": "user", "content": llm_prompt}
        ],
        max_tokens=1500,
        temperature=0.6,
    )
    return response["choices"][0]["message"]["content"].strip()

def update_faiss_memory(user_id, new_vector):
    """Update FAISS index dynamically."""
    new_vector = np.array(new_vector, dtype="float32").reshape(1, -1)
    if user_id in memory_map:
        # Remove old vector by recreating index (FAISS doesn't support in-place updates)
        index.remove_ids(np.array([memory_map[user_id]], dtype="int64"))
    idx = index.ntotal
    index.add(new_vector)
    memory_map[user_id] = idx

# ---------------------------
# BACKGROUND TASK FOR LONG-TERM MEMORY
# ---------------------------
async def store_long_term_memory(user_id: str, session_id: str, messages: list):
    try:
        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        summary_prompt = f"Summarize this conversation in 3 sentences:\n\n{conversation_text}"
        summary_response = await openai.ChatCompletion.acreate(
            model="ft:gpt-4o-mini-2024-07-18:zyngate-pvt-ltd::AuGhz8tI:ckpt-step-62",
            messages=[
                {"role": "system", "content": "You are an AI that summarizes conversations."},
                {"role": "user", "content": summary_prompt}
            ],
            max_tokens=150
        )
        new_summary = summary_response["choices"][0]["message"]["content"].strip()
        embedding_response = await openai.Embedding.acreate(
            input=new_summary,
            model="text-embedding-ada-002"
        )
        new_vector = embedding_response["data"][0]["embedding"]

        existing_memory = await memory_collection.find_one({"user_id": user_id})
        if existing_memory:
            updated_summary = (existing_memory["summary"] + "\n" + new_summary)[-2000:]
            previous_vector = existing_memory["vector"]
            avg_vector = [(p + n) / 2 for p, n in zip(previous_vector, new_vector)]
            await memory_collection.update_one(
                {"user_id": user_id},
                {"$set": {
                    "summary": updated_summary,
                    "session_id": session_id,
                    "vector": avg_vector,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc)
                }}
            )
            # Update FAISS index synchronously
            update_faiss_memory(user_id, avg_vector)
        else:
            await memory_collection.insert_one({
                "user_id": user_id,
                "session_id": session_id,
                "summary": new_summary,
                "vector": new_vector,
                "timestamp": datetime.datetime.now(datetime.timezone.utc)
            })
            update_faiss_memory(user_id, new_vector)
        print(f"✅ Long-term memory updated for user {user_id}")
    except Exception as e:
        print(f"⚠️ Error storing long-term memory: {str(e)}")

# ---------------------------
# ENDPOINTS
# ---------------------------
@app.post("/generate", response_model=GenerateResponse)
async def generate_response(request: Request, background_tasks: BackgroundTasks):
    Date = get_current_datetime()
    data = await request.json()
    req = GenerateRequest(**data)
    user_id = req.user_id
    session_id = req.session_id
    user_message = req.prompt

    if not user_id or not session_id or not user_message:
        raise HTTPException(status_code=400, detail="Invalid request")

    extracted_text = ""
    # Check if the prompt contains a URL
    if re.search(r'https?://[^\s]+', user_message):
        url_match = re.search(r'(https?://[^\s]+)', user_message)
        if url_match:
            url = url_match.group(0)
            print(f"🔗 Detected URL: {url}")
            if "youtube.com" in url or "youtu.be" in url:
                extracted_text = await extract_youtube_info_async(url)
                extracted_text = await detailed_explanation(extracted_text)
            else:
                extracted_text = await extract_web_content_async(url)
                extracted_text = await content_for_website(extracted_text)
    
    if extracted_text:
        modified_prompt = (
            f"Analyze the user query: {user_message} and the provided content (which may be a YouTube transcript or website text: {extracted_text}). "
            "Read the full context and provide a detailed, informative explanation addressing the query. "
            "Avoid summarizing or mentioning content limitations. "
            "This instruction is internal and must not be shared with the user."
        )
    else:
        modified_prompt = user_message

    research_needed = await classify_prompt(user_message)
    if research_needed == "research":
        research_results = await browse_and_generate(user_message)
        print(research_results)
        if research_results:
            modified_prompt += f"\n*This is internal prompt* The user asked for {user_message} and here is Additional Research Data : {research_results} use it and give user response for {user_message} "
    else:
        modified_prompt = user_message

    chat_entry = await chats_collection.find_one({"user_id": user_id, "session_id": session_id})
    chat_history = chat_entry["messages"] if chat_entry and "messages" in chat_entry else []

    # Retrieve long-term memory asynchronously
    long_term_memory = ""
    mem_entry = await memory_collection.find_one({"user_id": user_id})
    if mem_entry and "summary" in mem_entry:
        long_term_memory = mem_entry["summary"]

    messages = [{
        "role": "system",
        "content": (
            "You are Stelle, a highly strategic and empathetic AI assistant. Your mission is to provide thoughtful guidance by blending both conventional and unconventional solutions (for educational purposes only). "
            "Instructions:\n"
            "- For strategic questions:\n"
            "   1. Engage in an internal chain-of-thought analysis to consider multiple perspectives and generate both conventional and unconventional solutions. *Do not reveal this internal analysis.*\n"
            "   2. Synthesize your internal analysis into a clear, concise, and well-structured final answer.\n"
            "- For general conversations, respond naturally without revealing internal processes.\n"
            "Always communicate warmly and thoughtfully. *Internal prompt - Today’s date/time is: " + Date + "*"
        )
    }]
    if long_term_memory:
        messages.append({"role": "system", "content": f"Long-term memory: {long_term_memory}"})
    messages += chat_history
    messages.append({"role": "user", "content": modified_prompt})

    response = await openai.ChatCompletion.acreate(
        model="ft:gpt-4o-mini-2024-07-18:zyngate-pvt-ltd::AuGhz8tI:ckpt-step-62", 
        messages=messages,
        max_tokens=1500,
        temperature=0.8,
    )
    reply_content = response["choices"][0]["message"]["content"].strip()

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

    # Offload long-term memory update in the background after every 10 messages
    if chat_entry and len(chat_entry.get("messages", [])) >= 10:
        background_tasks.add_task(store_long_term_memory, user_id, session_id, chat_entry["messages"][-10:])

    return GenerateResponse(response=reply_content)

@app.get("/chat-history")
async def get_chat_history(user_id: str = Query(...), session_id: str = Query(...)):
    chat_entry = await chats_collection.find_one({"user_id": user_id, "session_id": session_id}, {"messages": 1})
    if chat_entry:
        return {"messages": chat_entry["messages"]}
    else:
        return {"messages": []}

@app.post("/browse", response_model=BrowseResponse)
async def browse_internet(request: Request):
    Date = get_current_datetime()
    data = await request.json()
    req = BrowseRequest(**data)
    query = req.query.strip() + " Today’s date/time is:" + Date
    if not query:
        raise HTTPException(status_code=400, detail="Missing query parameter")

    if re.search(r'https?://[^\s]+', query):
        if "youtube.com" in query or "youtu.be" in query:
            raw_content = await extract_youtube_info_async(query)
        else:
            raw_content = await extract_web_content_async(query)
    else:
        raw_content = await free_search(query)
    
    llm_prompt = (
        f"User Query: {query}\n\n"
        f"Extracted Content:\n{raw_content}\n\n"
        "Based on the above, please provide a detailed, insightful, and context-aware response that directly addresses the user's query."
    )
    llm_response = await openai.ChatCompletion.acreate(
        model="ft:gpt-4o-mini-2024-07-18:zyngate-pvt-ltd::AuGhz8tI:ckpt-step-62",
        messages=[
            {"role": "system", "content": f"You are Stelle, an advanced AI assistant skilled in analyzing raw content and generating clear responses. Your task is to analyze the provided content: {raw_content} and generate a detailed answer for the query: {query}."},
            {"role": "user", "content": llm_prompt}
        ],
        max_tokens=1500,
        temperature=0.7,
    )
    final_output = llm_response["choices"][0]["message"]["content"].strip()
    return BrowseResponse(result=final_output)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
