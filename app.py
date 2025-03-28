import os
import re
import asyncio
import datetime
import random
import logging
import faiss
import numpy as np
import httpx
from fastapi import FastAPI, Request, HTTPException, Query, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import unquote, urlparse, parse_qs
import fitz                     # For PDF extraction (PyMuPDF)
import docx2txt                 # For DOCX extraction
from PIL import Image           # For image processing
import easyocr                  # For OCR on images
from io import BytesIO
import uuid                     # For unique goal/task IDs

# NEW: BLIP model for image captioning fallback
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Groq API for LLM integration (replace with your actual library/client if needed)
from groq import Groq
from duckduckgo_search import DDGS
import trafilatura
from concurrent.futures import ThreadPoolExecutor

# Initialize ThreadPoolExecutor (add at module level)
executor = ThreadPoolExecutor(max_workers=20)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------------------------------------------------
# Logging & Environment Setup
# -------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
load_dotenv()

# -------------------------------------------------
# FastAPI and CORS Initialization
# -------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Global Error Handler
# -------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error, please try again later."})

# -------------------------------------------------
# Database & FAISS Setup
# -------------------------------------------------
def get_database():
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    return client["stelle_db"]

db = get_database()
chats_collection = db["chats"]
memory_collection = db["long_term_memory"]
uploads_collection = db["uploads"]
goals_collection = db["goals"]  # Collection for goals and tasks

# FAISS indices for long-term memory and uploaded files
llm_index = faiss.IndexFlatL2(1536)
llm_memory_map = {}  # Maps user_id to FAISS index ID for long-term memory

upload_index = faiss.IndexFlatL2(1536)
upload_memory_map = {}  # Maps file_id (FAISS index) to metadata (user_id, filename, modality, snippet, usage_count)

async def load_faiss_indices():
    try:
        async for mem in memory_collection.find():
            vector = np.array(mem["vector"], dtype="float32").reshape(1, -1)
            idx = llm_index.ntotal
            llm_index.add(vector)
            llm_memory_map[mem["user_id"]] = idx
    except Exception as e:
        logging.error(f"Error loading long-term FAISS index: {e}", exc_info=True)

@app.on_event("startup")
async def startup_event():
    await load_faiss_indices()

# -------------------------------------------------
# Utility Functions
# -------------------------------------------------
def get_current_datetime() -> str:
    return datetime.datetime.now().strftime("%B %d, %Y, %I:%M %p")

def filter_think_messages(messages: list) -> list:
    """
    Removes <think> ... </think> content from message strings.
    """
    filtered = []
    for msg in messages:
        content = msg.get("content", "")
        cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if cleaned:
            new_msg = msg.copy()
            new_msg["content"] = cleaned
            filtered.append(new_msg)
    return filtered

def convert_object_ids(document: dict) -> dict:
    """
    Recursively convert ObjectId values to strings for JSON serialization.
    """
    for key, value in document.items():
        if key == "_id":
            document[key] = str(value)
        elif isinstance(value, dict):
            document[key] = convert_object_ids(value)
        elif isinstance(value, list):
            document[key] = [convert_object_ids(item) if isinstance(item, dict) else item for item in value]
    return document

# -------------------------------------------------
# Pydantic Models
# -------------------------------------------------
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

# -------------------------------------------------
# Web Content & YouTube Extraction
# -------------------------------------------------
async def async_get(url: str, timeout: int = 20) -> httpx.Response:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        return await client.get(url, timeout=timeout)
    
async def scrape_url(url: str) -> str:
    """
    Scrape the given URL using Trafilatura for reliable content extraction.
    Replaces extract_web_content_async to address incorrect outputs.
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(url, timeout=20)
            if response.status_code == 200:
                html = response.text
                content = trafilatura.extract(html)
                return content if content else ""
            else:
                logging.error(f"Failed to scrape {url}: Status {response.status_code}")
                return ""
    except Exception as e:
        logging.error(f"scrape_url error for {url}: {e}")
        return ""

# New search function replacing free_search
async def search_duckduckgo(query: str, max_results: int = 5) -> list[str]:
    """
    Search DuckDuckGo using duckduckgo_search library for accurate results.
    Replaces free_search to improve search reliability.
    """
    loop = asyncio.get_event_loop()
    def sync_search():
        with DDGS() as ddgs:
            return [result['href'] for result in ddgs.text(query, max_results=max_results)]
    try:
        urls = await loop.run_in_executor(executor, sync_search)
        return urls
    except Exception as e:
        logging.error(f"search_duckduckgo error for query '{query}': {e}")
        return []



def extract_youtube_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    return match.group(1) if match else None

def extract_actual_url(duckduckgo_url: str) -> str:
    if duckduckgo_url.startswith("/l/?"):
        parsed = urlparse(duckduckgo_url)
        return unquote(parse_qs(parsed.query).get("uddg", [""])[0])
    return duckduckgo_url

def _extract_youtube_info(url: str) -> str:
    try:
        video_id = extract_youtube_video_id(url)
        if not video_id:
            return "Could not extract video ID from URL."
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([item["text"] for item in transcript_list])
        logging.info(f"YouTube transcript extracted (first 200 chars): {transcript[:200]}...")
        return transcript
    except Exception as e:
        logging.error(f"Error extracting YouTube transcript: {e}", exc_info=True)
        return "Error extracting YouTube transcript."

async def extract_youtube_info_async(url: str) -> str:
    return await asyncio.to_thread(_extract_youtube_info, url)

# -------------------------------------------------
# Document Extraction Functions
# -------------------------------------------------
async def extract_text_from_pdf(file: UploadFile) -> str:
    try:
        file.file.seek(0)
        contents = await file.read()
        doc = fitz.open(stream=contents, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        logging.info(f"PDF text extracted (first 200 chars): {text[:200]}...")
        return text
    except Exception as e:
        logging.error(f"PDF extraction error: {e}", exc_info=True)
        return ""

async def extract_text_from_docx(file: UploadFile) -> str:
    try:
        file.file.seek(0)
        contents = await file.read()
        text = docx2txt.process(BytesIO(contents))
        logging.info(f"DOCX text extracted (first 200 chars): {text[:200]}...")
        return text
    except Exception as e:
        logging.error(f"DOCX extraction error: {e}", exc_info=True)
        return ""

async def extract_text_from_txt(file: UploadFile) -> str:
    try:
        file.file.seek(0)
        contents = await file.read()
        text = contents.decode('utf-8')
        logging.info(f"TXT text extracted (first 200 chars): {text[:200]}...")
        return text
    except Exception as e:
        logging.error(f"TXT extraction error: {e}", exc_info=True)
        return ""

# -------------------------------------------------
# Image Extraction: OCR & Captioning
# -------------------------------------------------
BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"
try:
    blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
    blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID)
    logging.info("BLIP model loaded for image captioning.")
except Exception as e:
    logging.error(f"Error loading BLIP model: {e}", exc_info=True)
    blip_processor = None
    blip_model = None

async def generate_image_caption(pil_image: Image.Image) -> str:
    if not blip_processor or not blip_model:
        return "No caption available."
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        blip_model.to(device)
        inputs = blip_processor(pil_image, return_tensors="pt").to(device)
        output_ids = blip_model.generate(**inputs, max_new_tokens=50)
        caption = blip_processor.decode(output_ids[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        logging.error(f"Image captioning error: {e}", exc_info=True)
        return "Error generating image caption."

async def extract_text_from_image(file: UploadFile) -> str:
    try:
        file.file.seek(0)
        contents = await file.read()
        pil_image = Image.open(BytesIO(contents))
        image_np = np.array(pil_image)
        reader = easyocr.Reader(['en'])
        ocr_result = await asyncio.to_thread(reader.readtext, image_np, detail=0)
        text = " ".join(ocr_result).strip()
        if not text:
            logging.info("No OCR text detected; using BLIP captioning.")
            text = await generate_image_caption(pil_image)
        else:
            logging.info(f"Image OCR extracted (first 200 chars): {text[:200]}...")
        return text
    except Exception as e:
        logging.error(f"Error extracting text from image: {e}", exc_info=True)
        return ""

# -------------------------------------------------
# Embedding Generation
# -------------------------------------------------
async def generate_text_embedding(text: str) -> list:
    """
    Uses OpenAI's text-embedding-ada-002 to generate an embedding.
    """
    try:
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = await openai.Embedding.acreate(input=text, model="text-embedding-ada-002")
        embedding = response["data"][0]["embedding"]
        return embedding
    except Exception as e:
        logging.error(f"Embedding generation error: {e}", exc_info=True)
        return []

# -------------------------------------------------
# Groq API Integration Functions
# -------------------------------------------------
async def content_for_website(content: str) -> str:
    prompt = (
        f"Summarize the following content concisely:\n\n{content}\n\n"
        "List key themes and provide a brief final summary."
    )
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY_CONTENT"))
        response = await asyncio.to_thread(
            client.chat.completions.create,
            messages=[
                {"role": "system", "content": "You are a content analysis expert."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
            max_tokens=700,
            temperature=0.8
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Content summary error: {e}", exc_info=True)
        return "Error generating content summary."

async def detailed_explanation(content: str) -> str:
    prompt = (
        "Provide a detailed explanation by listing key themes and challenges, "
        "and then generate a comprehensive summary of the content below:\n\n" + content
    )
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY_EXPLANATION"))
        response = await asyncio.to_thread(
            client.chat.completions.create,
            messages=[
                {"role": "system", "content": "You are an expert analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
            max_tokens=700,
            temperature=0.8
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Detailed explanation error: {e}", exc_info=True)
        return "Error generating detailed explanation."

async def classify_prompt(prompt: str) -> str:
    """
    Classifies if the user query needs "research" or "no research".
    """
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY_CLASSIFY"))
        response = await asyncio.to_thread(
            client.chat.completions.create,
            messages=[
                {"role": "system", "content": "Determine if this query requires real time research. Respond with 'research' or 'no research'."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192"
        )
        reply = response.choices[0].message.content.strip().lower()
        logging.info(f"Classify prompt response: {reply}")
        return reply
    except Exception as e:
        logging.error(f"Classify prompt error: {e}", exc_info=True)
        return "no research"


async def browse_and_generate(user_query: str) -> str:
    """
    Perform web search or extraction, then generate an LLM response.
    Updated to use the web scraping agent system.
    """
    current_date = get_current_datetime()
    query_with_date = f"{user_query.strip()} Today’s date/time is: {current_date}"
    logging.info(f"Browse query: {query_with_date}")
    try:
        url_match = re.search(r'https?://[^\s]+', user_query)
        if url_match:
            url = url_match.group(0)
            if "youtube.com" in url or "youtu.be" in url:
                raw_content = await extract_youtube_info_async(url)
            else:
                raw_content = await scrape_url(url)
        else:
            urls = await search_duckduckgo(user_query, max_results=5)
            contents = await asyncio.gather(*[scrape_url(url) for url in urls])
            raw_content = " ".join([c for c in contents if c])[:4000]

        llm_prompt = (
            f"User Query: {query_with_date}\n\n"
            f"Extracted Content:\n{raw_content}\n\n"
            "Provide a concise and insightful response based on the above."
        )
        client = Groq(api_key=os.getenv("GROQ_API_KEY_BROWSE"))
        response = await asyncio.to_thread(
            client.chat.completions.create,
            messages=[
                {"role": "system", "content": "You are an advanced assistant skilled in content analysis."},
                {"role": "user", "content": llm_prompt}
            ],
            model="llama3-70b-8192",
            max_tokens=1500,
            temperature=0.6
        )
        final_response = response.choices[0].message.content.strip()
        logging.info(f"Browse LLM response (first 300 chars): {final_response[:300]}...")
        return final_response
    except Exception as e:
        logging.error(f"Browse and generate error: {e}", exc_info=True)
        return "Error processing browse and generate request."
# -------------------------------------------------
# Multi-Modal Retrieval Integration
# -------------------------------------------------
async def retrieve_multimodal_context(query: str) -> str:
    """
    Generate query embedding and return combined text snippets from top matching uploaded files.
    """
    try:
        embedding = await generate_text_embedding(query)
        if not embedding or upload_index.ntotal == 0:
            return ""
        query_vector = np.array(embedding, dtype="float32").reshape(1, -1)
        k = 3  # Top 3 matches
        distances, indices = upload_index.search(query_vector, k)
        contexts = []
        for idx in indices[0]:
            if idx in upload_memory_map:
                meta = upload_memory_map[idx]
                usage_count = meta.get("usage_count", 0)
                # Provide context up to a certain usage count, then remove from index to avoid repetition
                if usage_count < 3:
                    meta["usage_count"] = usage_count + 1
                    contexts.append(
                        f"Filename: {meta['filename']}\nModality: {meta['modality']}\nSnippet: {meta['text_snippet']}"
                    )
                if meta.get("usage_count", 0) >= 3:
                    del upload_memory_map[idx]
        return "\n\n".join(contexts)
    except Exception as e:
        logging.error(f"Error during multimodal retrieval: {e}", exc_info=True)
        return ""

# -------------------------------------------------
# Long-Term Memory Functions
# -------------------------------------------------
async def efficient_summarize(previous_summary: str, new_messages: list, user_id: str, max_summary_length: int = 500) -> str:
    """
    Summarize conversation history, including current goals/tasks, in a compact form.
    """
    user_queries = "\n".join([msg["content"] for msg in new_messages if msg["role"] == "user"])
    context_text = f"User ID: {user_id}\n"
    if previous_summary:
        context_text += f"Previous Summary:\n{previous_summary}\n\n"
    context_text += f"New User Queries:\n{user_queries}\n\n"

    # Include current goals in the summary
    active_goals = await goals_collection.find({"user_id": user_id, "status": "active"}).to_list(None)
    goals_context = ""
    if active_goals:
        goals_context = "User's current goals and tasks:\n"
        for goal in active_goals:
            goals_context += f"- Goal: {goal['title']} ({goal['status']})\n"
            for task in goal['tasks']:
                goals_context += f"  - Task: {task['title']} ({task['status']})\n"
    context_text += goals_context

    summary_prompt = (
        f"Based on the following context, generate a concise summary (max {max_summary_length} characters) "
        f"that captures the user's interests, style, and ongoing goals:\n\n{context_text}"
    )
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY_MEMORY_SUMMARY"))
        response = await asyncio.to_thread(
            client.chat.completions.create,
            messages=[
                {"role": "system", "content": "You are an AI that creates personalized conversation summaries."},
                {"role": "user", "content": summary_prompt}
            ],
            model="llama3-70b-8192",
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Long-term memory summarization error: {e}", exc_info=True)
        return previous_summary if previous_summary else "Summary unavailable."

async def store_long_term_memory(user_id: str, session_id: str, new_messages: list):
    """
    Periodically store or update conversation summary in the long-term memory collection.
    """
    try:
        mem_entry = await memory_collection.find_one({"user_id": user_id})
        previous_summary = mem_entry.get("summary", "") if mem_entry else ""
        new_summary = await efficient_summarize(previous_summary, new_messages, user_id)

        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        embedding_response = await openai.Embedding.acreate(input=new_summary, model="text-embedding-ada-002")
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
            new_vector_np = np.array(new_vector, dtype="float32").reshape(1, -1)
            # Remove old vector from FAISS if needed
            if user_id in llm_memory_map:
                llm_index.remove_ids(np.array([llm_memory_map[user_id]], dtype="int64"))
            idx = llm_index.ntotal
            llm_index.add(new_vector_np)
            llm_memory_map[user_id] = idx
        else:
            await memory_collection.insert_one({
                "user_id": user_id,
                "session_id": session_id,
                "summary": new_summary,
                "vector": new_vector,
                "timestamp": datetime.datetime.now(datetime.timezone.utc)
            })
            new_vector_np = np.array(new_vector, dtype="float32").reshape(1, -1)
            idx = llm_index.ntotal
            llm_index.add(new_vector_np)
            llm_memory_map[user_id] = idx

        logging.info(f"Long-term memory updated for user {user_id}")
    except Exception as e:
        logging.error(f"Error storing long-term memory: {e}", exc_info=True)

# -------------------------------------------------
# Endpoints
# -------------------------------------------------
@app.post("/upload")
async def upload_file(user_id: str = Form(...), files: list[UploadFile] = File(...)):
    """
    Endpoint to handle file uploads (PDF, DOCX, TXT, images).
    Extract text (via OCR or direct) and store embeddings in FAISS + MongoDB.
    """
    allowed_text_types = [
        "text/plain",
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]
    allowed_image_types = ["image/jpeg", "image/png", "image/jpg"]
    responses = []

    for file in files:
        try:
            if file.content_type not in allowed_text_types + allowed_image_types:
                responses.append({"filename": file.filename, "success": False})
                continue

            extracted_text = ""
            if file.content_type == "application/pdf":
                extracted_text = await extract_text_from_pdf(file)
            elif file.content_type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                extracted_text = await extract_text_from_docx(file)
            elif file.content_type == "text/plain":
                extracted_text = await extract_text_from_txt(file)
            elif file.content_type in allowed_image_types:
                extracted_text = await extract_text_from_image(file)
            else:
                responses.append({"filename": file.filename, "success": False})
                continue

            if not extracted_text:
                responses.append({"filename": file.filename, "success": False})
                continue

            embedding_vector = await generate_text_embedding(extracted_text)
            if not embedding_vector:
                responses.append({"filename": file.filename, "success": False})
                continue

            new_vector = np.array(embedding_vector, dtype="float32").reshape(1, -1)
            new_id = upload_index.ntotal
            upload_index.add(new_vector)

            modality = "document" if file.content_type in allowed_text_types else "image"
            upload_memory_map[new_id] = {
                "user_id": user_id,
                "filename": file.filename,
                "modality": modality,
                "text_snippet": extracted_text[:200],
                "usage_count": 0
            }

            await uploads_collection.insert_one({
                "user_id": user_id,
                "filename": file.filename,
                "modality": modality,
                "text_snippet": extracted_text[:500],
                "embedding": embedding_vector,
                "timestamp": datetime.datetime.now(datetime.timezone.utc)
            })

            responses.append({"filename": file.filename, "success": True})
            logging.info(f"File '{file.filename}' uploaded for user {user_id}")
        except Exception as e:
            logging.error(f"Error processing file '{file.filename}': {e}", exc_info=True)
            responses.append({"filename": file.filename, "success": False})

    return {"results": responses}


@app.post("/generate", response_model=GenerateResponse)
async def generate_response(request: Request, background_tasks: BackgroundTasks):
    """
    Primary endpoint that:
      - Receives user prompt
      - Checks for goals/duplicates
      - Parses bracketed markers to create/delete/update goals/tasks
      - Optionally does web or multi-modal search
      - Returns final LLM response

    Now also handles new bracket markers for deadlines and progress:
      [TASK_DEADLINE: <task_id>: <deadline>]
      [TASK_PROGRESS: <task_id>: <progress_description>]
    """
    try:
        current_date = get_current_datetime()
        data = await request.json()
        req = GenerateRequest(**data)
        user_id, session_id, user_message = req.user_id, req.session_id, req.prompt

        if not user_id or not session_id or not user_message:
            raise HTTPException(status_code=400, detail="Invalid request parameters.")

        # Retrieve active goals for context
        active_goals = await goals_collection.find({"user_id": user_id, "status": {"$in": ["active","in progress"]}}).to_list(None)
        goals_context = ""
        if active_goals:
            goals_context = "User's current goals and tasks:\n"
            for goal in active_goals:
                goals_context += f"- Goal: {goal['title']} ({goal['status']}) [ID: {goal['goal_id']}]\n"
                for task in goal['tasks']:
                    goals_context += f"  - Task: {task['title']} ({task['status']}) [ID: {task['task_id']}]\n"

        # Extract external content if URL is provided
        external_content = ""
        url_match = re.search(r'https?://[^\s]+', user_message)
        if url_match:
            url = url_match.group(0)
            logging.info(f"Detected URL in prompt: {url}")
            if "youtube.com" in url or "youtu.be" in url:
                external_content = await extract_youtube_info_async(url)
                external_content = await detailed_explanation(external_content)
            else:
                external_content = await scrape_url(url)
                external_content = await content_for_website(external_content)

        # Retrieve multi-modal context from uploaded files
        multimodal_context = await retrieve_multimodal_context(user_message)

        # Construct unified prompt for the LLM
        unified_prompt = f"User Query: {user_message}\n"
        if external_content:
            unified_prompt += f"\n[External Content]:\n{external_content}\n"
        if multimodal_context:
            unified_prompt += f"\n[Retrieved File Context]:\n{multimodal_context}\n"
        unified_prompt += f"\nCurrent Date/Time: {current_date}\n\nProvide a detailed and context-aware response."

        # Optionally determine if additional research is needed
        research_needed = await classify_prompt(user_message)
        if research_needed == "research":
            research_results = await browse_and_generate(user_message)
            if research_results:
                unified_prompt += f"\n\n[Additional Research]:\n{research_results}"

        # Build chat history with long-term memory context
        chat_entry = await chats_collection.find_one({"user_id": user_id, "session_id": session_id})
        chat_history = filter_think_messages(chat_entry.get("messages", []))[-2:] if chat_entry else []
        long_term_memory = ""
        mem_entry = await memory_collection.find_one({"user_id": user_id})
        if mem_entry and "summary" in mem_entry:
            long_term_memory = mem_entry["summary"]

        # System prompt explaining bracketed markers for goals/tasks (including new ones!)
        system_prompt = (
            "You are Stelle, a strategic, empathetic AI assistant with goal/task management.\n"
            "When the user sets a new goal, use '[GOAL_SET: <goal_title>]' plus optional '[TASK: <task_desc>]' lines.\n"
            "To delete a goal: '[GOAL_DELETE: <goal_id>]'. To delete a task: '[TASK_DELETE: <task_id>]'.\n"
            "To add a new task: '[TASK_ADD: <goal_id>: <task_description>]'.\n"
            "To modify a task's title: '[TASK_MODIFY: <task_id>: <new_title_or_description>]'.\n"
            "To start a goal: '[GOAL_START: <goal_id>]'. To start a task: '[TASK_START: <task_id>]'.\n"
            "To complete a goal: '[GOAL_COMPLETE: <goal_id>]'. To complete a task: '[TASK_COMPLETE: <task_id>]'.\n"
            "NEW: To set a deadline on a task: '[TASK_DEADLINE: <task_id>: <deadline>]' (any date/time format).\n"
            "NEW: To log progress on a task: '[TASK_PROGRESS: <task_id>: <progress_description>]'.\n"
            "Only use these bracketed markers if the user explicitly requests those actions.\n"
            f"Current date/time: {current_date}\n"
        )

        messages = [{"role": "system", "content": system_prompt}]
        if long_term_memory:
            messages.append({"role": "system", "content": f"Long-term memory: {long_term_memory}"})
        if goals_context:
            messages.append({"role": "system", "content": goals_context})
        messages += chat_history
        messages.append({"role": "user", "content": unified_prompt})
        logging.info(f"LLM prompt messages: {messages}")

        # Call the LLM via Groq API (replace with your actual generation call)
        generate_api_keys = [os.getenv("GROQ_API_KEY_GENERATE_1"), os.getenv("GROQ_API_KEY_GENERATE_2")]
        generate_api_keys = [k for k in generate_api_keys if k]  # filter out None
        if not generate_api_keys:
            raise HTTPException(status_code=500, detail="No valid GROQ_API_KEY_GENERATE environment variables found.")

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

        # -------------------------------------------------
        # PARSE LLM OUTPUT FOR GOAL/TASK MANAGEMENT MARKERS
        # -------------------------------------------------
        # Track newly created goals so we can map user-supplied identifiers to real UUIDs
        new_goals_map = {}

        # 1) GOAL_SET: <goal_phrase>
        goal_set_matches = re.findall(r"\[GOAL_SET: (.*?)\]", reply_content)
        for goal_phrase in goal_set_matches:
            # Create random goal_id
            goal_id = str(uuid.uuid4())
            # Map the user-supplied phrase to the real ID
            new_goals_map[goal_phrase] = goal_id

            # Check for duplicates (same title) - optional
            existing_goal = await goals_collection.find_one({
                "user_id": user_id,
                "title": goal_phrase,
                "status": {"$in": ["active", "in progress"]}
            })
            if existing_goal:
                logging.info(f"Skipping creation of duplicate goal '{goal_phrase}' for user {user_id}.")
                continue

            new_goal = {
                "user_id": user_id,
                "goal_id": goal_id,
                "title": goal_phrase,
                "description": "",
                "status": "active",  # new goals default to 'active'
                "created_at": datetime.datetime.now(datetime.timezone.utc),
                "updated_at": datetime.datetime.now(datetime.timezone.utc),
                "tasks": []
            }
            # If the LLM included [TASK: ...] lines for immediate tasks under this goal
            task_matches = re.findall(r"\[TASK: (.*?)\]", reply_content)
            for task_desc in task_matches:
                task_id = str(uuid.uuid4())
                new_task = {
                    "task_id": task_id,
                    "title": task_desc,
                    "description": "",
                    "status": "not started",
                    "created_at": datetime.datetime.now(datetime.timezone.utc),
                    "updated_at": datetime.datetime.now(datetime.timezone.utc),
                    "deadline": None,
                    "progress": []
                }
                new_goal["tasks"].append(new_task)

            await goals_collection.insert_one(new_goal)
            logging.info(f"Goal set: '{goal_phrase}' with {len(task_matches)} tasks for user {user_id}")

        # 2) GOAL_DELETE: <goal_id>
        goal_delete_matches = re.findall(r"\[GOAL_DELETE: (.*?)\]", reply_content)
        for gid in goal_delete_matches:
            real_goal_id = new_goals_map.get(gid, gid)
            result = await goals_collection.delete_one({"user_id": user_id, "goal_id": real_goal_id})
            if result.deleted_count > 0:
                logging.info(f"Goal {real_goal_id} deleted for user {user_id}")
            else:
                logging.warning(f"Goal {real_goal_id} not found or could not be deleted.")

        # 3) TASK_DELETE: <task_id>
        task_delete_matches = re.findall(r"\[TASK_DELETE: (.*?)\]", reply_content)
        for tid in task_delete_matches:
            result = await goals_collection.update_one(
                {"user_id": user_id, "tasks.task_id": tid},
                {"$pull": {"tasks": {"task_id": tid}}}
            )
            if result.modified_count > 0:
                logging.info(f"Task {tid} deleted for user {user_id}")
            else:
                logging.warning(f"Task {tid} not found or could not be deleted.")

        # 4) TASK_ADD: <goal_id>: <task_description>
        task_add_matches = re.findall(r"\[TASK_ADD:\s*(.*?):\s*(.*?)\]", reply_content)
        for goal_id_str, task_desc in task_add_matches:
            real_goal_id = new_goals_map.get(goal_id_str, goal_id_str)
            task_id = str(uuid.uuid4())
            new_task = {
                "task_id": task_id,
                "title": task_desc,
                "description": "",
                "status": "not started",
                "created_at": datetime.datetime.now(datetime.timezone.utc),
                "updated_at": datetime.datetime.now(datetime.timezone.utc),
                "deadline": None,
                "progress": []
            }
            result = await goals_collection.update_one(
                {"user_id": user_id, "goal_id": real_goal_id},
                {
                    "$push": {"tasks": new_task},
                    "$set": {"updated_at": datetime.datetime.now(datetime.timezone.utc)}
                }
            )
            if result.modified_count > 0:
                logging.info(f"Added task '{task_desc}' to goal {real_goal_id}")
            else:
                logging.warning(f"Could not add task to goal {real_goal_id} (not found?).")

        # 5) TASK_MODIFY: <task_id>: <new_description>
        task_modify_matches = re.findall(r"\[TASK_MODIFY:\s*(.*?):\s*(.*?)\]", reply_content)
        for tid, new_desc in task_modify_matches:
            result = await goals_collection.update_one(
                {"user_id": user_id, "tasks.task_id": tid},
                {
                    "$set": {
                        "tasks.$.title": new_desc,
                        "tasks.$.updated_at": datetime.datetime.now(datetime.timezone.utc)
                    }
                }
            )
            if result.modified_count > 0:
                logging.info(f"Task {tid} modified to '{new_desc}'")
            else:
                logging.warning(f"Task {tid} not found for modification.")

        # 6) GOAL_START: <goal_id> → set status to "in progress"
        goal_start_matches = re.findall(r"\[GOAL_START: (.*?)\]", reply_content)
        for gid in goal_start_matches:
            real_goal_id = new_goals_map.get(gid, gid)
            result = await goals_collection.update_one(
                {"user_id": user_id, "goal_id": real_goal_id},
                {"$set": {"status": "in progress", "updated_at": datetime.datetime.now(datetime.timezone.utc)}}
            )
            if result.modified_count > 0:
                logging.info(f"Goal {real_goal_id} started (in progress).")
            else:
                logging.warning(f"Goal {real_goal_id} not found for GOAL_START.")

        # 7) TASK_START: <task_id> → set status to "in progress"
        task_start_matches = re.findall(r"\[TASK_START: (.*?)\]", reply_content)
        for tid in task_start_matches:
            result = await goals_collection.update_one(
                {"user_id": user_id, "tasks.task_id": tid},
                {
                    "$set": {
                        "tasks.$.status": "in progress",
                        "tasks.$.updated_at": datetime.datetime.now(datetime.timezone.utc)
                    }
                }
            )
            if result.modified_count > 0:
                logging.info(f"Task {tid} started (in progress).")
            else:
                logging.warning(f"Task {tid} not found for TASK_START.")

        # 8) GOAL_COMPLETE: <goal_id> → set status to "completed"
        goal_complete_matches = re.findall(r"\[GOAL_COMPLETE: (.*?)\]", reply_content)
        for gid in goal_complete_matches:
            real_goal_id = new_goals_map.get(gid, gid)
            result = await goals_collection.update_one(
                {"user_id": user_id, "goal_id": real_goal_id},
                {"$set": {"status": "completed", "updated_at": datetime.datetime.now(datetime.timezone.utc)}}
            )
            if result.modified_count > 0:
                logging.info(f"Goal {real_goal_id} marked as completed.")
            else:
                logging.warning(f"Goal {real_goal_id} not found for GOAL_COMPLETE.")

        # 9) TASK_COMPLETE: <task_id>
        task_complete_matches = re.findall(r"\[TASK_COMPLETE: (.*?)\]", reply_content)
        for tid in task_complete_matches:
            result = await goals_collection.update_one(
                {"user_id": user_id, "tasks.task_id": tid},
                {
                    "$set": {
                        "tasks.$.status": "completed",
                        "tasks.$.updated_at": datetime.datetime.now(datetime.timezone.utc)
                    }
                }
            )
            if result.modified_count > 0:
                logging.info(f"Task {tid} marked as completed.")
            else:
                logging.warning(f"Task {tid} not found for completion.")

        # ---------------------------------------
        # NEW 10) TASK_DEADLINE: <task_id>: <deadline>
        # ---------------------------------------
        task_deadline_matches = re.findall(r"\[TASK_DEADLINE:\s*(.*?):\s*(.*?)\]", reply_content)
        for tid, deadline_str in task_deadline_matches:
            # In production, you might parse/validate date strings, e.g. with dateutil.
            # For now, store as a string or attempt parse:
            try:
                # Example parse attempt (if you want an actual datetime object):
                # from dateutil.parser import parse
                # parsed_deadline = parse(deadline_str)
                # deadline_value = parsed_deadline
                # If parse fails, store as string:
                # ...
                deadline_value = deadline_str  # or the parsed datetime
            except Exception:
                deadline_value = deadline_str

            result = await goals_collection.update_one(
                {"user_id": user_id, "tasks.task_id": tid},
                {
                    "$set": {
                        "tasks.$.deadline": deadline_value,
                        "tasks.$.updated_at": datetime.datetime.now(datetime.timezone.utc)
                    }
                }
            )
            if result.modified_count > 0:
                logging.info(f"Set deadline for Task {tid} to {deadline_value}")
            else:
                logging.warning(f"Task {tid} not found for deadline update.")

        # ---------------------------------------
        # NEW 11) TASK_PROGRESS: <task_id>: <progress_description>
        # ---------------------------------------
        task_progress_matches = re.findall(r"\[TASK_PROGRESS:\s*(.*?):\s*(.*?)\]", reply_content)
        for tid, progress_desc in task_progress_matches:
            progress_entry = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc),
                "description": progress_desc
            }
            result = await goals_collection.update_one(
                {"user_id": user_id, "tasks.task_id": tid},
                {
                    "$push": {"tasks.$.progress": progress_entry},
                    "$set": {"tasks.$.updated_at": datetime.datetime.now(datetime.timezone.utc)}
                }
            )
            if result.modified_count > 0:
                logging.info(f"Added progress entry to Task {tid}: {progress_desc}")
            else:
                logging.warning(f"Task {tid} not found for progress update.")

        # -------------------------------------------------
        # CLEAN THE FINAL REPLY FOR THE USER
        # -------------------------------------------------
        # Remove bracketed lines from the final user-facing message:
        lines = reply_content.split("\n")
        clean_lines = [line for line in lines if not re.match(r"\[.*?: .*?\]", line.strip())]
        reply_content_clean = "\n".join(clean_lines).strip()

        # Update chat history in MongoDB
        new_messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": reply_content_clean}
        ]
        if chat_entry:
            await chats_collection.update_one(
                {"user_id": user_id, "session_id": session_id},
                {
                    "$push": {"messages": {"$each": new_messages}},
                    "$set": {"last_updated": datetime.datetime.now(datetime.timezone.utc)}
                }
            )
        else:
            await chats_collection.insert_one({
                "user_id": user_id,
                "session_id": session_id,
                "messages": new_messages,
                "last_updated": datetime.datetime.now(datetime.timezone.utc)
            })

        # Update long-term memory in background if chat history is long
        if chat_entry and len(chat_entry.get("messages", [])) >= 10:
            background_tasks.add_task(store_long_term_memory, user_id, session_id, chat_entry["messages"][-10:])

        return GenerateResponse(response=reply_content_clean)

    except Exception as e:
        logging.error(f"Error in /generate endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error processing your request.")


@app.get("/chat-history")
async def get_chat_history(user_id: str = Query(...), session_id: str = Query(...)):
    """
    Retrieve filtered chat history (omitting <think> ... </think> content).
    """
    try:
        chat_entry = await chats_collection.find_one({"user_id": user_id, "session_id": session_id}, {"messages": 1})
        if chat_entry and "messages" in chat_entry:
            return {"messages": filter_think_messages(chat_entry["messages"])}
        return {"messages": []}
    except Exception as e:
        logging.error(f"Chat history retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving chat history.")


@app.get("/get-goals")
async def get_goals(user_id: str = Query(...)):
    """
    Retrieve all goals and their tasks for a given user, with date fields serialized.
    """
    try:
        goals = await goals_collection.find({"user_id": user_id}).to_list(None)
        if not goals:
            return {"goals": []}
        # Convert ObjectIds and datetime objects for JSON serialization
        for goal in goals:
            goal = convert_object_ids(goal)
            goal["created_at"] = goal["created_at"].isoformat()
            goal["updated_at"] = goal["updated_at"].isoformat()
            for task in goal["tasks"]:
                if "_id" in task:
                    task["_id"] = str(task["_id"])
                task["created_at"] = task["created_at"].isoformat()
                task["updated_at"] = task["updated_at"].isoformat()
                if task.get("deadline"):
                    # If you store deadlines as strings or datetimes, adapt accordingly
                    if isinstance(task["deadline"], datetime.datetime):
                        task["deadline"] = task["deadline"].isoformat()
                    else:
                        task["deadline"] = str(task["deadline"])
                for progress in task.get("progress", []):
                    progress["timestamp"] = progress["timestamp"].isoformat()
        return {"goals": goals}
    except Exception as e:
        logging.error(f"Error retrieving goals for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving goals.")


# -------------------------------------------------
# Run the Application (for local dev)
# -------------------------------------------------
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
