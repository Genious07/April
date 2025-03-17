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
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from groq import Groq

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Logging & Environment Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
load_dotenv()

# FastAPI and CORS Initialization
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error, please try again later."})

# Database & FAISS Setup
def get_database():
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    return client["stelle_db"]

db = get_database()
chats_collection = db["chats"]
memory_collection = db["long_term_memory"]
uploads_collection = db["uploads"]

llm_index = faiss.IndexFlatL2(1536)
llm_memory_map = {}
upload_index = faiss.IndexFlatL2(1536)
upload_memory_map = {}  # Maps FAISS index ID to MongoDB document ID

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

# Utility Functions
def get_current_datetime() -> str:
    return datetime.datetime.now().strftime("%B %d, %Y, %I:%M %p")

def filter_think_messages(messages: list) -> list:
    filtered = []
    for msg in messages:
        content = msg.get("content", "")
        cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if cleaned:
            new_msg = msg.copy()
            new_msg["content"] = cleaned
            filtered.append(new_msg)
    return filtered

def split_text(text, chunk_size=500, overlap=100):
    """Split text into chunks for embedding and retrieval."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

# Pydantic Models
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

# Web Content & YouTube Extraction (unchanged)
async def async_get(url: str, timeout: int = 20) -> httpx.Response:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        return await client.get(url, timeout=timeout)

async def extract_web_content_async(url: str) -> str:
    try:
        response = await async_get(url, timeout=20)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in soup(["script", "style"]):
                tag.extract()
            lines = [line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip()]
            content = "\n".join(lines)
            logging.info(f"Web content extracted (first 200 chars): {content[:200]}...")
            return content
        else:
            logging.error(f"Error fetching webpage: {response.status_code}")
            return f"Error: Unable to fetch webpage. Status code: {response.status_code}"
    except Exception as e:
        logging.error(f"Exception in web extraction: {e}", exc_info=True)
        return "Error: Exception occurred while extracting webpage content."

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

# Document Extraction Functions
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

# Image Extraction: OCR & Captioning
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

# Embedding Generation
async def generate_text_embedding(text: str) -> list:
    try:
        import openai
        response = await openai.Embedding.acreate(input=text, model="text-embedding-ada-002")
        embedding = response["data"][0]["embedding"]
        return embedding
    except Exception as e:
        logging.error(f"Embedding generation error: {e}", exc_info=True)
        return []

# Groq API Integration Functions (unchanged)
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

async def free_search(query: str, limit: int = 10) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = "https://html.duckduckgo.com/html"
        data = {"q": query}
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.post(url, data=data, headers=headers, timeout=20)
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
        final_results = "\n".join(results) if results else "No relevant search results found."
        logging.info(f"Free search results (first 200 chars): {final_results[:200]}...")
        return final_results
    except Exception as e:
        logging.error(f"Free search error: {e}", exc_info=True)
        return "Error during search."

async def browse_and_generate(user_query: str) -> str:
    current_date = get_current_datetime()
    query_with_date = f"{user_query.strip()} Today’s date/time is: {current_date}"
    logging.info(f"Browse query: {query_with_date}")
    try:
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

# Multi-Modal Retrieval Integration
async def retrieve_multimodal_context(query: str, user_id: str) -> str:
    """Retrieve relevant chunks from user's uploaded files based on query embedding."""
    try:
        embedding = await generate_text_embedding(query)
        if not embedding or upload_index.ntotal == 0:
            return ""
        query_vector = np.array(embedding, dtype="float32").reshape(1, -1)
        k = 3  # Top 3 matches
        distances, indices = upload_index.search(query_vector, k)
        contexts = []
        to_remove = []
        for idx in indices[0]:
            if idx in upload_memory_map:
                doc_id = upload_memory_map[idx]
                chunk_doc = await uploads_collection.find_one({"_id": doc_id, "user_id": user_id})
                if chunk_doc and chunk_doc.get("usage_count", 0) < 3:
                    contexts.append(
                        f"From uploaded {chunk_doc['modality']} '{chunk_doc['filename']}': {chunk_doc['chunk_text']}"
                    )
                    new_usage_count = chunk_doc["usage_count"] + 1
                    await uploads_collection.update_one(
                        {"_id": doc_id},
                        {"$set": {"usage_count": new_usage_count}}
                    )
                    if new_usage_count >= 3:
                        to_remove.append(idx)
        for idx in to_remove:
            upload_index.remove_ids(np.array([idx], dtype="int64"))
            del upload_memory_map[idx]
        return "\n\n".join(contexts)
    except Exception as e:
        logging.error(f"Error during multimodal retrieval: {e}", exc_info=True)
        return ""

# Long-Term Memory Functions (unchanged)
async def efficient_summarize(previous_summary: str, new_messages: list, user_id: str, max_summary_length: int = 500) -> str:
    user_queries = "\n".join([msg["content"] for msg in new_messages if msg["role"] == "user"])
    context_text = f"User ID: {user_id}\n"
    if previous_summary:
        context_text += f"Previous Summary:\n{previous_summary}\n\n"
    context_text += f"New User Queries:\n{user_queries}"
    summary_prompt = (
        f"Based on the following queries, generate a concise summary (max {max_summary_length} characters) that captures the user's interests and style:\n\n{context_text}"
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
            new_vector = np.array(new_vector, dtype="float32").reshape(1, -1)
            llm_index.remove_ids(np.array([llm_memory_map[user_id]], dtype="int64")) if user_id in llm_memory_map else None
            idx = llm_index.ntotal
            llm_index.add(new_vector)
            llm_memory_map[user_id] = idx
        else:
            await memory_collection.insert_one({
                "user_id": user_id,
                "session_id": session_id,
                "summary": new_summary,
                "vector": new_vector,
                "timestamp": datetime.datetime.now(datetime.timezone.utc)
            })
            new_vector = np.array(new_vector, dtype="float32").reshape(1, -1)
            idx = llm_index.ntotal
            llm_index.add(new_vector)
            llm_memory_map[user_id] = idx
        logging.info(f"Long-term memory updated for user {user_id}")
    except Exception as e:
        logging.error(f"Error storing long-term memory: {e}", exc_info=True)

# Endpoints
@app.post("/upload")
async def upload_file(user_id: str = Form(...), files: list[UploadFile] = File(...)):
    """
    Handle multi-modal file uploads by splitting text into chunks and storing each chunk.
    - Text files (PDF, DOCX, TXT): Extract text and split into 500-char chunks with 100-char overlap.
    - Images: Extract text (OCR or caption) and treat as one chunk.
    - Store embeddings in FAISS and full chunk data in MongoDB.
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
            modality = "document" if file.content_type in allowed_text_types else "image"
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

            # Split text into chunks (images typically yield one chunk)
            chunks = split_text(extracted_text)
            for chunk in chunks:
                embedding_vector = await generate_text_embedding(chunk)
                if not embedding_vector:
                    continue
                doc = {
                    "user_id": user_id,
                    "filename": file.filename,
                    "modality": modality,
                    "chunk_text": chunk,
                    "embedding": embedding_vector,
                    "usage_count": 0,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc)
                }
                result = await uploads_collection.insert_one(doc)
                doc_id = result.inserted_id
                new_vector = np.array(embedding_vector, dtype="float32").reshape(1, -1)
                new_id = upload_index.ntotal
                upload_index.add(new_vector)
                upload_memory_map[new_id] = doc_id

            responses.append({"filename": file.filename, "success": True})
            logging.info(f"File '{file.filename}' uploaded and chunked for user {user_id}")
        except Exception as e:
            logging.error(f"Error processing file '{file.filename}': {e}", exc_info=True)
            responses.append({"filename": file.filename, "success": False})
    return {"results": responses}

@app.post("/generate", response_model=GenerateResponse)
async def generate_response(request: Request, background_tasks: BackgroundTasks):
    """
    Generate a response with enhanced multi-modal context retrieval.
    - Retrieve user-specific chunks from uploaded files.
    - Include full chunk texts in the prompt for detailed responses.
    """
    try:
        current_date = get_current_datetime()
        data = await request.json()
        req = GenerateRequest(**data)
        user_id, session_id, user_message = req.user_id, req.session_id, req.prompt

        if not user_id or not session_id or not user_message:
            raise HTTPException(status_code=400, detail="Invalid request parameters.")

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
                external_content = await extract_web_content_async(url)
                external_content = await content_for_website(external_content)

        # Retrieve multi-modal context from user's uploaded files
        multimodal_context = await retrieve_multimodal_context(user_message, user_id)

        # Construct unified prompt for the LLM
        unified_prompt = f"User Query: {user_message}\n"
        if external_content:
            unified_prompt += f"\n[External Content]:\n{external_content}\n"
        if multimodal_context:
            unified_prompt += f"\n[Retrieved File Contexts]:\n{multimodal_context}\n"
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

        messages = [{
            "role": "system",
            "content": f"You are Stelle, a strategic, empathetic AI assistant. Provide thoughtful guidance blending conventional and unconventional solutions. Current date/time: {current_date}"
        }]
        if long_term_memory:
            messages.append({"role": "system", "content": f"Long-term memory: {long_term_memory}"})
        messages += chat_history
        messages.append({"role": "user", "content": unified_prompt})
        logging.info(f"LLM prompt messages: {messages}")

        # Call the LLM via Groq API
        generate_api_keys = [os.getenv("GROQ_API_KEY_GENERATE_1"), os.getenv("GROQ_API_KEY_GENERATE_2")]
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

        # Update chat history in MongoDB
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

        # Update long-term memory in background if chat history is long
        if chat_entry and len(chat_entry.get("messages", [])) >= 10:
            background_tasks.add_task(store_long_term_memory, user_id, session_id, chat_entry["messages"][-10:])

        return GenerateResponse(response=reply_content)
    except Exception as e:
        logging.error(f"Error in /generate endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error processing your request.")

@app.get("/chat-history")
async def get_chat_history(user_id: str = Query(...), session_id: str = Query(...)):
    try:
        chat_entry = await chats_collection.find_one({"user_id": user_id, "session_id": session_id}, {"messages": 1})
        if chat_entry and "messages" in chat_entry:
            return {"messages": filter_think_messages(chat_entry["messages"])}
        return {"messages": []}
    except Exception as e:
        logging.error(f"Chat history retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving chat history.")

@app.post("/browse", response_model=BrowseResponse)
async def browse_internet(request: Request):
    try:
        current_date = get_current_datetime()
        data = await request.json()
        req = BrowseRequest(**data)
        query = f"{req.query.strip()} Today’s date/time is: {current_date}"
        if re.search(r'https?://[^\s]+', req.query):
            if "youtube.com" in req.query or "youtu.be" in req.query:
                raw_content = await extract_youtube_info_async(req.query)
            else:
                raw_content = await extract_web_content_async(req.query)
        else:
            raw_content = await free_search(query)
        llm_prompt = (
            f"User Query: {query}\n\nExtracted Content:\n{raw_content}\n\n"
            "Provide a detailed and insightful response that addresses the query."
        )
        client_browse = Groq(api_key=os.getenv("GROQ_API_KEY_BROWSE_ENDPOINT"))
        llm_response = await asyncio.to_thread(
            client_browse.chat.completions.create,
            messages=[
                {"role": "system", "content": "You are Stelle, an advanced assistant skilled in content analysis."},
                {"role": "user", "content": llm_prompt}
            ],
            model="deepseek-r1-distill-qwen-32b",
            max_tokens=1500,
            temperature=0.7,
        )
        final_output = llm_response.choices[0].message.content.strip()
        logging.info(f"/browse endpoint LLM response (first 300 chars): {final_output[:300]}...")
        return BrowseResponse(result=final_output)
    except Exception as e:
        logging.error(f"Browse endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during browsing.")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
