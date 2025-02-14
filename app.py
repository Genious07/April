import openai
import faiss
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Connect to MongoDB
client = MongoClient("mongodb+srv://satwiks788:GADF7TDf03nV37PG@password.otrvm.mongodb.net/")
db = client["stelle_db"]
chats_collection = db["chats"]
memory_collection = db["long_term_memory"]

# Initialize FAISS Index
index = faiss.IndexFlatL2(1536)
memory_map = {}  # Maps user_id to FAISS index

# Load existing memory from MongoDB into FAISS
for mem in memory_collection.find():
    vector = np.array(mem["vector"], dtype="float32").reshape(1, -1)
    idx = index.ntotal
    index.add(vector)
    memory_map[mem["user_id"]] = idx


def retrieve_long_term_memory(user_id):
    """Retrieve existing long-term memory for the user."""
    memory_entry = memory_collection.find_one({"user_id": user_id})
    return memory_entry["summary"] if memory_entry else ""


def update_faiss_memory(user_id, new_vector):
    """Update FAISS index dynamically."""
    new_vector = np.array(new_vector, dtype="float32").reshape(1, -1)
    if user_id in memory_map:
        # Remove old vector by recreating index (FAISS doesn't support in-place updates)
        index.remove_ids(np.array([memory_map[user_id]], dtype="int64"))
    idx = index.ntotal
    index.add(new_vector)
    memory_map[user_id] = idx


def store_long_term_memory(user_id, session_id, messages):
    """Summarize & store/update conversation in long-term memory, dynamically updating FAISS."""
    try:
        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        # Generate a summary
        summary_prompt = f"Summarize this conversation in 3 sentences:\n\n{conversation_text}"
        summary_response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an AI that summarizes conversations."},
                {"role": "user", "content": summary_prompt}
            ],
            max_tokens=150
        )
        new_summary = summary_response["choices"][0]["message"]["content"].strip()

        # Generate embedding vector
        embedding_response = openai.Embedding.create(
            input=new_summary,
            model="text-embedding-ada-002"
        )
        vector = embedding_response["data"][0]["embedding"]

        # Check if memory entry exists
        existing_memory = memory_collection.find_one({"user_id": user_id})
        
        if existing_memory:
            # Append to existing summary
            updated_summary = existing_memory["summary"] + "\n" + new_summary
            memory_collection.update_one(
                {"user_id": user_id},
                {"$set": {"summary": updated_summary, "timestamp": datetime.utcnow()}}
            )
            update_faiss_memory(user_id, vector)
        else:
            # Insert new memory
            memory_collection.insert_one({
                "user_id": user_id,
                "session_id": session_id,
                "summary": new_summary,
                "vector": vector,
                "timestamp": datetime.utcnow()
            })
            update_faiss_memory(user_id, vector)

        print(f"✅ Long-term memory updated for user {user_id}")
    
    except Exception as e:
        print(f"⚠️ Error storing long-term memory: {str(e)}")


# ---- Added Functions for Web Browsing & YouTube Extraction ----

def is_url(text):
    """Check if the given text contains a URL."""
    pattern = r'https?://[^\s]+'
    return re.search(pattern, text) is not None

def is_youtube_url(url):
    """Determine if a URL is a YouTube link."""
    return "youtube.com" in url or "youtu.be" in url

def extract_web_content(url):
    """Extract main text content from a webpage using BeautifulSoup."""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Remove unwanted script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text(separator="\n")
            # Clean and collapse whitespace/newlines
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            cleaned_text = "\n".join(lines)
            return cleaned_text
        else:
            return f"Error: Unable to fetch webpage. Status code: {response.status_code}"
    except Exception as e:
        return f"Error: Exception occurred: {str(e)}"

def extract_youtube_video_id(url):
    """Extract the video ID from a YouTube URL."""
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if match:
        return match.group(1)
    return None

def extract_youtube_info(url):
    """Extract transcript text from a YouTube video if available."""
    try:
        video_id = extract_youtube_video_id(url)
        if not video_id:
            return "Could not extract video ID from URL."
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([d["text"] for d in transcript_list])
        return transcript_text
    except Exception as e:
        return f"Error extracting YouTube transcript: {str(e)}"

def content_for_website(content):
    """Summarize extracted content using OpenAI's ChatCompletion."""
    try:
        prompt = (
    f"Summarize the following content in a concise and informative way:\n\n{content}\n\n"
    "Then, perform a detailed chain-of-thought analysis by following these steps:\n"
    "1. **Initial Reading:** Carefully read and understand the content provided above.\n"
    "2. **Identify Key Themes:** List the main topics, ideas, and themes that emerge from the content.\n"
    "3. **Analyze Structure:** Describe how the content is organized (e.g., headings, bullet points, sections) and what each part contributes to the overall message.\n"
    "4. **Evaluate Clarity and Detail:** Assess whether the content is clear, sufficiently detailed, and whether any critical information might be missing.\n"
    "5. **Provide a Concise Summary:** Finally, generate a concise summary that captures the essence of the content, making sure it is both informative and easy to understand.\n\n"
    "Your final output should include your step-by-step reasoning (chain-of-thought) and the final concise summary."
)

        response = openai.ChatCompletion.create(
            model="ft:gpt-4o-mini-2024-07-18:zyngate-pvt-ltd::AuGhz8tI:ckpt-step-62",
            messages=[
                {"role": "system", "content": "You are a expert website developer and SEO ranker"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=700,
            temperature=0.8
        )
        summary = response["choices"][0]["message"]["content"].strip()
        return summary
    except Exception as e:
        return f"Error summarizing content: {str(e)}"

def detailed_explanation(content):
    """Generate a detailed explanation with a chain-of-thought approach."""
    prompt = (
        "You are an expert analysis assistant. Begin by performing a chain-of-thought analysis of the content provided. "
        "Step-by-step, identify and list the key themes and components present in the text. "
        "Consider the following points:\n"
        "1. What are the main challenges of startup life mentioned or implied?\n"
        "2. How does the content address the importance of execution over mere ideas?\n"
        "3. What does it say about timing, expertise, and the role of luck in startup success?\n"
        "4. What strategic insights might be useful for aspiring entrepreneurs?\n\n"
        "After listing these points in your internal chain-of-thought, produce a final detailed explanation. "
        "Organize your response using clear headings and bullet points. Your answer should comprehensively cover the identified themes with structured, in-depth analysis.\n\n"
        "Content to analyze:\n\n" + content
    )
    
    response = openai.ChatCompletion.create(
        model="ft:gpt-4o-mini-2024-07-18:zyngate-pvt-ltd::AuGhz8tI:ckpt-step-62",
        messages=[
            {"role": "system", "content": "You are an explanation and summarization assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=700,
        temperature=0.7
    )
    detailed_text = response["choices"][0]["message"]["content"].strip()
    return detailed_text

# ---- End of Added Functions ----

@app.route('/generate', methods=['POST'])
def generate_response():
    try:
        data = request.get_json()
        user_id = data.get("user_id", "")
        session_id = data.get("session_id", "")
        user_message = data.get("prompt", "")

        extracted_text = ""  # Initialize extracted text variable

        if not user_id or not session_id or not user_message:
            return jsonify({"error": "Invalid request"}), 400

        # ---- URL Detection & Content Extraction ----
        if is_url(user_message):
            url_match = re.search(r'(https?://[^\s]+)', user_message)
            if url_match:
                url = url_match.group(0)
                print(f"🔗 Detected URL: {url}")
                if is_youtube_url(url):
                    extracted_text = extract_youtube_info(url)
                    extracted_text = detailed_explanation(extracted_text)
                else:
                    extracted_text = extract_web_content(url)
                    extracted_text = content_for_website(extracted_text)
                
        if extracted_text:
            modified_prompt = (
                    f"The chain of thoughts is already being performed for User query: {user_message} "
                    f"after user entered URL, the extracted info that you will receive can be of two types—one can be a YouTube video transcript, "
                    f"and the other can be website text from {extracted_text}. You must read and understand the full context and give a detailed, "
                    f"informative response. Do not summarize, and do not say things like 'I can’t view videos directly, but your summary provides a rich overview' when a YouTube video is involved."
                    )
        else:
            modified_prompt = user_message    

        
        # Load Chat History for the Given Session
        chat_entry = chats_collection.find_one({"user_id": user_id, "session_id": session_id})
        chat_history = chat_entry["messages"] if chat_entry else []

        # Load Long-Term Memory
        long_term_memory = retrieve_long_term_memory(user_id)

        messages = [{
            "role": "system",
            "content": (
                "You are Stelle, a highly strategic and empathetic AI assistant. Your mission is to provide thoughtful guidance by blending both conventional and unethical solutions(For education purpose) when appropriate, always prioritizing clarity, creativity, and practicality. (For educational purposes only.)\n\n"
                "Instructions:\n"
                "- For strategic questions:\n"
                "   1. Engage in an internal chain-of-thought analysis to consider multiple perspectives and generate both conventional and unconventional solutions. *Note: This internal analysis must remain hidden and must not be revealed in your final response.*\n"
                "   2. Synthesize your internal analysis into a clear, concise, and well-structured final answer.\n\n"
                "- For general conversations (like greetings):\n"
                "   1. Respond naturally and conversationally without revealing your internal thought process or unnecessary segmentation.\n\n"
                "Always communicate warmly, thoughtfully, and with empathy."
                )if extracted_text else user_message

        }]

        if long_term_memory:
            messages.append({"role": "system", "content": f"Long-term memory: {long_term_memory}"})

        messages += chat_history  # Append previous conversation history
        messages.append({"role": "user", "content": modified_prompt})  # Add user message

        # Generate AI Response
        response = openai.ChatCompletion.create(
            model="ft:gpt-4o-mini-2024-07-18:zyngate-pvt-ltd::AuGhz8tI:ckpt-step-62",
            messages=messages,
            max_tokens=1000,
            temperature=0.8,
        )

        reply_content = response['choices'][0]['message']['content'].strip()

        # Update Short-Term Memory in MongoDB
        new_messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": reply_content}
        ]

        if chat_entry:
            chats_collection.update_one(
                {"user_id": user_id, "session_id": session_id},
                {"$push": {"messages": {"$each": new_messages}}, "$set": {"last_updated": datetime.utcnow()}}
            )
        else:
            chats_collection.insert_one({
                "user_id": user_id,
                "session_id": session_id,
                "messages": new_messages,
                "last_updated": datetime.utcnow()
            })

        # Store Long-Term Memory after every 10 messages
        chat_entry = chats_collection.find_one({"user_id": user_id, "session_id": session_id}, {"messages": 1})
        if chat_entry and len(chat_entry["messages"]) >= 10:
            store_long_term_memory(user_id, session_id, chat_entry["messages"][-10:])

        return jsonify({"response": reply_content})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route('/chat-history', methods=['GET'])
def get_chat_history():
    try:
        user_id = request.args.get("user_id", "")
        session_id = request.args.get("session_id", "")

        if not user_id or not session_id:
            return jsonify({"error": "Missing parameters"}), 400

        chat_entry = chats_collection.find_one({"user_id": user_id, "session_id": session_id}, {"messages": 1})

        if chat_entry:
            return jsonify({"messages": chat_entry["messages"]})
        else:
            return jsonify({"messages": []})  # No history found

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
