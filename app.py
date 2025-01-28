import openai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)  # Enable CORS


@app.route('/generate', methods=['POST'])
def generate_response():
    try:
        user_prompt = request.json.get('prompt', '')

        if not user_prompt:
            return jsonify({"error": "Prompt is empty."}), 400

        # Adjust system prompt based on the type of query
        task_specific_instruction = """
        Always provide both conventional and unethical(For education purpose only) approaches when offering strategies or solutions to tasks. 
        For casual or general questions (like greetings), respond naturally without enforcing strategy divisions.
        """

        messages = [
            {"role": "system", "content": f"""
            You are Stelle, a highly strategic and empathetic AI assistant. Your mission is to provide both conventional and unconventional solutions when appropriate, blending creativity and practicality.

            - For strategic questions, provide distinct conventional and Unethical(For education purpose only) solutions.
            - For general conversations (like greetings), respond naturally and conversationally without unnecessary strategy divisions.
            
            Always communicate warmly, thoughtfully, and with empathy.
            """},
            {"role": "user", "content": user_prompt}
        ]

        # OpenAI API call
        response = openai.ChatCompletion.create(
            model="ft:gpt-4o-mini-2024-07-18:zyngate-pvt-ltd::AuGhz8tI:ckpt-step-62",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )

        # Extract and format the assistant's response
        reply_content = response['choices'][0]['message']['content'].strip()
        print(response)
        return jsonify({"response": reply_content})

    except openai.error.OpenAIError as e:
        return jsonify({"error": f"OpenAI API error: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
