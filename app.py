from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import google.generativeai as genai
import anthropic
import os

app = Flask(__name__)
CORS(app)

# LLM client classes
class OpenAIChatGPT:
    def __init__(self, api_key):
        openai.api_key = api_key

    def chat(self, messages):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.5,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response.choices[0].message['content']
        except Exception as e:
            return f"Error: {e}"

class GoogleGemini:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.5,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain"
            }
        )
        self.chat_session = self.model.start_chat(history=[])

    def chat(self, message):
        try:
            response = self.chat_session.send_message(message)
            return response.text
        except Exception as e:
            return f"Error: {e}"

class Claude:
    def __init__(self, api_key):
        self.client = anthropic.Client(api_key=api_key)

    def chat(self, message):
        try:
            response = self.client.completions.create(
                model="claude-2.1",
                max_tokens_to_sample=1000,
                temperature=0.5,
                prompt=f"\n\nHuman: {message}\n\nAssistant:"
            )
            return response.completion
        except Exception as e:
            return f"Error: {e}"

# Initialize LLM clients with your actual API keys from environment variables
openai_client = OpenAIChatGPT(api_key=os.getenv('OPENAI_API_KEY'))
gemini_client = GoogleGemini(api_key=os.getenv('GEMINI_API_KEY'))
claude_client = Claude(api_key=os.getenv('CLAUDE_API_KEY'))

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    llm_choice = data.get('llm_choice')
    user_input = data.get('user_input')

    if not llm_choice or not user_input:
        return jsonify({"error": "Invalid input"}), 400

    if llm_choice == 'openai':
        messages = [{"role": "user", "content": user_input}]
        response = openai_client.chat(messages)
    elif llm_choice == 'gemini':
        response = gemini_client.chat(user_input)
    elif llm_choice == 'claude':
        response = claude_client.chat(user_input)
    else:
        return jsonify({"error": "Invalid LLM choice"}), 400

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
