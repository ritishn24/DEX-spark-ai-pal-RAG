import os
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class LLMClient:
    def __init__(self):
        self.llm_server_url = os.getenv('LLM_SERVER_URL', 'http://localhost:8000/v1/chat/completions')
        self.llm_model_path = os.getenv('LLM_MODEL_PATH', '/root/.cache/huggingface/')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        self.openai_client = OpenAI(api_key=self.openai_api_key) if self.openai_api_key else None

    def generate_response(self, user_message, context="", system_context="", max_tokens=1000, temperature=0.1):
        """Try local LLM first, then fallback to OpenAI if needed"""
        try:
            local_response = self._call_local_llm(user_message, context, system_context, max_tokens, temperature)
            if local_response:
                return local_response
        except Exception as e:
            print(f"[Local LLM Error] {e}")

        try:
            if self.openai_client:
                openai_response = self._call_openai_llm(user_message, context, system_context, max_tokens, temperature)
                if openai_response:
                    return openai_response
        except Exception as e:
            print(f"[OpenAI Fallback Error] {e}")

        return "I'm sorry, I'm currently unable to process your request. Please try again later."

    def _call_local_llm(self, user_message, context="", system_context="", max_tokens=1000, temperature=0.1):
        """Call local LLM server endpoint"""
        message = self._format_message_with_context(user_message, context)
        
        system_prompt = system_context or "You are a helpful AI assistant. You can answer questions, help with tasks, and provide information based on the context provided."

        payload = {
            "model": self.llm_model_path,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        headers = { "Content-Type": "application/json" }

        response = requests.post(
            self.llm_server_url,
            json=payload,
            headers=headers,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("choices"):
                return data["choices"][0]["message"]["content"].strip()
        else:
            print(f"[Local LLM] HTTP {response.status_code} - {response.text}")

        return None

    def _call_openai_llm(self, user_message, context="", system_context="", max_tokens=1000, temperature=0.1):
        """Call OpenAI GPT-3.5 Turbo using new SDK"""
        if not self.openai_client:
            return None

        message = self._format_message_with_context(user_message, context)
        system_prompt = system_context or "You are a helpful AI assistant. You can answer questions, help with tasks, and provide information based on the context provided."

        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )

        return response.choices[0].message.content.strip()

    def _format_message_with_context(self, user_message, context=""):
        """Embed context into the user message if provided"""
        if context.strip():
            return f"""Context:\n{context}\n\nUser Question: {user_message}"""
        return user_message

    def test_connection(self):
        """Test availability of local LLM and OpenAI fallback"""
        status = {
            "local_llm": False,
            "openai_api": False
        }

        try:
            if self._call_local_llm("Hello") is not None:
                status["local_llm"] = True
        except Exception as e:
            print(f"[Local LLM Test Failed] {e}")

        try:
            if self.openai_client and self._call_openai_llm("Hello") is not None:
                status["openai_api"] = True
        except Exception as e:
            print(f"[OpenAI API Test Failed] {e}")

        return status
    