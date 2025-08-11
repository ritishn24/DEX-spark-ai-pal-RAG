import os
import base64
import requests
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class ImageProcessor:
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        self.llm_server_url = os.getenv('LLM_SERVER_URL', 'http://localhost:8000/v1/chat/completions')
        self.llm_model_path = os.getenv('LLM_MODEL_PATH', '/root/.cache/huggingface/')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_client = OpenAI(api_key=self.openai_api_key) if self.openai_api_key else None

    def is_supported_format(self, filename):
        return any(filename.lower().endswith(fmt) for fmt in self.supported_formats)

    def image_to_base64(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error converting image to base64: {str(e)}")
            return None

    def resize_image(self, image_path, max_size=(800, 600)):
        try:
            with Image.open(image_path) as img:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                img.save(image_path, optimize=True, quality=85)
                return True
        except Exception as e:
            print(f"Error resizing image: {str(e)}")
            return False

    def get_image_info(self, image_path):
        try:
            with Image.open(image_path) as img:
                return {
                    'width': img.width,
                    'height': img.height,
                    'format': img.format,
                    'mode': img.mode,
                    'size_kb': os.path.getsize(image_path) // 1024
                }
        except Exception as e:
            print(f"Error getting image info: {str(e)}")
            return None

    def describe_image(self, image_path):
        try:
            info = self.get_image_info(image_path)
            if not info:
                return "Unable to analyze image."

            description = f"This is a {info['format']} image with dimensions {info['width']}x{info['height']} pixels. "

            if info['size_kb'] > 1000:
                description += f"The image size is {info['size_kb']} KB. "

            if info['mode'] == 'RGB':
                description += "It's a full-color image. "
            elif info['mode'] == 'L':
                description += "It's a grayscale image. "
            elif info['mode'] == 'RGBA':
                description += "It's a color image with transparency. "

            aspect_ratio = info['width'] / info['height']
            if aspect_ratio > 1.5:
                description += "The image has a landscape orientation. "
            elif aspect_ratio < 0.67:
                description += "The image has a portrait orientation. "
            else:
                description += "The image has a square or near-square aspect ratio. "

            description += "For detailed content analysis, advanced vision AI would be needed."
            return description

        except Exception as e:
            print(f"Error describing image: {str(e)}")
            return "Unable to analyze the uploaded image."

    def analyze_with_ai(self, image_path):
        """Try local LLM for image analysis; fall back to OpenAI GPT-4o vision or basic metadata description"""
        try:
            base64_image = self.image_to_base64(image_path)
            if not base64_image:
                return "Could not convert image to base64."

            ext = os.path.splitext(image_path)[1].lower()
            mime_type = "image/png" if ext == ".png" else "image/jpeg"
            image_url_data = f"data:{mime_type};base64,{base64_image}"

            # ========== Try Local LLM FIRST ==========
            try:
                payload = {
                    "model": self.llm_model_path,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this image in detail."},
                                {"type": "image_url", "image_url": {"url": image_url_data}}
                            ]
                        }
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.4
                }
                headers = {"Content-Type": "application/json"}
                response = requests.post(self.llm_server_url, json=payload, headers=headers, timeout=20)
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content'].strip()
                else:
                    print(f"[Local LLM] Failed with status code {response.status_code}")
            except Exception as e:
                print(f"[Local LLM] Error: {str(e)}")

            # ========== Fallback: OpenAI GPT-4o ==========
            if self.openai_client:
                try:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Describe this image in detail."},
                                    {"type": "image_url", "image_url": {"url": image_url_data}}
                                ]
                            }
                        ],
                        max_tokens=1000
                    )
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    print(f"[OpenAI Vision] Error: {str(e)}")

            return "No vision model available. Showing basic metadata:\n" + self.describe_image(image_path)

        except Exception as e:
            print(f"[AI Vision Error] {str(e)}")
            return "Error analyzing image.\n" + self.describe_image(image_path)
