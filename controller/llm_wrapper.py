import os
import openai
from openai import Stream, ChatCompletion
import base64
from typing import Union, Optional

GPT3 = "gpt-3.5-turbo-16k"
GPT4 = "gpt-4"
LLAMA3 = "meta-llama/Meta-Llama-3-8B-Instruct"
TINY_LLAMA = "robot-pilot-v36"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
chat_log_path = os.path.join(CURRENT_DIR, "assets/chat_log.txt")


class LLMWrapper:
    def __init__(self, temperature=0.0):
        self.temperature = temperature
        self.llama_client = openai.OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="token-abc123",
        )
        self.gpt_client = openai.OpenAI(
            base_url="http://localhost:11434/v1",  # Force to local
            api_key="token-abc123",
        )

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 string"""
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                return base64_image
        except Exception as e:
            print(f"[ERROR] Failed to encode image {image_path}: {e}")
            raise e

    def _create_vision_messages(self, prompt: str, image_path: str) -> list:
        """
        Create OpenAI vision API compatible messages with image
        Supports both OpenAI format and Ollama vision models
        """
        try:
            # Encode image to base64
            base64_image = self._encode_image_to_base64(image_path)

            # OpenAI/Ollama vision format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            return messages

        except Exception as e:
            print(f"[ERROR] Failed to create vision messages: {e}")
            # Fallback to text-only if image processing fails
            return [{"role": "user", "content": prompt}]

    def request(self, prompt: str, model_name: str = TINY_LLAMA, stream: bool = False,
                image_path: Optional[str] = None) -> Union[str, Stream[ChatCompletion.ChatCompletionChunk]]:
        """
        Make a request to the LLM, with optional image support for VLM

        Args:
            prompt: Text prompt
            model_name: Model to use
            stream: Whether to stream response
            image_path: Optional path to image file for VLM requests

        Returns:
            String response or stream object
        """
        print(f"[DEBUG] Using model: {model_name}")
        print(f"[DEBUG] Prompt: {prompt[:100]}...")
        print(f"[DEBUG] Image provided: {image_path is not None}")
        if image_path:
            print(f"[DEBUG] Image path: {image_path}")

        client = self.gpt_client  # Always use local now

        try:
            # Determine message format based on whether image is provided
            if image_path is not None:
                # VLM request with image
                messages = self._create_vision_messages(prompt, image_path)
                print(f"[DEBUG] Using VLM format with image")
            else:
                # Regular text-only request
                messages = [{"role": "user", "content": prompt}]
                print(f"[DEBUG] Using text-only format")

            # Make the API call
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=self.temperature,
                stream=stream,
            )

            # Log the interaction
            with open(chat_log_path, "a", encoding='utf-8') as f:
                f.write(f"MODEL: {model_name}\n")
                f.write(f"IMAGE: {image_path if image_path else 'None'}\n")
                f.write(prompt + "\n---\n")
                if not stream:
                    f.write(response.model_dump_json(indent=2) + "\n---\n")

            if stream:
                return response

            return response.choices[0].message.content

        except Exception as e:
            print(f"[ERROR] LLM request failed: {e}")
            print(f"[ERROR] Model: {model_name}, Image: {image_path is not None}")

            # If this was a VLM request that failed, try fallback to text-only
            if image_path is not None:
                print(f"[DEBUG] VLM request failed, trying text-only fallback")
                try:
                    fallback_response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        stream=stream,
                    )

                    if stream:
                        return fallback_response
                    return fallback_response.choices[0].message.content

                except Exception as fallback_error:
                    print(f"[ERROR] Text-only fallback also failed: {fallback_error}")
                    raise fallback_error
            else:
                # Re-raise original error if it wasn't a VLM request
                raise e

    def test_vlm_support(self, model_name: str) -> bool:
        """
        Test if a model supports vision/images

        Args:
            model_name: Model to test

        Returns:
            True if model supports VLM, False otherwise
        """
        try:
            # Create a simple test image (1x1 pixel)
            import tempfile
            from PIL import Image

            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                # Create a tiny test image
                test_img = Image.new('RGB', (1, 1), color='red')
                test_img.save(tmp.name)

                # Try a simple VLM request
                response = self.request(
                    prompt="What do you see in this image?",
                    model_name=model_name,
                    image_path=tmp.name
                )

                # Clean up
                os.unlink(tmp.name)

                # If we got any response, assume VLM works
                return response is not None and len(response.strip()) > 0

        except Exception as e:
            print(f"[DEBUG] VLM test failed for {model_name}: {e}")
            return False