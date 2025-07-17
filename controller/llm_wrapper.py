import os
import openai
from openai import Stream, ChatCompletion

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

    def request(self, prompt, model_name=TINY_LLAMA, stream=False) -> str | Stream[ChatCompletion.ChatCompletionChunk]:
        print(f"[DEBUG] Using model: {model_name}")
        print(f"[DEBUG] Prompt: {prompt[:100]}...")
        client = self.gpt_client  # Always use local now

        # ========== FIX: Actually use the model_name parameter ==========
        response = client.chat.completions.create(
            model=model_name,  # ✅ FIXED: Use the actual model_name passed in
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stream=stream,
        )

        # save the message in a txt
        with open(chat_log_path, "a") as f:
            f.write(f"MODEL: {model_name}\n")  # Also log which model was used
            f.write(prompt + "\n---\n")
            if not stream:
                f.write(response.model_dump_json(indent=2) + "\n---\n")

        if stream:
            return response

        return response.choices[0].message.content