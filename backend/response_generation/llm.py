import requests
import json


class VLLMClient:
    def __init__(
        self,
        base_url: str,
        model_name="Qwen/Qwen3.5-2B",
        max_tokens=512,
        temperature=0.0,
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    def stream_request(self, messages):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        with requests.post(self.base_url, json=payload, stream=True) as response:
            for line in response.iter_lines():
                if not line:
                    continue

                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    data = decoded[len("data: "):]
                    if data == "[DONE]":
                        break

                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta:
                        yield delta["content"]