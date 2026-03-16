# generator.py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

class Generator:
    def __init__(self, model_name: str, quantize_4bit: bool = True, device: str = "cuda"):
        self.model_name = model_name

        # 4-bit quantization config
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        ) if quantize_4bit else None

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=self.bnb_config,
            device_map="auto" if device == "cuda" else None
        )

        # HF text-generation pipeline
        self.generator = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1
        )

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7):
        outputs = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.1
        )
        return outputs[0]["generated_text"]


# helper to match your old get_generator call
def get_generator(model_name="Qwen/Qwen3.5-4B", quantize_4bit=True, device="cuda"):
    return Generator(model_name=model_name, quantize_4bit=quantize_4bit, device=device)


def get_generator_old(model_name: str = "Qwen/Qwen3.5-4B"):
    '''

    :param model_name: llm for generation
        (- "Qwen/Qwen3.5-9B" --> likely too large)
        - "Qwen/Qwen3.5-4B" --> high quality for small models -> 3-4GB RAM
        - "Qwen/Qwen3.5-2B" --> smaller, balanced
        - "Qwen/Qwen3.5-0.8B" --> fast
    :return:
    '''
    generator = pipeline(
        "text-generation",
        model=model_name,
        max_new_tokens=1000
    )
    return generator