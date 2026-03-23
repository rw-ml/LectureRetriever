import re
from response_generation.citation_builder import build_sources

def clean_llm_output(output: str) -> str:
    """
    Remove <think> sections from LLM output.
    - If <think>...</think> exists → remove entire block.
    - If <think> exists but no closing tag → remove everything after <think>.
    """
    # Case 1: <think>...</think> exists
    output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL)

    # Case 2: <think> exists but no </think>
    if "<think>" in output:
        output = output.split("<think>")[0]

    return output.strip()

def build_context(results, max_chars=1.5e5):
    """
    Combine retrieved chunks into a prompt context.
    """

    text = "\n"
    total_chars = 0

    for r in results:
        chunk_text = r["text"]
        pages = r["pages"]

        part = f"[Pages {pages}]\n{chunk_text}\n"

        if total_chars + len(part) > max_chars:
            break

        text += chunk_text + "\n"
        total_chars += len(part)

    return text


def get_system_prompt():
    return """
You are a teaching assistant answering questions about a lecture.

Use only the provided lecture context to answer the question. 
Base your answer strictly on the information in the context and do not rely on external knowledge.

Synthesize information across the context when needed.

Give a structured explanation with clear sections and, if helpful, bullet points.

Be concise but complete. Avoid unnecessary repetition.

If the answer cannot be derived from the provided context, explicitly say: "I do not know based on the provided context."
"""

def build_user_prompt(context, question):
    return f"""
Context:
{context}

Question:
{question}

Answer:
"""

def build_prompt(context, question):
    prompt = get_system_prompt() + build_user_prompt(context, question)
    return prompt

class RAGPipeline:
    def __init__(self, retriever, vllm_client):
        self.retriever = retriever
        self.vllm_client = vllm_client

    def ask_stream(self, question, lecture_name):
        '''
            RAG (streamed) response generation with vLLM backend
        '''
        # stage 1: retrieval
        candidates = self.retriever.retrieve(
            query=question,
            lecture_name=lecture_name,
            top_k=30
        )

        # stage 2: reranking
        results = self.retriever.rerank(
            query=question,
            candidates=candidates,
            top_k=5
        )

        context = build_context(results)
        messages = [
            {
                "role": "system",
                "content": get_system_prompt()
            },
            {
                "role": "user",
                "content": build_user_prompt(context, question)
            }
        ]

        # stream tokens
        def stream():
            full_answer = ""
            for token in self.vllm_client.stream_request(messages):
                full_answer += token
                yield token

            # after generation append sources
            sources = build_sources(results, lecture_name)
            yield "\n\n" + sources
        return stream()