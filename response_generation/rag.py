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

def build_context(results, max_chars=5e2):
    """
    Combine retrieved chunks into a prompt context.
    """

    context_parts = []
    total_chars = 0

    for r in results:
        chunk_text = r["text"]
        pages = r["pages"]

        part = f"[Pages {pages}]\n{chunk_text}\n"

        if total_chars + len(part) > max_chars:
            break

        context_parts.append(part)
        total_chars += len(part)

    return "\n".join(context_parts)


def build_prompt(context, question):
    prompt = f"""
You are a teaching assistant answering questions about a lecture.

Only use the provided lecture context.
If the answer is not contained in the context, say you do not know.

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt



class RAGPipeline:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def ask(self, question, lecture_name):
        # stage 1: semantic retrieval
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

        # build context
        context = build_context(results)

        # build prompt
        prompt = build_prompt(context, question)

        # generate answer
        output = self.generator(
            prompt,
            do_sample=False
        )
        answer = output[0]["generated_text"]
        answer = clean_llm_output(answer)

        sources = build_sources(results, lecture_name)
        return answer + "\n\n" + sources