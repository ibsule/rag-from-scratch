def build_prompt(query: str, retrieved_chunks: list[str]) -> str:
    context = "\n\n--\n\n".join(retrieved_chunks)
    return f"""Answer the question using ONLY the context below. If the answer isn't in the context, say "I don't have enough informaton to answer that."

Context:
{context}

Question: {query}

Answer:"""

