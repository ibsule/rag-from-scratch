from pathlib import Path
from chunker import chunk_text
from embedder import embed_chunks
from retriever import retrieve
from prompt import build_prompt
from generator import generate_answer

def load_corpus(data_dir: str = "data") -> str:
    """Concatenate all .txt files in the data directory into one string"""
    all_text = ""
    for filepath in Path(data_dir).glob("*.txt"):
        all_text += filepath.read_text(encoding="utf-8") + "\n\n"
    return all_text

def answer_question(query: str, chunks: list[str], embeddings) -> str:
    retrieved = retrieve(query, chunks, embeddings, k=3)
    retrieved_texts = [chunk for chunk, score in retrieved]
    prompt = build_prompt(query, retrieved_texts)
    return generate_answer(prompt)

if __name__ == "__main__":
    text = load_corpus()
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    print(f"Indexed {len(chunks)} chunks")

    # Test queries
    test_queries = [
        "Who invented the AC induction motor?",
        "What year did the Wright brothers applied to patent?",
        "which countries were the soldiers from?",
        "einstein's equation is about what?",
    ]

    for query in test_queries:
        print(f"\n=== Query: {query} ===")
        answer = answer_question(query, chunks, embeddings)
        print(f"A: {answer}\n")