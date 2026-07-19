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

def main():
    print("Loading and indexing corpus...")
    text = load_corpus()
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    embeddings = embed_chunks(chunks)
    print(f"Ready to go...Indexed {len(chunks)} chunks from the provided corpus")

    print("Ask as question (or type 'quit' to exit):\n")
    while True:
        query = input("q> ").strip()
        if query.lower() in ('quit', 'exit', 'q'):
            break
        if not query:
            continue

        answer = answer_question(query, chunks, embeddings)
        print(f"\na< {answer}\n")

if __name__ == "__main__":
    main()