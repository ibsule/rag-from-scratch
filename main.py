from pathlib import Path
from chunker import chunk_text
from embedder import embed_chunks
from retriever import retrieve

def load_corpus(data_dir: str = "data") -> str:
    """Concatenate all .txt files in the data directory into one string"""
    all_text = ""
    for filepath in Path(data_dir).glob("*.txt"):
        all_text += filepath.read_text(encoding="utf-8") + "\n\n"
    return all_text

if __name__ == "__main__":
    text = load_corpus()
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    print(f"Indexed {len(chunks)} chunks")

    # Test queries
    test_queries = [
        "which countries were the soldiers from",
        "what does the dog that runs finds",
        "einstein's equation is about what",
        "the spanish colonel",
        "what is a noun"
    ]

    for query in test_queries:
        print(f"\n=== Query: {query} ===")
        results = retrieve(query, chunks, embeddings, k=3)
        for chunk, score in results:
            print(f"\n[score: {score:.3f}]")
            print(chunk[:200]) # print first 200 characters to output