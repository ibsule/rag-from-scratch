from pathlib import Path
from chunker import chunk_text
from embedder import embed_chunks

def load_corpus(data_dir: str = "data") -> str:
    """Concatenate all .txt files in the data directory into one string"""
    all_text = ""
    for filepath in Path(data_dir).glob("*.txt"):
        all_text += filepath.read_text(encoding="utf-8") + "\n\n"
    return all_text

if __name__ == "__main__":
    text = load_corpus()
    print(f"Loaded {len(text)} characters")

    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")

    embeddings = embed_chunks(chunks)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings[0] = {embeddings[0]}")