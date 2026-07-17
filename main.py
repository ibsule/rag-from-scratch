from pathlib import Path
from chunker import chunk_text

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

    # Prrint 3 sample chunks
    for i in [0, 1, 2]:
        print(f"\n--- Chunk {i} ---")
        print(chunks[i])