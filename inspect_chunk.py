from pathlib import Path
from recursive_chunker import recursive_chunk

def load_corpus(data_dir: str = "data") -> str:
    all_text = ""
    for filepath in Path(data_dir).glob("*.txt"):
        all_text += filepath.read_text(encoding="utf-8") + "\n\n"
    return all_text

text = load_corpus()
chunks = recursive_chunk(text, chunk_size=500)
print(f"Created {len(chunks)} chunks")

for i in [0, 1, 2]:
    print(f"\n--- Chunk {i} ---")
    print(chunks[i])
