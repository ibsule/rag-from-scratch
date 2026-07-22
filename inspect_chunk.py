from pathlib import Path
from chunker import chunk_text

def load_corpus(data_dir: str = "data") -> str:
    all_text = ""
    for filepath in Path(data_dir).glob("*.txt"):
        all_text += filepath.read_text(encoding="utf-8") + "\n\n"
    return all_text

text = load_corpus()
chunks = chunk_text(text, chunk_size=500, overlap=50)

# Print chunks that starrt or end mid-sentence (primitive heuristic => doesn't start 
# with a capital letter, or doesn't end with ./!/?)
for i, chunk in enumerate(chunks):
    stripped = chunk.strip()
    if not stripped:
        continue
    starts_awkward = not stripped[0].isupper()
    ends_awkward = stripped[-1] not in ".!?\"'"
    
    if starts_awkward or ends_awkward:
        print(f"--- Chunk {i} (start_awkward={starts_awkward}, end_awkward={ends_awkward}) ---")
        print(stripped[:150], "...", stripped[-100:])
        print()
