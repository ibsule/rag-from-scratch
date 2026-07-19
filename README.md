# Retrieval-Augmented Generation (RAG) From Scratch

A RAG system built from first principles which implements chunking, embeddings, similarity search, and generation directly to show how RAG actually works under the hood.

## How it works

1. **Chunking**: source documents are split into overlapping ~500-character chunks.
2. **Embedding**: each chunk is embedded into a 384-dim vector using `all-MiniLM-L6-v2`.
3. **Retrieval**: a query is embedded the same way, then compared against every chunk
   using cosine similarity to find the top-(3) most relevant chunks
4. **Generation**: the retrieved chunks are injected into a prompt and sent to Openai to generate a grounded answer

## Setup

```bash
uv sync
```

Create a `.env` file with your API key:
```
OPENAI_API_KEY=
```

Add your source `.txt` files into `data/`.

## Usage

```bash
uv run main.py
```

## Example

```
q> who invented airplanes?

a< The Wright brothers, Wilbur and Orville Wright, invented the airplane.
```

## Roadmap
- [x] Native brute force retrieval
- [] FAISS/Chroma vector store
- [] Smarter chunking 
- [] Evaluation harness measuring retrieval and answer quality.