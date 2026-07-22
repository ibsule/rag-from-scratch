import re

def split_into_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]

def split_into_sentences(text: str) -> list[str]:
    # Split the sentences naively. splits on ./!/? followed by a space and capital letter.
    # Might struggle with abbreviations like "Mr." or "U.S."
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]

def recursive_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    current_chunk = ""

    paragraphs = split_into_paragraphs(text)

    for paragraph in paragraphs:
        # if the whole paragraph fits, treat as one unit.
        candidate = (current_chunk + "\n\n" + paragraph).strip() if current_chunk else paragraph

        if len(candidate) <= chunk_size:
            current_chunk = candidate
            continue

        # paragraph doesn't fit. Ignore what we have, then handle the paragraph itself
        if current_chunk:
            chunks.append(current_chunk)
            current_chunk = ""

        if len(paragraph) <= chunk_size:
            current_chunk = paragraph
            continue

        # Paragraph itself is too big. Fall back to sentence level splitting
        sentences = split_into_sentences(paragraph)
        sentence_chunk = ""
        for sentence in sentences:
            candidate = (sentence_chunk + " " + sentence).strip() if sentence_chunk else sentence
            if len(candidate) <= chunk_size:
                sentence_chunk = candidate
            else:
                if sentence_chunk:
                    chunks.append(sentence_chunk)
                # Last resort to cut sentence if it's too big
                if len(sentence) <= chunk_size:
                    sentence_chunk = sentence
                else:
                    chunks.append(sentence[:chunk_size])
                    sentence_chunk = sentence_chunk[chunk_size:]
        if sentence_chunk:
            current_chunk = sentence_chunk

    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

