# Chunking Strategy Comparison

## Setup
- Naive chunker: fixed 500-char windows, 50-char overlap to N chunks
- Recursive chunker: paragraph -> sentence -> char fallback. 500 char target to M chunks

## Findings

### Query: "who invented the airplane?"
- Naive: Response was cut off mid-sentence was close to a full answer
- Recursive: Response was correct with full answer spread across 1-2 chunks

### Query: "what year was the airplane invented?"
- Naive: Response included other important dates but didn't include when the airplane was invented
- Recursive: Response included the year the airplane was invented in a full sentence

### Query: "when did nikola tesla migrated to the united states?"
- Naive: Response did not include any information related to when Tesla migrated to the United States
- Recursive: Response included when Tesla became a citizen of the United States but not when he migrated

# Takeaway
The most notable takeaway from this comparism test is that recursive chunking did perform better and returned more closely related response compared to naive chunking. Although, in scenarios when the answer was spread across multiple chunks that are distant from each other, both recursive and naive performed poorly.
