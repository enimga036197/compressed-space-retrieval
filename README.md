# Compressed Space Retrieval

A tiny neural network learns to retrieve data directly from compressed storage — without knowing anything about compression or SQL.

## The Idea

Instead of decompress → query → result, a small NN learns which regions of compressed data to fetch for a given query. The model emits "tool calls" (block addresses) into the compressed space. An environment handles decompression invisibly. Training and inference are identical from the model's perspective.

```
Query → NN → [block addresses] → Environment fetches & decompresses → Results
                                  (invisible to model)
```

The model learns a retrieval policy, not compression or SQL. It's a learned index over compressed storage.

## Results

| Scenario | Recall | Data Scanned | Model Size |
|----------|--------|-------------|------------|
| Clustered data | 100% | 7.2% | 29k params |
| Random data | 90.6% | 59.6% | 29k params |

Clustered data (sorted by category) gives the model clean block boundaries to learn. All results verified bit-perfect against SQL ground truth.

## Run

```
pip install torch
python prototype.py
```

## How It Works

1. A SQLite database is created as ground truth
2. The same data is serialized and compressed into independent blocks (zlib)
3. A small MLP learns which blocks to request for each query type
4. At inference, the model's block predictions are actioned against the compressed store
5. Results are compared against SQL to verify correctness

The key property: compression is deterministic, so the mapping from query semantics to block addresses is stable and learnable.

## License

MIT
