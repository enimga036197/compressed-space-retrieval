# Compressed Space Retrieval

Exploring whether a small neural network can learn to predict which blocks of compressed data to fetch for a given query, without the model having any knowledge of compression or SQL.

Early prototype — testing feasibility, not making claims.

## The Idea

A small NN receives a query and outputs block address predictions. An environment actions those predictions against a compressed data store, handling decompression invisibly. The model only ever sees query in, results back — identical in training and inference.

```
Query -> NN -> [block addresses] -> Environment fetches & decompresses -> Results
                                    (invisible to model)
```

## Status

One synthetic test so far. 2000 rows, 80 compressed blocks, 10 categories, compound queries.

| Scenario | Recall | Blocks Scanned | Notes |
|----------|--------|---------------|-------|
| Clustered (sorted) data | 100% | 7.2% | Model learns clean block boundaries |
| Random (unsorted) data | 90.6% | 59.6% | Scattered data = scattered access |

The clustered result is encouraging but expected — sorted data naturally groups into blocks. The random result shows the model struggles when there's no spatial locality to learn. Neither result is surprising on its own; the question is whether this scales or generalizes to anything useful.

## Run

```
pip install torch
python prototype.py
```

## How It Works

1. A SQLite database provides ground truth query results
2. The same data is serialized and compressed into independent zlib blocks
3. A small MLP is trained to predict which blocks contain matching rows
4. At inference, predicted blocks are fetched from the compressed store
5. Results are compared against SQL ground truth to check correctness

## Next

- Test with unseen query combinations (train/test split on query types, not just instances)
- Try real-world data (CSV/Parquet) instead of synthetic
- Iterative retrieval — model requests blocks in rounds, refining based on what comes back
- Byte-level addressing instead of block-level
- Compare against a traditional index on the same data to see if the NN adds anything

## License

MIT
