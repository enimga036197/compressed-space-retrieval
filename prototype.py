"""
Compressed Space Retrieval Agent — Prototype

Core idea: A tiny NN learns a retrieval policy over compressed data.
It has ZERO knowledge of compression or SQL. From its perspective,
it receives a query, emits "tool calls" (address requests), and gets
data back. The environment handles compression/decompression invisibly.

Training and inference are identical experiences for the model.

Scenarios:
  A) Clustered data (sorted by category) — model should learn selective access
  B) Random data — model must learn scattered access patterns
  C) Compound queries (category + value range) — harder retrieval
"""

import sqlite3
import zlib
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim


# ── Data Layer ──

CATEGORIES = [
    "electronics", "clothing", "food", "books", "tools",
    "sports", "music", "health", "auto", "garden",
]

REGIONS = ["north", "south", "east", "west"]


def create_database(n_rows=2000, clustered=True):
    conn = sqlite3.connect(":memory:")
    c = conn.cursor()
    c.execute("""CREATE TABLE items (
        id INT PRIMARY KEY, category TEXT, region TEXT, value INT
    )""")

    rows = []
    for i in range(n_rows):
        cat = CATEGORIES[i % len(CATEGORIES)] if clustered else random.choice(CATEGORIES)
        region = REGIONS[i % len(REGIONS)] if clustered else random.choice(REGIONS)
        val = random.randint(1, 100)
        rows.append((i, cat, region, val))

    if clustered:
        # Sort by category then region — groups similar data in nearby blocks
        rows = sorted(rows, key=lambda r: (r[1], r[2]))
        # Reassign IDs to match sorted order
        rows = [(i, r[1], r[2], r[3]) for i, r in enumerate(rows)]

    c.executemany("INSERT INTO items VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    return conn, rows


class CompressedStore:
    def __init__(self, rows, block_size=25):
        self.block_size = block_size
        self.blocks = []
        self.row_to_block = {}

        for start in range(0, len(rows), block_size):
            chunk = rows[start : start + block_size]
            raw = json.dumps(chunk).encode()
            compressed = zlib.compress(raw, level=9)
            for row in chunk:
                self.row_to_block[row[0]] = len(self.blocks)
            self.blocks.append(compressed)

        self.n_blocks = len(self.blocks)
        self.compressed_bytes = sum(len(b) for b in self.blocks)
        self.raw_bytes = sum(
            len(json.dumps(rows[s : s + block_size]).encode())
            for s in range(0, len(rows), block_size)
        )

    def fetch(self, block_indices):
        results = []
        for idx in block_indices:
            if 0 <= idx < self.n_blocks:
                raw = zlib.decompress(self.blocks[idx])
                results.extend(json.loads(raw))
        return results


# ── Query Types ──

class QueryEncoder:
    """Encodes different query types into fixed-size vectors."""

    def __init__(self):
        # category (10) + region (4) + value_low (1) + value_high (1) = 16
        self.dim = len(CATEGORIES) + len(REGIONS) + 2

    def encode(self, query):
        vec = [0.0] * self.dim
        cat_idx = CATEGORIES.index(query["category"])
        vec[cat_idx] = 1.0

        if query.get("region"):
            reg_idx = len(CATEGORIES) + REGIONS.index(query["region"])
            vec[reg_idx] = 1.0

        # Normalize value range to [0, 1]
        vec[-2] = query.get("value_low", 0) / 100.0
        vec[-1] = query.get("value_high", 100) / 100.0

        return vec


def generate_queries(n=100):
    """Generate diverse queries for training/testing."""
    queries = []
    for _ in range(n):
        q = {"category": random.choice(CATEGORIES)}
        # 40% include region filter
        if random.random() < 0.4:
            q["region"] = random.choice(REGIONS)
        # 30% include value range
        if random.random() < 0.3:
            lo = random.randint(1, 70)
            hi = random.randint(lo + 10, 100)
            q["value_low"] = lo
            q["value_high"] = hi
        else:
            q["value_low"] = 0
            q["value_high"] = 100
        queries.append(q)
    return queries


# ── Environment ──

class Environment:
    def __init__(self, conn, store):
        self.conn = conn
        self.store = store
        self.encoder = QueryEncoder()

    def ground_truth(self, query):
        sql = "SELECT id, category, region, value FROM items WHERE category = ?"
        params = [query["category"]]
        if query.get("region"):
            sql += " AND region = ?"
            params.append(query["region"])
        if query.get("value_low", 0) > 0 or query.get("value_high", 100) < 100:
            sql += " AND value >= ? AND value <= ?"
            params.extend([query["value_low"], query["value_high"]])
        c = self.conn.cursor()
        c.execute(sql, params)
        return c.fetchall()

    def ground_truth_blocks(self, query):
        gt_rows = self.ground_truth(query)
        blocks = set()
        for row in gt_rows:
            blocks.add(self.store.row_to_block[row[0]])
        return sorted(blocks), gt_rows

    def action_tool_calls(self, block_indices, query):
        fetched = self.store.fetch(block_indices)

        # Filter fetched results to match query (environment handles this)
        matching = [r for r in fetched if r[1] == query["category"]]
        if query.get("region"):
            matching = [r for r in matching if r[2] == query["region"]]
        lo, hi = query.get("value_low", 0), query.get("value_high", 100)
        if lo > 0 or hi < 100:
            matching = [r for r in matching if lo <= r[3] <= hi]

        gt_rows = self.ground_truth(query)
        found_ids = {r[0] for r in matching}
        expected_ids = {r[0] for r in gt_rows}
        hits = len(found_ids & expected_ids)

        return {
            "results": matching,
            "recall": hits / len(expected_ids) if expected_ids else 1.0,
            "hits": hits,
            "expected": len(expected_ids),
            "blocks_used": len(block_indices),
        }


# ── The Agent ──

class RetrievalAgent(nn.Module):
    def __init__(self, query_dim, n_blocks, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(query_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_blocks),
        )

    def forward(self, query_vec):
        return torch.sigmoid(self.net(query_vec))


# ── Training ──

def train(model, env, train_queries, n_epochs=400, lr=0.005):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Training on {len(train_queries)} queries for {n_epochs} epochs")
    print("-" * 60)

    for epoch in range(n_epochs):
        total_loss = 0.0
        random.shuffle(train_queries)

        for query in train_queries:
            qvec = torch.tensor([env.encoder.encode(query)])
            gt_blocks, _ = env.ground_truth_blocks(query)

            target = torch.zeros(1, env.store.n_blocks)
            for b in gt_blocks:
                target[0, b] = 1.0

            pred = model(qvec)
            loss = nn.functional.binary_cross_entropy(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 100 == 0:
            avg = total_loss / len(train_queries)
            print(f"  Epoch {epoch+1:>4}/{n_epochs} | avg loss: {avg:.6f}")

    print()


# ── Evaluation ──

def evaluate(model, env, queries, label="Test", threshold=0.5):
    print(f"{label}: {len(queries)} queries -> tool calls into compressed space")
    print("-" * 60)

    model.eval()
    totals = {"correct": 0, "expected": 0, "blocks": 0, "perfect": 0, "queries": 0}

    with torch.no_grad():
        for i, query in enumerate(queries):
            qvec = torch.tensor([env.encoder.encode(query)])
            pred = model(qvec)
            selected = (pred[0] > threshold).nonzero(as_tuple=True)[0].tolist()

            result = env.action_tool_calls(selected, query)
            gt_blocks, _ = env.ground_truth_blocks(query)

            ok = result["recall"] == 1.0
            if ok:
                totals["perfect"] += 1

            # Print first 10 and any misses
            if i < 10 or not ok:
                desc = f"cat={query['category']}"
                if query.get("region"):
                    desc += f" reg={query['region']}"
                if query.get("value_low", 0) > 0 or query.get("value_high", 100) < 100:
                    desc += f" val={query['value_low']}-{query['value_high']}"

                status = "OK" if ok else "MISS"
                print(
                    f"  {desc:<35}"
                    f"  blk: {result['blocks_used']:>3}/{env.store.n_blocks}"
                    f"  (need {len(gt_blocks):>3})"
                    f"  rows: {result['hits']:>3}/{result['expected']:>3}"
                    f"  [{status}]"
                )

            totals["correct"] += result["hits"]
            totals["expected"] += result["expected"]
            totals["blocks"] += result["blocks_used"]
            totals["queries"] += 1

    print()
    recall = totals["correct"] / totals["expected"] * 100 if totals["expected"] else 100
    avg_blocks = totals["blocks"] / totals["queries"]
    scan_pct = avg_blocks / env.store.n_blocks * 100
    print("=" * 60)
    print(f"  Recall:        {recall:.1f}% ({totals['correct']}/{totals['expected']} rows)")
    print(f"  Perfect:       {totals['perfect']}/{totals['queries']} queries")
    print(f"  Avg blocks:    {avg_blocks:.1f}/{env.store.n_blocks} ({scan_pct:.1f}% scanned)")
    print(f"  Selectivity:   {100 - scan_pct:.1f}% of data NOT touched")
    print(f"  Compression:   {env.store.raw_bytes/env.store.compressed_bytes:.1f}x")
    print(f"  Model params:  {sum(p.numel() for p in model.parameters()):,}")
    print()
    return recall, scan_pct


def verify(model, env, queries, threshold=0.5):
    print("Verification: compressed results vs SQL ground truth")
    print("-" * 60)
    model.eval()
    all_match = True
    with torch.no_grad():
        for query in queries[:20]:  # spot check 20
            qvec = torch.tensor([env.encoder.encode(query)])
            pred = model(qvec)
            selected = (pred[0] > threshold).nonzero(as_tuple=True)[0].tolist()
            result = env.action_tool_calls(selected, query)

            gt_rows = env.ground_truth(query)
            gt_set = {tuple(r) for r in gt_rows}
            comp_set = {tuple(r) for r in result["results"]}

            if comp_set == gt_set:
                print(f"  EXACT MATCH ({len(gt_set)} rows)")
            elif comp_set < gt_set:
                print(f"  SUBSET — missing {len(gt_set - comp_set)} rows")
                all_match = False
            else:
                print(f"  MISMATCH — missing {len(gt_set - comp_set)}, extra {len(comp_set - gt_set)}")
                all_match = False

    print()
    if all_match:
        print("  ALL VERIFIED: compressed space returns identical data to SQL")
    else:
        print("  Some queries incomplete (model needs tuning)")
    print()


def run_scenario(name, n_rows, block_size, clustered, n_train, n_test):
    print()
    print("#" * 60)
    print(f"  SCENARIO: {name}")
    print("#" * 60)
    print()

    conn, rows = create_database(n_rows=n_rows, clustered=clustered)
    store = CompressedStore(rows, block_size=block_size)

    print(f"  Rows:       {len(rows)}")
    print(f"  Blocks:     {store.n_blocks} (x {store.block_size} rows each)")
    print(f"  Raw:        {store.raw_bytes:,} bytes")
    print(f"  Compressed: {store.compressed_bytes:,} bytes")
    print(f"  Ratio:      {store.raw_bytes/store.compressed_bytes:.1f}x")
    print(f"  Clustered:  {clustered}")
    print()

    env = Environment(conn, store)
    model = RetrievalAgent(
        query_dim=env.encoder.dim,
        n_blocks=store.n_blocks,
        hidden=128,
    )

    train_queries = generate_queries(n_train)
    test_queries = generate_queries(n_test)

    train(model, env, train_queries, n_epochs=400)
    recall, scan = evaluate(model, env, test_queries, label="Inference")
    if recall == 100.0:
        verify(model, env, test_queries)

    conn.close()
    return recall, scan


def main():
    random.seed(42)
    torch.manual_seed(42)

    results = []

    # Scenario A: Clustered data, small blocks — best case for selectivity
    r, s = run_scenario(
        "Clustered data (sorted by category+region)",
        n_rows=2000, block_size=25, clustered=True,
        n_train=200, n_test=50,
    )
    results.append(("Clustered", r, s))

    # Scenario B: Random data — harder, model must learn scattered patterns
    random.seed(42)
    torch.manual_seed(42)
    r, s = run_scenario(
        "Random data (categories scattered across blocks)",
        n_rows=2000, block_size=25, clustered=False,
        n_train=200, n_test=50,
    )
    results.append(("Random", r, s))

    # Summary
    print()
    print("#" * 60)
    print("  SUMMARY")
    print("#" * 60)
    print()
    print(f"  {'Scenario':<20} {'Recall':>8} {'Scanned':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*10}")
    for name, recall, scan in results:
        print(f"  {name:<20} {recall:>7.1f}% {scan:>9.1f}%")
    print()
    print("  Key metric: high recall + low scan% = model is selective")
    print("  (it fetches only the blocks it needs, not everything)")
    print()


if __name__ == "__main__":
    main()
