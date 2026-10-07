"""Step 2: run the experiment grid - one change at a time - and print a comparison table.

Each experiment re-indexes into its own database folder (chroma_db_<name>) and evaluates
vector-only and hybrid (RRF). Everything runs locally (no API cost). ~5 min per experiment.

Run (Codespace terminal):  python experiments.py            all experiments
                           python experiments.py E0 E3      only these
Results: printed table + eval/experiments.md + one line per run in eval/results.jsonl
"""
import json, os, subprocess, sys

# name: (description, env overrides)  - E0 is the baseline; E1/E2 add ONE improvement each.
EXPERIMENTS = {
    "E0": ("baseline (as Step 1)",                    {"RAG_CLEAN": "0", "RAG_CONTEXT_HEADER": "0", "RAG_CHUNK_WORDS": "300"}),
    "E1": ("+ clean (drop cover/TOC pages)",          {"RAG_CLEAN": "1", "RAG_CONTEXT_HEADER": "0", "RAG_CHUNK_WORDS": "300"}),
    "E2": ("+ context header only",                   {"RAG_CLEAN": "0", "RAG_CONTEXT_HEADER": "1", "RAG_CHUNK_WORDS": "300"}),
    "E3": ("+ clean + context header",                {"RAG_CLEAN": "1", "RAG_CONTEXT_HEADER": "1", "RAG_CHUNK_WORDS": "300"}),
    "E4": ("E3 with chunks of 150 words",             {"RAG_CLEAN": "1", "RAG_CONTEXT_HEADER": "1", "RAG_CHUNK_WORDS": "150", "RAG_OVERLAP_WORDS": "25"}),
    "E5": ("E3 with chunks of 500 words",             {"RAG_CLEAN": "1", "RAG_CONTEXT_HEADER": "1", "RAG_CHUNK_WORDS": "500", "RAG_OVERLAP_WORDS": "75"}),
}


def run(cmd, env):
    subprocess.run([sys.executable, *cmd], env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def last_result():
    with open("eval/results.jsonl", encoding="utf-8") as fh:
        return json.loads(fh.readlines()[-1])


def main():
    names = sys.argv[1:] or list(EXPERIMENTS)
    rows = []
    for name in names:
        desc, over = EXPERIMENTS[name]
        env = {**os.environ, "RAG_BM25_STEM": "0", "RAG_RERANK": "0", **over, "RAG_DB_DIR": f"chroma_db_{name}"}
        print(f"[{name}] {desc}: indexing ...", flush=True)
        run(["ingest.py"], env)
        for mode, flags in [("vector", []), ("hybrid-rrf", ["--hybrid", "--fusion", "rrf"])]:
            print(f"[{name}] evaluating {mode} ...", flush=True)
            run(["evaluate.py", *flags, "--note", f"{name} {desc} [{mode}]"], env)
            s = last_result()["summary"]["all"]
            rows.append((name, desc, mode, s["hit"], s["mrr"]))
    lines = ["| Exp | Change | Search | Hit@5 | MRR |", "|---|---|---|---|---|"]
    lines += [f"| {n} | {d} | {m} | {100*h:.0f}% | {r:.2f} |" for n, d, m, h, r in rows]
    table = "\n".join(lines)
    print("\n" + table)
    with open("eval/experiments.md", "w", encoding="utf-8") as fh:
        fh.write(table + "\n")


if __name__ == "__main__":
    main()
