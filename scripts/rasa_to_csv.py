#!/usr/bin/env python3
# Converts a Rasa NLU nlu.yml into data/intents.csv (text,intent)
# Usage:
#   python scripts/rasa_to_csv.py --in path/to/nlu.yml --out data/intents.csv
import argparse, csv, os, re
try:
    import yaml  # pip install pyyaml
except ImportError:
    raise SystemExit("Please: pip install pyyaml")

def parse_examples(raw):
    """
    Rasa examples are often a multiline string with lines beginning with '- ':
      examples: |
        - hi
        - hello there
    """
    lines = []
    for line in raw.splitlines():
        m = re.match(r"^\s*-\s+(.*)$", line)
        if m:
            lines.append(m.group(1).strip())
    return lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Rasa nlu.yml path")
    ap.add_argument("--out", dest="outp", default="data/intents.csv", help="output CSV path")
    args = ap.parse_args()

    with open(args.inp, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    nlu = doc.get("nlu", [])
    rows = []
    for block in nlu:
        intent = block.get("intent")
        ex = block.get("examples")
        if not intent or not ex:
            continue
        for text in parse_examples(ex):
            rows.append({"text": text, "intent": intent})

    os.makedirs(os.path.dirname(args.outp), exist_ok=True)
    with open(args.outp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text","intent"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows → {args.outp}")

if __name__ == "__main__":
    main()
