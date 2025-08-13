#!/usr/bin/env python3
# Converts a SNIPS-style dataset JSON into data/intents.csv (text,intent)
# Usage:
#   python scripts/snips_to_csv.py --in path/to/train.json --out data/intents.csv
import json, csv, argparse, os, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="SNIPS JSON (e.g., train.json)")
    ap.add_argument("--out", dest="outp", default="data/intents.csv", help="output CSV path")
    args = ap.parse_args()

    with open(args.inp) as f:
        data = json.load(f)

    # expected structure: { "intents": { "IntentName": { "utterances": [ { "data": [ {"text": "..."} , ... ] }, ... ] } , ... } }
    if "intents" not in data:
        print("ERROR: JSON missing 'intents' key. Check your SNIPS file.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for intent, body in data["intents"].items():
        for utt in body.get("utterances", []):
            text = "".join([seg.get("text", "") for seg in utt.get("data", [])]).strip()
            if text:
                rows.append({"text": text, "intent": intent})

    os.makedirs(os.path.dirname(args.outp), exist_ok=True)
    with open(args.outp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text","intent"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows → {args.outp}")

if __name__ == "__main__":
    main()
