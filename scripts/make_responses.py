#!/usr/bin/env python3
# Generates data/responses.json from data/intents.csv
# Usage:
#   python scripts/make_responses.py --csv data/intents.csv --out data/responses.json
import argparse, csv, json, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="input CSV (text,intent)")
    ap.add_argument("--out", default="data/responses.json", help="output JSON")
    args = ap.parse_args()

    intents = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            intents.append(row["intent"])
    unique = sorted(set(intents))

    resp = {}
    for intent in unique:
        nice = intent.replace("_", " ").replace(".", " ").title()
        resp[intent] = [
            f"Routing to the best agent for {nice}.",
            f"{nice} request detected. Connecting you to the appropriate agent."
        ]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(resp, f, indent=2, ensure_ascii=False)
    print(f"Wrote responses for {len(unique)} intents → {args.out}")

if __name__ == "__main__":
    main()
