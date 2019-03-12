import json
import argparse
import code
from pprint import pprint
from termcolor import colored

import numpy as np
from typing import Dict, Any

keywords = [
    "django", "os.", "urllib.", "sys.", "scipy.", "numpy.",
    "pickle.", "struct.", "subprocess.", "datetime.", "time.",
    "request.", "re.", "map(", "filter(", "reduce(",
    "print("
]

def get_unique(xs):
    u, c = np.unique(xs, return_counts=True)
    _c = np.argsort(-c)

    return u[_c], c[_c]

def query_by_key(data: Dict[str, Any], key):
    return [data[i][key] for i in range(len(data))]

def get_by_qid(data: Dict[str, Any], qid: int):
    return filter(lambda ex: ex["question_id"] == qid, data)

def get_by_keywords(data: Dict[str, Any]):
    xs = {l: [] for l in keywords}

    for q in data:
        i = q["intent"]
        ri = q["rewritten_intent"]
        s = q["snippet"]
        for l in keywords:
            if (i and l in i) or (ri and l in ri) or (s and l in s):
                xs[l].append(q)

    return xs

def main(args):
    data = json.load(open(args.file, "rt"))

    keys = data[0].keys()

    uids, cuids = get_unique(query_by_key(data, "question_id"))

    print(f"There are {uids.size} unique ids")

    input()

    xs = get_by_keywords(data)
    print(len(data))

    for l, qs in sorted(xs.items(), key=lambda k : len(k[1]), reverse=True):
        print(f"{l} -> {len(qs)}({round(100.0 * len(qs)/len(data), 3)}%)")
    print()

    code.interact(local=locals())

    for l, qs in xs.items():
        print(colored(l, "green"))
        input()
        for q in qs:
            pprint(q)
            print()

    # for i, uid in enumerate(uids, start=1):
    #     print(f"{i}:")
    #     for q in get_by_qid(data, uid):
    #         print("> ", q["rewritten_intent"])
    #         print("> ", q["snippet"])
    #         print()
    #     print("-"*32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, required=True)

    main(parser.parse_args())
