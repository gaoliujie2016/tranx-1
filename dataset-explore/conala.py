import json
import argparse
import code
from pprint import pprint

import numpy as np


def get_unique(xs):
    u, c = np.unique(xs, return_counts=True)
    _c = np.argsort(-c)

    return u[_c], c[_c]

def query_by(data, key):
    return [data[i][key] for i in range(len(data))]

def get_by_qid(data, qid):
    return filter(lambda ex: ex["question_id"] == qid, data)

def main(args):
    data = json.load(open(args.file, "rt"))

    keys = data[0].keys()

    uids, cuids = get_unique(query_by(data, "question_id"))

    print(f"There are {uids.size} unique ids")

    # code.interact(local=locals())

    for i in range(10):
        print(f"Top {i}:")
        for q in get_by_qid(data, uids[i]):
            print(q["rewritten_intent"])
            print(q["snippet"])
            print("-" * 10)
        print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, required=True)

    main(parser.parse_args())
