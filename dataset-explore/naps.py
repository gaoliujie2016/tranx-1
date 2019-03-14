import json
import argparse
from pprint import pprint
import random
import numpy as np

def get_by_key(example, key):
    if isinstance(example[key], str):
        ch = " " if key in ['text', 'code_sequence'] else ""
        return ch.join(example[key])
    else:
        return example[key]

def get_all_by_key(data, key):
    return [get_by_key(d, key) for d in data]

def main(args):
    lines = [l.strip() for l in open(args.file, "rt").readlines()]
    data = [dict(json.loads(l)) for l in lines]

    keys = list(data[0].keys())
    print(f"Keys: {keys}\n")

    while True:
        idx = int(input("Index: "))
        keys = [l.strip() for l in str(input("Keys: ")).split()]

        for k in keys:
            print(k, "\t"*2, get_by_key(data[idx], k))
        print("-" * 32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, required=True)

    main(parser.parse_args())
