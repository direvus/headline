#!/usr/bin/env python
import os
import sys
from collections import defaultdict


DIRNAME = os.path.dirname(sys.argv[0])
WORDLIST = defaultdict(list)


def load_wordlist():
    path = os.path.join(DIRNAME, 'wordlist', 'words')
    with open(path, 'r') as fp:
        for line in fp:
            word = line.strip()
            WORDLIST[len(word)].append(word)


def get_sequence(word):
    result = [0] * len(word)
    n = 1
    for c in sorted(set(word)):
        for i, x in enumerate(word):
            if x == c:
                result[i] = n
                n += 1
    return result


def main(seq):
    load_wordlist()
    length = len(seq)
    words = WORDLIST[length]
    for word in words:
        if seq == get_sequence(word):
            print(word)


if __name__ == '__main__':
    seq = [int(x) for x in sys.argv[1:]]
    main(seq)
