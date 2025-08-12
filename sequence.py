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


def build_comparisons(sequence):
    result = []
    for i in range(1, len(sequence)):
        comps = []
        for j in range(i):
            comps.append(sequence[i] < sequence[j])
        result.append(tuple(comps))
    return tuple(result)


def get_sequence(word):
    result = [0] * len(word)
    n = 1
    for c in sorted(set(word)):
        for i, x in enumerate(word):
            if x == c:
                result[i] = n
                n += 1
    return result


def test_word(comps, word):
    for i in range(1, len(word)):
        letter = word[i]
        for j, lt in enumerate(comps[i - 1]):
            if lt != (letter < word[j]):
                return False
    return True


def main(seq):
    load_wordlist()
    length = len(seq)
    comps = build_comparisons(seq)

    words = WORDLIST[length]
    for word in words:
        if test_word(comps, word):
            print(word)

    # Concatenations of two words
    for i in range(1, length):
        for word1 in WORDLIST[i]:
            if not test_word(comps, word1):
                continue
            j = length - i
            for word2 in WORDLIST[j]:
                if test_word(comps, word1 + word2):
                    print(f'{word1} {word2}')


if __name__ == '__main__':
    seq = [int(x) for x in sys.argv[1:]]
    main(seq)
