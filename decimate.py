#!/usr/bin/env python
import sys

from rich import print


def decimate(sequence, step):
    result = []
    i = 0
    length = len(sequence)
    for _ in range(length):
        result.append(sequence[i])
        i = (i + step) % length
    return result


def main(sequence):
    for step in (3, 5, 7, 9, 11):
        result = decimate(sequence, step)
        text = ''.join(result)
        print(f'[yellow]{step:2d}[/]. {text}')


if __name__ == '__main__':
    main(sys.argv[1])
