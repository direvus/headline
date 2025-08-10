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
    highlights = 'VWXYZ'
    labels = [
            f'[bold green]{x}[/]' if x in highlights else x
            for x in sequence]
    print('     ' + ''.join(labels))
    for step in (3, 5, 7, 9, 11):
        decimation = decimate(sequence, step)
        labels = [
                f'[bold green]{x}[/]' if x in highlights else x
                for x in decimation]
        print(f'[yellow]{step:3d}[/]. ' + ''.join(labels))


if __name__ == '__main__':
    main(sys.argv[1])
