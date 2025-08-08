#!/usr/bin/env python
import argparse
import os
import re
import sys
from collections import defaultdict
from string import ascii_uppercase

from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt


FREQUENCY = 'ETAOINSHRDLCUMWFGYPBVKJXQZ'
NON_WORD_CHARS = re.compile(r"[^A-Z'-]")
SUB_PROMPT = re.compile(r'^[A-Z].?[A-Z]$')
WORD_PROMPT = re.compile(r'^\d+$')
MATCH_LIMIT = 10


console = Console()


def letter_index(letter):
    return ord(letter) - 65


def index_letter(index):
    return chr(65 + index)


def substitute(cipher, alphabet):
    result = []
    for c in cipher:
        if c not in ascii_uppercase:
            result.append(c)
            continue

        index = letter_index(c)
        sub = alphabet[index]
        if sub is None:
            result.append('_')
        else:
            result.append(sub)
    return ''.join(result)


class Workspace:
    def __init__(self, ciphertext=None):
        self.ciphertext = ciphertext
        self.alphabet = [None] * 26

        self.wordlist = defaultdict(list)
        dirname = os.path.dirname(sys.argv[0])
        path = os.path.join(dirname, 'wordlist', 'words')
        with open(path, 'r') as fp:
            for line in fp:
                word = line.strip()
                self.wordlist[len(word)].append(word)

        self.words = self.ciphertext.split()
        self.matches = []
        self.solos = []
        self.update_matches()

    def find_word_matches(self, word):
        # Strip out punctuation characters except apostrophe and hyphen
        word = NON_WORD_CHARS.sub('', word)

        # Make up a regex character class for all the unsolved letters
        solved = {c for c in self.alphabet if c is not None}
        unsolved = set(ascii_uppercase) - solved
        group = '([' + ''.join(list(unsolved)) + '])'
        parts = ['^']
        groups = {}
        groupnum = 0
        for c in word:
            if c not in ascii_uppercase:
                parts.append(c)
            else:
                index = letter_index(c)
                sub = self.alphabet[index]
                if sub is None:
                    if c in groups:
                        parts.append(f'\\{groups[c]}')
                    else:
                        parts.append(group)
                        groupnum += 1
                        groups[c] = groupnum
                else:
                    parts.append(sub)
        parts.append('$')
        pattern = re.compile(''.join(parts))

        wordlist = self.wordlist.get(len(word), [])
        matches = filter(lambda x: pattern.match(x), wordlist)
        return matches

    def update_matches(self):
        self.matches = []
        self.solos = []
        for i, word in enumerate(self.words):
            matches = list(self.find_word_matches(word))
            self.matches.append(matches)
            if len(matches) == 1 and not self.is_word_solved(i):
                self.solos.append(i)

    def set_substitution(self, source, target):
        index = letter_index(source)
        self.alphabet[index] = target

    def set_all_substitutions(self, index):
        word = self.words[index]
        target = self.matches[index][0]
        for i, c in enumerate(word):
            if c in ascii_uppercase:
                self.set_substitution(c, target[i])
        self.update_matches()

    def is_word_solved(self, index):
        word = self.words[index]
        for c in word:
            if c in ascii_uppercase:
                if self.alphabet[letter_index(c)] is None:
                    return False
        return True

    def print(self):
        clears = []
        parts = []
        for i, word in enumerate(self.words):
            clear = substitute(word, self.alphabet)
            if self.is_word_solved(i):
                clears.append(f'[green]{clear}[/]')
            else:
                clears.append(clear)

            num = len(self.matches[i])
            if num > MATCH_LIMIT:
                if num > MATCH_LIMIT ** 2:
                    style = 'red'
                else:
                    style = 'yellow'
            else:
                style = 'green'

            text = str(num)
            if len(text) > len(word):
                part = '.' * len(word)
            else:
                pad = ' ' * (len(word) - len(text))
                part = pad + text
            parts.append(f'[{style}]' + part + '[/]')
        lines = [' '.join(parts)]
        print(Panel(self.ciphertext + '\n' + ' '.join(clears)))

        for i in range(MATCH_LIMIT):
            parts = []
            for j, word in enumerate(self.words):
                m = self.matches[j]
                if i < len(m):
                    part = m[i]
                    pad = ' ' * (len(word) - len(part))
                    parts.append(part + pad)
                else:
                    parts.append(' ' * len(word))
            lines.append(' '.join(parts))
        print(Panel('\n'.join(lines)))

        letters = []
        targets = []
        unused = []
        for i, c in enumerate(ascii_uppercase):
            if c in self.ciphertext:
                target = self.alphabet[i]
                if target:
                    letters.append(f'[green]{c}[/]')
                    targets.append(f'[green]{target}[/]')
                else:
                    letters.append(f'[red]{c}[/]')
                    targets.append(' ')
            else:
                letters.append(' ')
                targets.append(' ')
            if c in self.alphabet:
                unused.append(' ')
            else:
                unused.append(c)

        print(Panel(
                ' '.join(letters) + '\n' +
                ' '.join(targets) + '\n' +
                '[yellow]' + ' '.join(unused) + '[/]'))
        print()

    def print_reverse_alphabet(self):
        print("Your final substitution alphabet was:\n")
        letters = []
        for c in ascii_uppercase:
            try:
                index = self.alphabet.index(c)
                letters.append(index_letter(index))
            except ValueError:
                letters.append('_')
        print(''.join(letters))

    def make_prompt(self):
        prompt = ''
        if self.solos:
            labels = []
            for i in self.solos:
                labels.append(f'[yellow]{i + 1}[/]')

            prompt += (
                    ', '.join(labels) + ' to set all substitutions '
                    'for that word\n')

        prompt += (
                "[yellow]XY[/] to set a substitution, or "
                "[yellow]Q[/] to quit")
        return prompt

    def run(self):
        if self.ciphertext is None:
            text = Prompt.ask("Input the ciphertext")
            self.ciphertext = text

        while True:
            console.clear()
            self.print()
            prompt = self.make_prompt()
            choice = Prompt.ask(prompt).upper()
            if SUB_PROMPT.match(choice):
                source = choice[0]
                target = choice[-1]
                self.set_substitution(source, target)
                self.update_matches()
            elif WORD_PROMPT.match(choice):
                index = int(choice) - 1
                if index in self.solos:
                    self.set_all_substitutions(index)
            elif len(choice) > 0 and choice[0] == 'Q':
                print("OK, quitting.\n")
                self.print_reverse_alphabet()
                break


def main(args):
    if args.interactive:
        w = Workspace(args.ciphertext)
        w.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--interactive', action='store_true')
    parser.add_argument('-c', '--ciphertext')

    args = parser.parse_args()
    main(args)
