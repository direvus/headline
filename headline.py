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
NUMBER_PREFIX = re.compile(r'^\d+\.?\s*')
SUB_PROMPT = re.compile(r'^[A-Z].?[A-Z]$')
PLAIN_INTEGER = re.compile(r'^\d+$')
MATCH_LIMIT = 10
DIRNAME = os.path.dirname(sys.argv[0])
WORDLIST = defaultdict(list)


console = Console()


def load_wordlist():
    path = os.path.join(DIRNAME, 'wordlist', 'words')
    with open(path, 'r') as fp:
        for line in fp:
            word = line.strip()
            WORDLIST[len(word)].append(word)


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


def reverse_alphabet(alphabet):
    letters = []
    for c in ascii_uppercase:
        try:
            index = alphabet.index(c)
            letters.append(index_letter(index))
        except ValueError:
            letters.append('_')
    return ''.join(letters)


class CipherView:
    def __init__(self, ciphertext=None, alphabet=None):
        self.set_ciphertext(ciphertext)
        if alphabet is None:
            self.alphabet = [None] * 26
        else:
            self.alphabet = alphabet
        self.words = self.ciphertext.split()
        self.matches = []
        self.update_matches()

    def set_ciphertext(self, text):
        self.ciphertext = text.strip().upper()
        # If there is a numeric prefix, remove it
        self.ciphertext = NUMBER_PREFIX.sub('', self.ciphertext)

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
                        # Use negative lookaheads to require that no previously
                        # matched groups can match here
                        groupnum += 1
                        for i in range(1, groupnum):
                            parts.append(f'(?!\\{i})')
                        parts.append(group)
                        groups[c] = groupnum
                else:
                    parts.append(sub)
        parts.append('$')
        pattern = re.compile(''.join(parts))

        wordlist = WORDLIST.get(len(word), [])
        matches = filter(lambda x: pattern.match(x), wordlist)
        return matches

    def update_matches(self):
        self.matches = []
        for i, word in enumerate(self.words):
            matches = list(self.find_word_matches(word))
            self.matches.append(matches)

    def set_substitution(self, source, target):
        index = letter_index(source)
        self.alphabet[index] = target

    def set_all_substitutions(self, index, target):
        word = self.words[index]
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

    def make_prompt(self):
        max = len(self.words)
        prompt = (
                f"[yellow]1[/]-[yellow]{max}[/] to select a matching word,\n"
                "[yellow]XY[/] to set a substitution, "
                "[yellow]R[/] to reset,\n"
                "[yellow]S[/] to save solution and exit, "
                "or [yellow]X[/] to exit without saving")
        return prompt

    def select_match(self, index):
        if index < 0 or index >= len(self.words):
            return
        matches = self.matches[index]
        if len(matches) == 0:
            return
        if len(matches) == 1:
            self.set_all_substitutions(index, matches[0])
            return

        print()
        count = min(MATCH_LIMIT, len(matches))
        for i in range(count):
            m = matches[i]
            n = i + 1
            print(f'[yellow]{n:3d}[/]. {m}')

        while True:
            prompt = (
                    f"\n[yellow]1[/]-[yellow]{count}[/] to select a word, or "
                    "[yellow]X[/] to exit")
            choice = Prompt.ask(prompt).upper()
            if choice == 'X':
                return
            elif PLAIN_INTEGER.match(choice):
                i = int(choice) - 1
                if i >= 0 and i < count:
                    self.set_all_substitutions(index, matches[i])
                return

    def run(self):
        if self.ciphertext is None:
            text = Prompt.ask("Input the ciphertext")
            self.set_ciphertext(text)

        while True:
            console.clear()
            self.print()
            prompt = self.make_prompt()
            choice = Prompt.ask(prompt).upper()
            if not choice.strip():
                continue
            if SUB_PROMPT.match(choice):
                source = choice[0]
                target = choice[-1]
                self.set_substitution(source, target)
                self.update_matches()
            elif PLAIN_INTEGER.match(choice):
                index = int(choice) - 1
                self.select_match(index)
            elif choice[0] == 'S':
                print("OK, saving solution and exiting.\n")
                return self.alphabet
            elif choice[0] == 'X':
                print("OK, exiting and discarding solution.\n")
                return None
            elif choice[0] == 'R':
                self.alphabet = [None] * 26
                self.update_matches()


class PuzzleView:
    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.ciphers = []
        self.solutions = []
        with open(os.path.join(DIRNAME, 'puzzles', puzzle), 'r') as fp:
            for line in fp:
                line = line.strip().upper()
                # If there is a numeric prefix, remove it
                line = NUMBER_PREFIX.sub('', line)
                self.ciphers.append(line)
                self.solutions.append(None)
        try:
            self.solution_path = os.path.join(DIRNAME, 'solutions', puzzle)
            solution_dir = os.path.dirname(self.solution_path)
            os.makedirs(solution_dir, exist_ok=True)

            with open(self.solution_path, 'r') as fp:
                index = 0
                for line in fp:
                    line = line.strip().upper()
                    alphabet = [
                            x if x in ascii_uppercase else None
                            for x in line[:26]]
                    self.solutions[index] = alphabet
                    index += 1
        except IOError:
            pass

    def print(self):
        lines = []
        for i, cipher in enumerate(self.ciphers):
            n = i + 1
            lines.append(f'[yellow]{n:3d}[/]. {cipher}')
            solution = self.solutions[i]
            if solution is None:
                lines.append('\n\n')
            else:
                line = ' ' * 5 + substitute(cipher, solution)
                if '_' not in line:
                    line = f'[green]{line}[/]'
                lines.append(line)

                alphabet = reverse_alphabet(solution)
                chars = [x or '_' for x in alphabet]
                lines.append(' ' * 5 + ''.join(chars) + '\n')
        print(Panel('\n'.join(lines), title=self.puzzle))

    def save_solutions(self):
        with open(self.solution_path, 'w') as fp:
            for solution in self.solutions:
                if solution is None:
                    fp.write('_' * 26)
                else:
                    chars = [x or '_' for x in solution]
                    fp.write(''.join(chars))
                fp.write('\n')

    def run(self):
        while True:
            console.clear()
            self.print()

            count = len(self.ciphers)
            prompt = (
                    f"\n[yellow]1[/]-[yellow]{count}[/] to select a cipher, "
                    "or [yellow]Q[/] to quit")
            choice = Prompt.ask(prompt).strip().upper()
            if not choice:
                continue
            if PLAIN_INTEGER.match(choice):
                index = int(choice) - 1
                if index >= 0 and index < len(self.ciphers):
                    view = CipherView(self.ciphers[index])
                    solution = view.run()
                    if solution is not None:
                        self.solutions[index] = solution
            elif choice[0] == 'Q':
                print("OK, quitting.\n")
                self.save_solutions()
                break


def main(args):
    load_wordlist()
    if args.interactive:
        menu = PuzzleView(args.puzzle)
        menu.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--interactive', action='store_true')
    parser.add_argument('-p', '--puzzle')

    args = parser.parse_args()
    main(args)
