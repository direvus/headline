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

from decimate import decimate


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
        result.append('_' if sub is None else sub)
    return ''.join(result)


def reverse_alphabet(alphabet):
    letters = []
    for c in ascii_uppercase:
        try:
            index = alphabet.index(c)
            letters.append(index_letter(index))
        except ValueError:
            letters.append(None)
    return letters


def reorder_alphabet(alphabet, key):
    for i, k in enumerate(key):
        c = alphabet[letter_index(k)]
        if c is not None and c in key:
            j = key.index(c)
            offset = j - i
            return key[offset:] + key[:offset]


def serialise_alphabet(alphabet):
    chars = [x or '_' for x in alphabet]
    return ''.join(chars)


def deserialise_alphabet(text):
    return [x if x in ascii_uppercase else None for x in text]


def find_chains(alphabet):
    if alphabet is None:
        return set()
    values = set(alphabet)
    if values == {None}:
        return set()
    chains = set()
    for i, c in enumerate(alphabet):
        source = index_letter(i)
        # Look for a source that isn't a known target, so we know we're at the
        # beginning of a chain
        if c is None or source in values:
            continue
        target = c
        chain = [source]
        while target is not None and target in ascii_uppercase:
            chain.append(target)
            target = alphabet[letter_index(target)]
        chains.add(''.join(chain))
    return chains


class ChainGrid:
    MIN_Y = -7
    MAX_Y = 7

    def __init__(self, across, down):
        self.cells = {}
        self.candidates_v = set()
        self.candidates_h = set()
        self.run_length = 0
        self.run_start = None
        self.across = sorted(across, key=len, reverse=True)
        self.down = sorted(down, key=len, reverse=True)

        self.add_across(self.across[0], 0, 0)
        count = 0
        while self.run_length < 26 and count < 4:
            for chain in self.down:
                self.find_intersections_down(chain)
            for chain in self.across:
                self.find_intersections_across(chain)
            self.run_length, self.run_start = self.find_longest_run()
            count += 1

    def find_longest_run(self):
        locs = self.cells.keys()
        xs = {x for x, _ in locs}
        ys = {y for _, y in locs}
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        max_length = 0
        best_start = None
        chars = set()
        for y in range(min_y, max_y + 1):
            length = 0
            start = None
            for x in range(min_x, max_x + 1):
                loc = (x, y)
                if loc not in self.cells:
                    length = 0
                    start = None
                    chars = set()
                    continue
                if start is None:
                    start = loc
                char = self.cells[loc]
                if char in chars:
                    # Loop detected, break the run here
                    length = 1
                    start = loc
                    chars = {char}
                    continue
                chars.add(char)
                length += 1
                if length > max_length:
                    max_length = length
                    best_start = start
        return (max_length, best_start)

    def get_run(self):
        if self.run_start is None:
            return None
        x, y = self.run_start
        end = x + self.run_length
        chars = []
        while x < end:
            chars.append(self.cells.get((x, y), ' '))
            x += 1
        return ''.join(chars)

    def add_across(self, text, x, y):
        for c in text:
            loc = (x, y)
            if y < ChainGrid.MIN_Y or y > ChainGrid.MAX_Y:
                continue
            self.cells[loc] = c
            self.candidates_v.add(loc)
            if loc in self.candidates_h:
                self.candidates_h.remove(loc)
            x += 1

    def add_down(self, text, x, y):
        for c in text:
            loc = (x, y)
            if y < ChainGrid.MIN_Y or y > ChainGrid.MAX_Y:
                continue
            self.cells[loc] = c
            self.candidates_h.add(loc)
            if loc in self.candidates_v:
                self.candidates_v.remove(loc)
            y += 1

    def find_intersections_down(self, text):
        chars = set(text)
        for loc in list(self.candidates_v):
            char = self.cells[loc]
            if char not in chars:
                continue
            x, y = loc
            y -= text.index(char)
            self.add_down(text, x, y)

    def find_intersections_across(self, text):
        chars = set(text)
        for loc in list(self.candidates_h):
            char = self.cells[loc]
            if char not in chars:
                continue
            x, y = loc
            x -= text.index(char)
            self.add_across(text, x, y)

    def __str__(self):
        lines = []
        locs = self.cells.keys()
        xs = {x for x, _ in locs}
        ys = {y for _, y in locs}
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        for y in range(min_y, max_y + 1):
            line = []
            x = min_x
            while x < max_x + 1:
                loc = (x, y)
                if loc == self.run_start:
                    run = self.get_run()
                    line.append(f'[green]{run}[/]')
                    x += self.run_length
                else:
                    line.append(self.cells.get((x, y), ' '))
                    x += 1
            lines.append(''.join(line))
        return '\n'.join(lines)


class ChainView:
    def __init__(self, alphabets=None):
        self.chain_sets = []
        self.grid = None
        self.across = None
        self.down = None
        if alphabets:
            for a in alphabets:
                self.add_alphabet(a)

    def add_alphabet(self, alphabet):
        chains = find_chains(alphabet)
        self.chain_sets.append(chains)

    def extend_chain(self, index, chain, char):
        chainset = self.chain_sets[index]
        if chain not in chainset:
            return
        chainset.remove(chain)
        chain += char

        # Look for any other chains that start with the new ending character
        # and merge them together.
        for other in list(chainset):
            if other.startswith(char[-1]):
                chainset.remove(other)
                chain += other[1:]
        chainset.add(chain)

    def make_grid(self):
        # By default, the chain set with the longest chain will be written
        # horizontally, and the set with the second-longest will be written
        # vertically.
        setlist = [(i, x) for i, x in enumerate(self.chain_sets)]
        setlist.sort(key=lambda x: max([len(y) for y in x[1]]) if x[1] else 0)

        if self.across is None:
            self.across = setlist.pop()[0]
        if self.down is None:
            self.down = setlist.pop()[0]
        grid = ChainGrid(
                self.chain_sets[self.across],
                self.chain_sets[self.down])
        return grid

    def select_extension(self):
        print()
        count = len(self.chain_sets)
        for i, chainset in enumerate(self.chain_sets):
            n = i + 1
            text = ' '.join(chainset)
            print(f'[yellow]{n:3d}[/]. {text}')

        while True:
            prompt = (
                    f"\n[yellow]1[/]-[yellow]{count}[/] to select "
                    "a chain set, or [yellow]X[/] to exit")
            choice = Prompt.ask(prompt).upper()
            if choice == 'X':
                return
            elif PLAIN_INTEGER.match(choice):
                i = int(choice) - 1
                if i >= 0 and i < count:
                    index = i
                    break

        chainset = self.chain_sets[index]
        chains = list(chainset)
        chains.sort(key=len, reverse=True)
        count = len(chains)
        print()
        for i, chain in enumerate(chains):
            n = i + 1
            print(f'[yellow]{n:3d}[/]. {chain}')

        while True:
            prompt = (
                    f"\n[yellow]1[/]-[yellow]{count}[/] to select a chain, or "
                    "[yellow]X[/] to exit")
            choice = Prompt.ask(prompt).upper()
            if choice == 'X':
                return
            elif PLAIN_INTEGER.match(choice):
                i = int(choice) - 1
                if i >= 0 and i < count:
                    chain = chains[i]
                    break

        prompt = f"Enter characters to append to '{chain}'"
        choice = Prompt.ask(prompt).strip().upper()
        self.extend_chain(index, chain, choice)
        self.grid = self.make_grid()

    def merge_run(self):
        chainset = self.chain_sets[self.across]
        run = self.grid.get_run()
        run_chains = {x for x in chainset if x in run}
        if len(run_chains) > 1:
            chainset = chainset - run_chains
            chainset.add(run)
            self.chain_sets[self.across] = chainset

    def print(self):
        lines = []
        for i, chainset in enumerate(self.chain_sets):
            n = i + 1
            text = ' '.join(chainset)
            if i == self.across:
                marker = ' :arrow_right:'
            elif i == self.down:
                marker = ' :arrow_down:'
            else:
                marker = '  '

            lines.append(f'[yellow]{marker}{n:2d}[/]. {text}')
        print(Panel('\n'.join(lines), title='Chain listing'))

        if self.grid:
            text = str(self.grid)
            print(Panel(text, title='Chain grid'))

            run = self.grid.get_run()[:26]
            length = len(run)
            label = run
            if length == 26:
                label = f'[green]{label}[/]'
            print(Panel(label, title=f'Longest run ([yellow]{length}[/])'))
        else:
            text = "Populate at least two chain sets to build a grid"
            print(Panel(text, title='Chain grid'))

        print()

    def run(self):
        if self.grid is None:
            self.grid = self.make_grid()

        while True:
            console.clear()
            self.print()
            prompt = (
                    "[yellow]A<N>[/] or [yellow]D<N>[/] to set the "
                    "Across or Down selection,\n"
                    "[yellow]M[/] to merge chains in the longest run,\n"
                    "[yellow]E[/] to extend a chain manually, or\n"
                    "[yellow]X[/] to exit chain view")
            choice = Prompt.ask(prompt).strip().upper()

            if re.match(r'^A(\d+)$', choice):
                index = int(choice[1:]) - 1
                if (index >= 0 and index < len(self.chain_sets) and
                        index != self.down):
                    self.across = index
                    self.grid = self.make_grid()
            elif re.match(r'^D(\d+)$', choice):
                index = int(choice[1:]) - 1
                if (index >= 0 and index < len(self.chain_sets) and
                        index != self.across):
                    self.down = index
                    self.grid = self.make_grid()
            elif choice == 'M':
                self.merge_run()
            elif choice == 'E':
                self.select_extension()
            elif choice == 'X':
                run = self.grid.get_run()[:26]
                if len(run) == 26:
                    return run
                return None


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
        self.chain = None
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
                    if line == '':
                        break
                    alphabet = [
                            x if x in ascii_uppercase else None
                            for x in line[:26]]
                    self.solutions[index] = alphabet
                    index += 1
                # Any remaining lines will be:
                # 1. Chain
                # 2. Key
                # 3. Hat
                lines = [line.strip().upper() for line in fp]
                if lines:
                    line = lines.pop()
                    if line:
                        # Chain
                        self.chain = line
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
                text = serialise_alphabet(alphabet)
                lines.append(' ' * 5 + text + '\n')
        print(Panel('\n'.join(lines), title=self.puzzle))

        if self.has_complete_chain():
            lines = ['     ' + self.chain]
            for i, solution in enumerate(self.solutions):
                if solution is None:
                    lines.append('')
                    continue
                alpha = reverse_alphabet(solution)
                alpha = reorder_alphabet(alpha, self.chain)
                n = i + 1
                marker = f'[yellow]{n:3d}[/]. '
                text = serialise_alphabet(alpha)
                lines.append(marker + f'[green]{text[0]}[/]{text[1:]}')
            print(Panel('\n'.join(lines), title='Reordered alphabets'))

            highlights = 'VWXYZ'
            labels = [
                    f'[bold green]{x}[/]' if x in highlights else x
                    for x in self.chain]
            lines = ['     ' + ''.join(labels)]
            for step in (3, 5, 7, 9, 11):
                decimation = decimate(self.chain, step)
                labels = [
                        f'[bold green]{x}[/]' if x in highlights else x
                        for x in decimation]
                lines.append(f'[yellow]{step:3d}[/]. ' + ''.join(labels))
            print(Panel('\n'.join(lines), title='Chain decimations'))

    def has_complete_chain(self):
        return self.chain is not None and len(self.chain) == 26

    def save_solutions(self):
        with open(self.solution_path, 'w') as fp:
            for solution in self.solutions:
                if solution is None:
                    fp.write('_' * 26)
                else:
                    fp.write(serialise_alphabet(solution))
                fp.write('\n')

            fp.write('\n')
            fp.write(self.chain or '')
            fp.write('\n')

    def select_setting(self):
        lines = []
        markers = []
        for n in range(1, 27):
            markers.append(f'{n:3d}')
        lines.append('[yellow]' + ''.join(markers) + '[/]')

        for solution in self.solutions:
            if solution is None:
                lines.append('')
                continue
            alpha = reverse_alphabet(solution)
            alpha = reorder_alphabet(alpha, self.chain)
            text = serialise_alphabet(alpha)
            lines.append('  ' + '  '.join(text))
        print('\n' + '\n'.join(lines))

        while True:
            prompt = (
                    "\n[yellow]1[/]-[yellow]26[/] to select a setting, or\n"
                    "[yellow]X[/] to exit")
            choice = Prompt.ask(prompt).strip().upper()
            if PLAIN_INTEGER.match(choice):
                index = int(choice) - 1
                if index >= 0 and index < 26:
                    self.chain = self.chain[index:] + self.chain[:index]
                return
            elif choice[0] == 'X':
                return

    def run(self):
        while True:
            console.clear()
            self.print()

            count = len(self.ciphers)
            prompt = (
                    f"\n[yellow]1[/]-[yellow]{count}[/] to select a cipher,\n"
                    "[yellow]C[/] to view chains, ")
            if self.has_complete_chain():
                prompt += "\n[yellow]T[/] to choose a setting, "

            prompt += "or\n[yellow]Q[/] to quit"
            choice = Prompt.ask(prompt).strip().upper()
            if not choice:
                continue
            if PLAIN_INTEGER.match(choice):
                index = int(choice) - 1
                if index >= 0 and index < len(self.ciphers):
                    view = CipherView(
                            self.ciphers[index],
                            self.solutions[index])
                    solution = view.run()
                    if solution is not None:
                        self.solutions[index] = solution
            elif choice[0] == 'C':
                alphas = []
                for solution in self.solutions:
                    if solution is not None:
                        alphas.append(reverse_alphabet(solution))
                view = ChainView(alphas)
                self.chain = view.run()
            elif choice[0] == 'T' and self.has_complete_chain():
                self.select_setting()
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
