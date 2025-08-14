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
from sequence import build_comparisons, test_word


FREQUENCY = 'ETAOINSHRDLCUMWFGYPBVKJXQZ'
NON_WORD_CHARS = re.compile(r"[^A-Z'-]")
NUMBER_PREFIX = re.compile(r'^\d+\.?\s*')
SUB_PROMPT = re.compile(r'^[A-Z].?[A-Z]$')
PLAIN_INTEGER = re.compile(r'^\d+$')
MATCH_LIMIT = 10
DIRNAME = os.path.dirname(sys.argv[0])
WORDLIST = defaultdict(list)
DECIMATIONS = (3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25)
HIGHLIGHTS = 'VWXYZ'
UNKNOWN_LABEL = '[grey30]None[/]'


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


def highlight(text):
    return f'[green]{text}[/]'


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
        word = NON_WORD_CHARS.sub('', self.words[index])
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
                if index >= 0 and index < len(self.chain_sets):
                    if index == self.down:
                        self.down = self.across
                    self.across = index
                    self.grid = self.make_grid()
            elif re.match(r'^D(\d+)$', choice):
                index = int(choice[1:]) - 1
                if index >= 0 and index < len(self.chain_sets):
                    if index == self.across:
                        self.across = self.down
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


class KeyView:
    def __init__(self, chain, key='', sequence=None):
        self.key = key or ''
        self.step = None
        self.chain = chain
        self.sequence = sequence
        self.decimations = {1: self.chain}
        for step in DECIMATIONS:
            decimation = decimate(self.chain, step)
            self.decimations[step] = decimation

    def highlight_chain(self, chain):
        labels = [highlight(x) if x in HIGHLIGHTS else x for x in chain]
        return ''.join(labels)

    def make_matrix(self, step, width):
        chain = ''.join(self.decimations[step])
        letters = set(chain) - set(self.key)
        alphabet = self.key + ''.join(sorted(letters))

        matrix = []
        columns = [[] for _ in range(width)]
        row = []
        for i in range(len(alphabet)):
            mod = i % width
            letter = alphabet[i]
            columns[mod].append(letter)
            row.append(letter)
            if mod == width - 1:
                matrix.append(row)
                row = []
        if row:
            matrix.append(row)

        matches = set()
        for col in columns:
            for i in range(len(col) - 1):
                seq = ''.join(col[i:])
                if seq in chain:
                    matches |= set(seq)
                    break
        return matrix, matches

    def get_sequence(self, matrix, matches):
        chain = self.decimations[self.step]
        row = matrix[0]

        # Sort the letters from the top row of the matrix, according to their
        # order of appearance in the chain
        letters = sorted(row, key=chain.index)

        # Pair the letters with their index numbers, and sort it according to
        # the order that the letters appear in the top row.
        pairs = sorted(enumerate(letters), key=lambda x: row.index(x[1]))
        return tuple((i + 1 for i, c in pairs))

    def print(self):
        lines = []
        for step in sorted(self.decimations.keys()):
            text = self.highlight_chain(self.decimations[step])
            marker = '[bold yellow]*[/]' if step == self.step else ' '
            lines.append(f'{marker}[yellow]{step:4d}[/]. {text}')
        print(Panel('\n'.join(lines), title='Chain decimations'))

    def print_matrix(self, matrix, matches):
        lines = []
        width = len(matrix[0])
        chain = self.decimations[self.step]
        if len(matches) == len(chain):
            self.sequence = self.get_sequence(matrix, matches)
            markers = (f'[yellow]{n:3d}[/]' for n in self.sequence)
            lines.append(''.join(markers))

        for row in matrix:
            cells = [
                    highlight(x) if x in matches else x
                    for x in row]
            lines.append('  ' + '  '.join(cells))
        lines.append('')
        letters = [highlight(x) if x in matches else x for x in chain]
        lines.append(''.join(letters))
        print(Panel('\n'.join(lines), title=f'Hat width {width}'))

    def run(self):
        while True:
            console.clear()
            self.print()

            if self.step:
                for width in range(7, 13):
                    matrix, matches = self.make_matrix(self.step, width)
                    self.print_matrix(matrix, matches)

            prompt = (
                    '\n[yellow]<N>[/] to select a step, '
                    '[yellow]K[/] to enter a key, or '
                    '[yellow]X[/] to exit')
            choice = Prompt.ask(prompt).strip().upper()

            if PLAIN_INTEGER.match(choice):
                step = int(choice)
                if step in self.decimations:
                    self.step = step
            elif choice == 'K':
                prompt = 'Enter the new key'
                key = Prompt.ask(prompt).strip().upper()
                self.key = ''.join([x for x in key if x in self.chain])
            elif choice == 'X':
                return (self.key, self.sequence)


class HatView:
    def __init__(self, sequence):
        self.sequence = sequence
        self.comparisons = build_comparisons(sequence)
        self.filter = None
        self.limit = 30
        self.hat = None

    def test_word(self, word):
        return test_word(self.comparisons, word)

    def get_matches(self):
        length = len(self.sequence)
        words = WORDLIST[length]
        if self.filter:
            words = filter(lambda x: self.filter in x, words)
        results = list(filter(self.test_word, words))

        # Concatenations of two words
        for i in range(1, length):
            j = length - i
            for a in WORDLIST[i]:
                # Don't bother with this as a first word if it doesn't match
                # the first part of the sequence
                if not self.test_word(a):
                    continue
                words = (a + b for b in WORDLIST[j])
                if self.filter:
                    words = filter(lambda x: self.filter in x, words)
                results.extend(filter(self.test_word, words))
        return results

    def print(self):
        lines = []
        if self.sequence:
            label = ' '.join((f'{x:2d}' for x in self.sequence)).strip()
        else:
            label = UNKNOWN_LABEL
        lines.append(f' [yellow]Sequence[/]: {label}')

        label = self.filter or UNKNOWN_LABEL
        lines.append(f' [yellow]  Filter[/]: {label}')

        settings_panel = Panel('\n'.join(lines), title='Hat settings')

        lines = []
        self.matches = self.get_matches()
        num_matches = len(self.matches)
        for i, m in enumerate(self.matches[:self.limit]):
            n = i + 1
            lines.append(f'[yellow]{n:3d}[/]. {m}')
        if num_matches > self.limit:
            excess = num_matches - self.limit
            lines.append(f'[yellow]... and {excess} more[/]')
        matches_panel = Panel('\n'.join(lines), title='Matches')

        print(settings_panel)
        print(matches_panel)

    def run(self):
        while True:
            console.clear()
            self.print()

            prompt = (
                    '\n[yellow]<N>[/] to select a hat, '
                    '[yellow]F[/] to set a filter, or '
                    '[yellow]X[/] to exit')
            choice = Prompt.ask(prompt).strip().upper()

            if PLAIN_INTEGER.match(choice):
                choice = int(choice)
                if choice >= 0 and choice < len(self.matches):
                    self.hat = self.matches[choice - 1]
                    return self.hat
            elif choice == 'F':
                prompt = (
                        'Enter the filter text to set, or leave blank to '
                        'remove the filter')
                self.filter = Prompt.ask(prompt).strip().upper()
            elif choice == 'X':
                return self.hat


class PuzzleView:
    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.ciphers = []
        self.solutions = []
        self.chain = None
        self.setting = None
        self.key = None
        self.sequence = None
        self.hat = None
        with open(self.puzzle, 'r') as fp:
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
                # 3. Sequence
                # 4. Hat
                lines = [line.strip().upper() for line in fp]
                if len(lines) > 0 and lines[0]:
                    self.chain = lines[0]
                if len(lines) > 1 and lines[1]:
                    self.key = lines[1]
                if len(lines) > 2 and lines[2]:
                    self.sequence = tuple((int(x) for x in lines[2].split()))
                if len(lines) > 3 and lines[3]:
                    self.hat = lines[3]
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
        cipher_panel = Panel('\n'.join(lines), title='Ciphers')

        setting_panel = None
        decimation_panel = None
        if self.has_complete_chain():
            setting = []
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
                setting.append(text[0])
            text = '\n'.join(lines)
            setting_panel = Panel(text, title='Reordered alphabets')
            self.setting = ''.join(setting)

            labels = [
                    highlight(x) if x in HIGHLIGHTS else x
                    for x in self.chain]
            lines = ['     ' + ''.join(labels)]
            for step in DECIMATIONS:
                decimation = decimate(self.chain, step)
                labels = [
                        highlight(x) if x in HIGHLIGHTS else x
                        for x in decimation]
                lines.append(f'[yellow]{step:3d}[/]. ' + ''.join(labels))
            text = '\n'.join(lines)
            decimation_panel = Panel(text, title='Chain decimations')

        lines = []
        label = self.chain or UNKNOWN_LABEL
        lines.append(f' [yellow]   Chain[/]: {label}')

        label = self.setting or UNKNOWN_LABEL
        lines.append(f' [yellow] Setting[/]: {label}')

        label = self.key or UNKNOWN_LABEL
        lines.append(f' [yellow]     Key[/]: {label}')

        if self.sequence:
            label = ' '.join((f'{x:2d}' for x in self.sequence)).strip()
        else:
            label = UNKNOWN_LABEL
        lines.append(f' [yellow]Sequence[/]: {label}')

        label = self.hat or UNKNOWN_LABEL
        lines.append(f' [yellow]     Hat[/]: {label}')

        summary_panel = Panel('\n'.join(lines), title=self.puzzle)

        print(summary_panel)
        print(cipher_panel)

        if setting_panel:
            print(setting_panel)
        if decimation_panel:
            print(decimation_panel)

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
            fp.write(self.key or '')
            fp.write('\n')
            if self.sequence:
                fp.write(' '.join((str(x) for x in self.sequence)))
            else:
                fp.write('')
            fp.write('\n')
            fp.write(self.hat or '')
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
                prompt += (
                        "\n[yellow]T[/] to choose a setting, "
                        "[yellow]K[/] to find the key, ")
                if self.sequence:
                    prompt += '[yellow]H[/] to find the hat, '

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
            elif choice[0] == 'K' and self.has_complete_chain():
                view = KeyView(self.chain, self.key, self.sequence)
                self.key, self.sequence = view.run()
            elif choice[0] == 'H' and self.sequence:
                view = HatView(self.sequence)
                self.hat = view.run()
            elif choice[0] == 'Q':
                print("OK, quitting.\n")
                self.save_solutions()
                break


def main(args):
    load_wordlist()
    menu = PuzzleView(args.puzzle)
    menu.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('puzzle', help="path to puzzle file")

    args = parser.parse_args()
    main(args)
