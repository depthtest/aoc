import math
import re
from heapq import heapify, heappush, heappop

def parse_day1():
    deers = []
    curr_deer = []
    with open('input.txt', 'r') as ff:
        for line in ff:
            if line == "\n":
                deers.append(curr_deer)
                curr_deer = []
                continue
            else:
                curr_deer.append(int(line))
        deers.append(curr_deer)
    return deers
def day1p1():
    deers = parse_day1()
    print(max(map(lambda x: sum(x), deers)))
def day1p2():
    deers = parse_day1()
    print(sum(list(sorted(map(lambda x: sum(x), deers))[-3:])))

def parse_day2():
    plays = []
    with open('input') as ff:
        for line in ff:
            other, me = line.strip().split(' ')
            plays.append((other, me))
    return plays
def day2p1():
    mapper = {
        'A':'R', 'B':'P', 'C':'S',
        'X':'R', 'Y':'P', 'Z':'S',
    }
    score_mapper = {'R': 1,'P': 2,'S': 3,}
    win = {('R','P'), ('P','S'), ('S', 'R')}
    los = {('R','S'), ('P','R'), ('S', 'P')}

    plays = parse_day2()
    acc_score = 0
    for play in map(lambda y: tuple(map(lambda x: mapper[x], y)), plays):
        if play in win:
            acc_score += 6 + score_mapper[play[1]]
        elif play in los:
            acc_score += 0 + score_mapper[play[1]]
        else:
            acc_score += 3 + score_mapper[play[1]]
    print(acc_score)
def day2p2():
    mapper = {
        'A':'R', 'B':'P', 'C':'S',
        'X':'L', 'Y':'T', 'Z':'W',
    }
    score_mapper = {'R': 1, 'P': 2, 'S': 3}
    win = {'R':'P', 'P':'S', 'S':'R'}
    los = {'R':'S', 'P':'R', 'S':'P'}

    plays = parse_day2()
    acc_score = 0
    for play in map(lambda y: tuple(map(lambda x: mapper[x], y)), plays):
        if play[1] == 'W':
            acc_score += 6 + score_mapper[win[play[0]]]
        elif play[1] == 'L':
            acc_score += 0 + score_mapper[los[play[0]]]
        else:
            acc_score += 3 + score_mapper[play[0]]
    print(acc_score)

def parse_day3():
    with open('input') as ff:
        lines = list(map(lambda x:x.strip(), ff.readlines()))
    return lines
def day3_score(letter):
    ilet = ord(letter)
    if ilet > 96:
        return ilet - 96
    else:
        return ilet - 64 + 26
def day3p1():
    lines = parse_day3()
    accum = 0
    for line in lines:
        accum += day3_score(next(iter(set(line[:len(line)//2]).intersection(set(line[len(line)//2:])))))
    print(accum)
def day3p2():
    lines = parse_day3()
    accum = 0
    for i in range(0, len(lines), 3):
        accum += day3_score(next(iter(set(lines[i]).intersection(set(lines[i+1])).intersection(set(lines[i+2])))))
    print(accum)

def parse_day4():
    ranges = []
    with open('input') as ff:
        for line in ff:
            ran = line.strip().split(',')
            ranges.append(list(map(lambda y: (int(y[0]), int(y[1])), map(lambda x: x.split('-'), ran))))
    return ranges
def day4p1():
    ranges = parse_day4()
    accum = 0
    for rans in ranges:
        if (rans[0][0] >= rans[1][0] and rans[0][-1] <= rans[1][-1]) or \
            (rans[0][0] <= rans[1][0] and rans[0][-1] >= rans[1][-1]):
            accum += 1
    print(accum)
def day4p2():
    ranges = parse_day4()
    accum = 0
    for rans in ranges:
        if (rans[0][0] <= rans[1][0] <= rans[0][1]) or \
            (rans[0][0] <= rans[1][1] <= rans[0][1]) or \
            (rans[1][0] <= rans[0][0] <= rans[1][1]) or \
            (rans[1][0] <= rans[0][1] <= rans[1][1]):
            accum += 1
    print(accum)

def parse_day5():
    cols = []
    moves = []
    with open('input') as ff:
        in_cols = True
        for line in ff:
            if line == '\n' or line.startswith(' 1'):
                in_cols = False
                continue
            if in_cols:
                for i in range(1, len(line), 4):
                    j = (i - 1) // 4
                    if len(cols) < (j+1): cols.append([])
                    if line[i] != ' ': cols[j].insert(0, line[i])
            else:
                matches = re.match(r'^move (?P<hm>\d+) from (?P<fr>\d+) to (?P<to>\d+)', line)
                if matches:
                    moves.append((
                        int(matches.group('hm')),
                        int(matches.group('fr')) - 1,
                        int(matches.group('to')) - 1
                    ))
    return cols, moves
def day5p1():
    cols, moves = parse_day5()
    for hm, fr, to in moves:
        cols[to].extend(reversed(cols[fr][-hm:]))
        cols[fr] = cols[fr][:-hm]
    print(''.join(map(lambda x: x[-1], cols)))
def day5p2():
    cols, moves = parse_day5()
    for hm, fr, to in moves:
        cols[to].extend(cols[fr][-hm:])
        cols[fr] = cols[fr][:-hm]
    print(''.join(map(lambda x: x[-1], cols)))

def day6p1():
    with open('input') as ff:
        line = ff.readlines()[0]
    for i in range(3, len(line)):
        prev = set(line[i-3:i])
        if len(prev) == 3 and line[i] not in prev:
            print(i+1)
            break
def day6p2():
    with open('input') as ff:
        line = ff.readlines()[0]
    for i in range(13, len(line)):
        prev = set(line[i-13:i])
        if len(prev) == 13 and line[i] not in prev:
            print(i+1)
            break

class TreeNode:
    def __init__(self, name, type, size=0):
        self._name = name
        self._type = type
        self._size = size
        self._parent = None
        self._children = []
    def append(self, child):
        assert self._type=='file'
        self._children.append(child)
        child._parent = self
        self._upd_size(child._size)
    def _upd_size(self, add_size):
        self._size += add_size
        if self._parent:
            self._parent._upd_size(add_size)
    def __repr__(self) -> str:
        return f'{"D" if self._type=="dir" else "F"} {self._name}'
def parse_day7():
    cd = re.compile(r'^\$ cd (?P<folder>\/|[a-z]+|(\.\.))$')
    ls = re.compile(r'^\$ ls$')
    dr = re.compile(r'^dir (?P<dir>[a-z]+)$')
    fl = re.compile(r'^(?P<size>[0-9]+) (?P<filename>.*)$')
    tree = TreeNode('/', 'dir')
    curr_node = None
    mode = 'cd'
    with open('input') as ff:
        for line in ff:
            matches_cd = cd.match(line)
            if matches_cd:
                mode = 'cd'
                to_where = matches_cd.group('folder')
                if to_where == '/':
                    curr_node = tree
                elif to_where == '..':
                    curr_node = curr_node._parent
                else:
                    curr_node = next(iter(filter(lambda x: x._name==to_where and x._type=='dir', curr_node._children)))
                continue
            matches_ls = ls.match(line)
            if matches_ls:
                mode = 'ls'
                continue
            matches_dr = dr.match(line)
            if matches_dr:
                assert mode=='ls'
                curr_node.append(TreeNode(matches_dr.group('dir'), type='dir'))
                continue
            matches_fl = fl.match(line)
            if matches_fl:
                assert mode=='ls'
                curr_node.append(TreeNode(matches_fl.group('filename'), type='file', size=int(matches_fl.group('size'))))
                continue
    return tree
def day7p1():
    tree = parse_day7()
    accum = 0
    queue = [tree]
    while queue:
        el = queue.pop(0)
        if el._size <= 100000:
            accum += el._size
        for ch in filter(lambda x: x._type=='dir', el._children):
            queue.append(ch)
    print(accum)
def day7p2():
    tree = parse_day7()
    total = 70000000
    requi = 30000000
    unuse = total - tree._size
    todel = requi - unuse
    min_folder = tree
    queue = [tree]
    while queue:
        el = queue.pop(0)
        if el._size >= todel and el._size < min_folder._size:
            min_folder = el
        for ch in filter(lambda x: x._type=='dir', el._children):
            queue.append(ch)
    print(min_folder._size)


import sys
eval('day' + sys.argv[1] + '()')