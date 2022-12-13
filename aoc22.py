import math
import re
from heapq import heapify, heappush, heappop
from pprint import pprint

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

def parse_day8():
    with open('input') as ff:
        lines = []
        for line in ff:
            lines.append(list(map(lambda x: int(x), line.strip())))
    return lines
def day8p1():
    def isvisible(mm, i, j):
        isVisible_north = True
        isVisible_south = True
        isVisible_east = True
        isVisible_west = True
        #north
        for i2 in range(i):
            if mm[i2][j] >= mm[i][j]:
                isVisible_north = False
                break
        #south
        for i2 in range(len(mm)-1, i, -1):
            if mm[i2][j] >= mm[i][j]:
                isVisible_south = False
                break
        #west
        for j2 in range(j):
            if mm[i][j2] >= mm[i][j]:
                isVisible_west = False
                break
        #east
        for j2 in range(len(mm[0])-1, j, -1):
            if mm[i][j2] >= mm[i][j]:
                isVisible_east = False
                break
        return isVisible_north or isVisible_south or isVisible_east or isVisible_west
    mat = parse_day8()
    accum = (len(mat[0]) * 2) + (len(mat) - 2) * 2
    for i, row in enumerate(mat[1:-1], 1):
        for j, col in enumerate(row[1:-1], 1):
            if isvisible(mat, i, j):
                accum += 1
    print(accum)
def day8p2():
    def scenic_score(mm, i, j):
        score_north = 0
        score_south = 0
        score_west = 0
        score_east = 0
        #north
        for i2 in range(i-1, -1, -1):
            score_north += 1
            if mm[i2][j] >= mm[i][j]:
                break
        #south
        for i2 in range(i+1, len(mm), 1):
            score_south += 1
            if mm[i2][j] >= mm[i][j]:
                break
        #west
        for j2 in range(j-1, -1, -1):
            score_west += 1
            if mm[i][j2] >= mm[i][j]:
                break
        #east
        for j2 in range(j+1, len(mm[0]), 1):
            score_east += 1
            if mm[i][j2] >= mm[i][j]:
                break
        return score_north * score_south * score_west * score_east
    mat = parse_day8()
    max_score = 0
    for i, row in enumerate(mat):
        for j, col in enumerate(row):
            ij_score = scenic_score(mat, i, j)
            if ij_score > max_score:
                max_score = ij_score
    print(max_score)

def parse_day9():
    moves = []
    with open('input') as ff:
        for line in ff:
            matches = re.match(r'^(?P<move>U|D|R|L) (?P<step>\d+)$', line)
            moves.append((matches.group('move'), int(matches.group('step'))))
    return moves
def day9p1():
    def is_touching(pos1, pos2):
        if (pos1[0]-pos2[0])**2 > 1 or (pos1[1]-pos2[1])**2 > 1:
            return False
        return True
    moves = parse_day9()
    head_pos = (0,0)
    tail_pos = (0,0)
    tail_visited = set(tail_pos)
    for move in moves:
        for step in range(move[1]):
            if move[0] == 'U':
                head_pos = (head_pos[0], head_pos[1]+1)
            elif move[0] == 'D':
                head_pos = (head_pos[0], head_pos[1]-1)
            elif move[0] == 'L':
                head_pos = (head_pos[0]-1, head_pos[1])
            elif move[0] == 'R':
                head_pos = (head_pos[0]+1, head_pos[1])
            if not is_touching(head_pos, tail_pos):
                if tail_pos[0] == head_pos[0]:
                    tail_pos = (tail_pos[0], tail_pos[1] + (head_pos[1]-tail_pos[1])//2)
                elif tail_pos[1] == head_pos[1]:
                    tail_pos = (tail_pos[0] + (head_pos[0]-tail_pos[0])//2, tail_pos[1])
                else:
                    tail_pos = (
                        tail_pos[0] + (head_pos[0]-tail_pos[0]) / abs(head_pos[0]-tail_pos[0]),
                        tail_pos[1] + (head_pos[1]-tail_pos[1]) / abs(head_pos[1]-tail_pos[1])
                    )
                tail_visited.add(tail_pos)
    print(len(tail_visited))
def day9p2():
    def is_touching(pos1, pos2):
        if (pos1[0]-pos2[0])**2 > 1 or (pos1[1]-pos2[1])**2 > 1:
            return False
        return True
    moves = parse_day9()
    poses = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
    tail_visited = set(poses[-1])
    for move in moves:
        for step in range(move[1]):
            if move[0] == 'U':
                poses[0] = (poses[0][0], poses[0][1]+1)
            elif move[0] == 'D':
                poses[0] = (poses[0][0], poses[0][1]-1)
            elif move[0] == 'L':
                poses[0] = (poses[0][0]-1, poses[0][1])
            elif move[0] == 'R':
                poses[0] = (poses[0][0]+1, poses[0][1])
            for idx, _ in enumerate(poses[1:], 1):
                if not is_touching(poses[idx-1], poses[idx]):
                    if poses[idx][0] == poses[idx-1][0]:
                        poses[idx] = (poses[idx][0], poses[idx][1] + (poses[idx-1][1]-poses[idx][1])//2)
                    elif poses[idx][1] == poses[idx-1][1]:
                        poses[idx] = (poses[idx][0] + (poses[idx-1][0]-poses[idx][0])//2, poses[idx][1])
                    else:
                        poses[idx] = (
                            poses[idx][0] + (poses[idx-1][0]-poses[idx][0]) / abs(poses[idx-1][0]-poses[idx][0]),
                            poses[idx][1] + (poses[idx-1][1]-poses[idx][1]) / abs(poses[idx-1][1]-poses[idx][1])
                        )
                    if idx==9:
                        tail_visited.add(poses[idx])
    print(len(tail_visited))

def parse_day10():
    lines = []
    with open('input') as ff:
        for line in ff:
            if line.startswith('noop'):
                lines.append(('noop',0))
            else:
                matches = re.match(r'^addx (?P<disp>-?\d+)$', line.strip())
                lines.append(('addx', int(matches.group('disp'))))
    return lines
def day10p1():
    lines = parse_day10()
    acc_cycles = [1]
    acc_regs = [1]
    for line in lines:
        if line[0] == 'noop':
            acc_cycles.append(acc_cycles[-1]+1)
        elif line[0] == 'addx':
            acc_cycles.append(acc_cycles[-1]+2)
        acc_regs.append(acc_regs[-1] + line[1])
    cycle_search = [20,60,100,140,180,220]
    accum = 0
    for c in cycle_search:
        id, _ = list(filter(lambda x: x[1]>=c, enumerate(acc_cycles)))[0]
        accum += c * acc_regs[id]
    print(accum)
def day10p2():
    lines = parse_day10()
    acc_cycles = [1]
    acc_regs = [1]
    for line in lines:
        if line[0] == 'noop':
            acc_cycles.append(acc_cycles[-1]+1)
        elif line[0] == 'addx':
            acc_cycles.append(acc_cycles[-1]+2)
        acc_regs.append(acc_regs[-1] + line[1])
    c = 1
    stri = ''
    while c < acc_cycles[-1]:
        id, _ = list(filter(lambda x: x[1]>=c, enumerate(acc_cycles)))[0]
        if c % 40 in [acc_regs[id]-1, acc_regs[id], acc_regs[id]+1]:
            stri += '#'
        else:
            stri += '.'
        if c % 40 == 0:
            print(stri)
            stri = ''
        c += 1

def parse_day11():
    monkey = re.compile(r'^Monkey (?P<idmon>\d+):$')
    st_itm = re.compile(r'\s*Starting items: (?P<items>.+)$')
    operat = re.compile(r'\s*Operation: (?P<where>[a-z]+) = (?P<op1>[a-z]+) (?P<op>\+|\*|) (?P<op2>([a-z]+)|\d+)')
    testop = re.compile(r'\s*Test: divisible by (?P<div>\d+)')
    iftrue = re.compile(r'\s*If true: throw to monkey (?P<tomonkey>\d+)')
    iffals = re.compile(r'\s*If false: throw to monkey (?P<tomonkey>\d+)')
    monkeys = []
    curr_monkey = []
    with open('input') as ff:
        for line in ff:
            if monkey.match(line):
                curr_monkey = []
                continue
            mm = st_itm.match(line)
            if mm: 
                curr_monkey.append(list(map(lambda x:int(x), mm.group('items').split(','))))
                continue
            mm = operat.match(line)
            if mm: 
                curr_monkey.append((
                    mm.group('where'),
                    mm.group('op1'),
                    mm.group('op'),
                    mm.group('op2'),
                ))
                continue
            mm = testop.match(line)
            if mm: 
                curr_monkey.append(int(mm.group('div')))
                continue
            mm = iftrue.match(line)
            if mm: 
                curr_monkey.append(int(mm.group('tomonkey')))
                continue
            mm = iffals.match(line)
            if mm: 
                curr_monkey.append(int(mm.group('tomonkey')))
                continue
            monkeys.append(curr_monkey)
            #curr_monkey = []
        monkeys.append(curr_monkey)
    return monkeys
def day11p1():
    monkeys = parse_day11()
    rounds = 20
    inspections = [0 for _ in monkeys]
    for r in range(rounds):
        for idx, _ in enumerate(monkeys):
            while monkeys[idx][0]:
                curr_worry = monkeys[idx][0].pop(0)
                howmuch = curr_worry if monkeys[idx][1][-1] == 'old' else int(monkeys[idx][1][-1])
                
                if monkeys[idx][1][-2] == '*':
                    curr_worry *= howmuch
                elif monkeys[idx][1][-2] == '+':
                    curr_worry += howmuch
                else: raise ValueError()
                
                curr_worry = curr_worry // 3
                if curr_worry % monkeys[idx][2] == 0:
                    monkeys[monkeys[idx][3]][0].append(curr_worry)
                else:
                    monkeys[monkeys[idx][4]][0].append(curr_worry)
                inspections[idx] += 1
    insp_sort = sorted(inspections)
    print(insp_sort[-1]*insp_sort[-2])
def day11p2():
    monkeys = parse_day11()
    rounds = 10000
    acc = 1
    for i in monkeys:
        acc *= i[2]
    inspections = [0 for _ in monkeys]
    for r in range(rounds):
        for idx, _ in enumerate(monkeys):
            while monkeys[idx][0]:
                curr_worry = monkeys[idx][0].pop(0)
                howmuch = curr_worry if monkeys[idx][1][-1] == 'old' else int(monkeys[idx][1][-1])
                
                if monkeys[idx][1][-2] == '*':
                    curr_worry *= howmuch
                elif monkeys[idx][1][-2] == '+':
                    curr_worry += howmuch
                else: raise ValueError()
                
                curr_worry = curr_worry % acc
                if curr_worry % monkeys[idx][2] == 0:
                    monkeys[monkeys[idx][3]][0].append(curr_worry)
                else:
                    monkeys[monkeys[idx][4]][0].append(curr_worry)
                inspections[idx] += 1
    insp_sort = sorted(inspections)
    print(insp_sort[-1]*insp_sort[-2])

def parse_day12():
    maze = []
    with open('input') as ff:
        maze = list(map(lambda x: x.strip(), ff.readlines()))
    start = None
    end = None
    for i, line in enumerate(maze):
        s_pos = line.find('S')
        if s_pos != -1: start = (i, s_pos)
        e_pos = line.find('E')
        if e_pos != -1: end = (i, e_pos)
    maze[start[0]] = maze[start[0]].replace('S','a')
    maze[end[0]] = maze[end[0]].replace('E', 'z')
    return maze, start, end
def day12p1():
    maze, start, end = parse_day12()
    def isvalid(maze, pos, curr):
        if (0 <= pos[0] < len(maze)) and (0 <= pos[1] < len(maze[pos[0]])) and (ord(maze[pos[0]][pos[1]])-ord(maze[curr[0]][curr[1]]))<2:
            return True
        return False
    def add_disp(pos, neig):
        return (pos[0]+neig[0], pos[1]+neig[1])
    neighs = [(1,0),(-1,0),(0,1),(0,-1)]
    queue = [(start, [])]
    visited = {}
    path = None
    while queue and not path:
        curr = queue.pop(0)
        visited[curr[0]] = 'V'
        for nn in neighs:
            new_pos = add_disp(curr[0], nn)
            if isvalid(maze, new_pos, curr[0]) and new_pos not in visited:
                visited[new_pos] = 'F'
                if new_pos == end:
                    path = curr[1] + [curr[0] + new_pos]
                    break
                queue.append((new_pos, curr[1]+[curr[0]]))
    print(len(path))
def day12p2():
    maze, _, end = parse_day12()
    def isvalid(maze, pos, curr):
        if (0 <= pos[0] < len(maze)) and (0 <= pos[1] < len(maze[pos[0]])) and (ord(maze[curr[0]][curr[1]])-ord(maze[pos[0]][pos[1]]))<2:
            return True
        return False
    def add_disp(pos, neig):
        return (pos[0]+neig[0], pos[1]+neig[1])
    neighs = [(1,0),(-1,0),(0,1),(0,-1)]
    queue = [(end, [])]
    visited = {}
    path = None
    while queue and not path:
        curr = queue.pop(0)
        visited[curr[0]] = 'V'
        for nn in neighs:
            new_pos = add_disp(curr[0], nn)
            if isvalid(maze, new_pos, curr[0]) and new_pos not in visited:
                visited[new_pos] = 'F'
                if maze[new_pos[0]][new_pos[1]] == 'a':
                    path = curr[1] + [curr[0] + new_pos]
                    break
                queue.append((new_pos, curr[1]+[curr[0]]))
    print(len(path))

import sys
eval('day' + sys.argv[1] + '()')