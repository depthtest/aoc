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

import sys
eval('day' + sys.argv[1] + '()')