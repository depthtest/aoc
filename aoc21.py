def day1p1():
    inc, prev = 0, 0
    with open('res\\input1', 'r') as opfile:
        for i, line in enumerate(opfile):
            if i > 0 and int(line)>prev:
                inc += 1
            prev=int(line)
    print(inc)

def day1p2():
    inc = 0
    with open('res\\input1', 'r') as opfile:
        lines = list(map(lambda x: int(x), opfile.readlines()))
        for i in range(1, len(lines)-2):
            if (lines[i]+lines[i+1]+lines[i+2]) > (lines[i-1]+lines[i]+lines[i+1]):
                inc += 1
    print(inc)

def day2p1():
    x, d = 0, 0
    with open('res\\input2', 'r') as opfile:
        for line in opfile:
            ll = line.split()
            if ll[0] == 'forward':
                x += int(ll[1])
            elif ll[0] == 'up':
                d -= int(ll[1])
            elif ll[0] == 'down':
                d += int(ll[1])
    print(x, d, x*d)

def day2p2():
    x, d, aim = 0, 0, 0
    with open('res\\input2', 'r') as opfile:
        for line in opfile:
            ll = line.split()
            if ll[0] == 'forward':
                x += int(ll[1])
                d += aim*int(ll[1])
            elif ll[0] == 'up':
                aim -= int(ll[1])
            elif ll[0] == 'down':
                aim += int(ll[1])
    print(x, d, x*d)

def day3p1():
    with open('res\\input3','r') as opfile:
        onecount = []
        total_lines = 0
        for line in map(lambda x: x.split()[0], opfile):
            if len(onecount) < len(line): onecount = [0 for _ in line]
            for idx, elem in enumerate(line):
                if elem == '1':
                    onecount[idx] += 1
            total_lines += 1
    mcb = ''.join(['1' if onecount[i] > total_lines//2 else '0' for i in range(len(onecount))])
    lcb = ''.join(['1' if mcb_i=='0' else '0' for mcb_i in mcb])

    mcb = int(mcb, 2)
    lcb = int(lcb, 2)
    
    print(mcb, lcb, mcb*lcb)

def day3p2():
    with open('res\\input3','r') as opfile:
        lines = list(map(lambda x: x.split()[0], opfile.readlines()))

    def getMostCommon(lines, idx):
        onecount = 0
        for line in lines:
            if line[idx] == '1':
                onecount += 1
        return '1' if onecount >= len(lines)/2 else '0'
    def getLeastCommon(lines, idx):
        zerocount = 0
        for line in lines:
            if line[idx] == '0':
                zerocount += 1
        return '0' if zerocount <= len(lines)/2 else '1'

    mcb_filtered = lines
    lcb_filtered = lines

    for idx, _ in enumerate(lines[0]):
        mcb = getMostCommon(mcb_filtered, idx)
        lcb = getLeastCommon(lcb_filtered, idx)
        if len(mcb_filtered) > 1:
            mcb_filtered = list(filter(lambda x: x[idx]==mcb, mcb_filtered))
        if len(lcb_filtered) > 1:
            lcb_filtered = list(filter(lambda x: x[idx]==lcb, lcb_filtered))

    mcb_filtered = int(mcb_filtered[0], 2)
    lcb_filtered = int(lcb_filtered[0], 2)

    print(mcb_filtered, lcb_filtered, mcb_filtered*lcb_filtered)

class Day4:
    class BingoMat:
        @staticmethod
        def readMat(fp):
            ret = Day4.BingoMat()
            ret._rows = []
            for _ in range(5):
                ret._rows.append(list(map(lambda x : int(x), fp.readline().split())))
            ret._vals = [len(ret._rows[0]) for _ in ret._rows]
            return ret
        def markVal(self, val):
            for idx, _ in enumerate(self._rows):
                for j in range(len(self._rows[idx])):
                    if val == self._rows[idx][j]:
                        self._rows[idx][j] = -1
                        self._vals[idx] -= 1
                        return
        def hasHor(self):
            for i in self._vals:
                if i == 0:
                    return True
            return False
        def hasVer(self):
            verts = [0 for _ in range(len(self._rows[0]))]
            for idx, _ in enumerate(self._rows):
                verts = list(map(lambda x: x[0]+x[1], zip(map(lambda x: x==-1, self._rows[idx]), verts)))
            for v in verts:
                if v==5:
                    return True
            return False
        def hasWon(self):
            return self.hasHor() or self.hasVer()
        def __repr__(self):
            return '\n'.join(map(lambda x: str(x), self._rows))
        def compVal(self):
            val = 0
            for r in self._rows:
                for v in r:
                    if v != -1:
                        val += v
            return val

def day4p1():
    with open('res\\input4', 'r') as opfile:
        num_list = list(map(lambda x : int(x), opfile.readline().split(',')))
        boards = []
        while True:
            emp_line = opfile.readline()
            if emp_line == '': break
            boards.append(Day4.BingoMat.readMat(opfile))

    for n in num_list:
        for _, b in enumerate(boards):
            b.markVal(n)
            if b.hasWon():
                print(b.compVal() * n)
                return

def day4p2():
    with open('res\\input4', 'r') as opfile:
        num_list = list(map(lambda x : int(x), opfile.readline().split(',')))
        boards = []
        while True:
            emp_line = opfile.readline()
            if emp_line == '': break
            boards.append(Day4.BingoMat.readMat(opfile))

    b_idx = set([i for i in range(len(boards))])
    for n in num_list:
        if len(b_idx) == 1:
            b = next(iter(b_idx))
            boards[b].markVal(n)
            if boards[b].hasWon():
                print(boards[b].compVal() * n)
                return
        else:
            for idx, b in enumerate(boards):
                if idx not in b_idx: continue
                b.markVal(n)
                if b.hasWon(): b_idx.remove(idx)

class Day5:
    class Line:
        def __init__(self, x1, y1, x2, y2):
            self._x1 = x1
            self._y1 = y1
            self._x2 = x2
            self._y2 = y2
        def isVer(self):
            return self._x1 == self._x2
        def isHor(self):
            return self._y1 == self._y2
        def isDia(self):
            return abs(self._y1 - self._y2) == abs(self._x1 - self._x2)

def day5p1():
    points = {}
    with open('res\\input5', 'r') as opfile:
        for l in opfile:
            x1, x2 = list(map(lambda x: x.split(','), l.strip().split(' -> ')))
            line = Day5.Line(
                    int(x1[0]), int(x1[1]),
                    int(x2[0]), int(x2[1]),
            )
            if line.isVer():
                ini = min(line._y1, line._y2)
                fin = max(line._y1, line._y2)
                for i in range(ini, fin + 1):
                    p = (line._x1, i)
                    if p not in points: points[p] = 1
                    else: points[p] += 1
            elif line.isHor():
                ini = min(line._x1, line._x2)
                fin = max(line._x1, line._x2)
                for i in range(ini, fin + 1):
                    p = (i, line._y1)
                    if p not in points: points[p] = 1
                    else: points[p] += 1
    tot_points = 0
    for p, v in points.items():
        if v > 1:
            tot_points += 1
    print(tot_points)

def day5p2():
    points = {}
    with open('res\\input5', 'r') as opfile:
        for l in opfile:
            x1, x2 = list(map(lambda x: x.split(','), l.strip().split(' -> ')))
            line = Day5.Line(
                    int(x1[0]), int(x1[1]),
                    int(x2[0]), int(x2[1]),
            )
            if line.isVer():
                ini = min(line._y1, line._y2)
                fin = max(line._y1, line._y2)
                for i in range(ini, fin + 1):
                    p = (line._x1, i)
                    if p not in points: points[p] = 1
                    else: points[p] += 1
            elif line.isHor():
                ini = min(line._x1, line._x2)
                fin = max(line._x1, line._x2)
                for i in range(ini, fin + 1):
                    p = (i, line._y1)
                    if p not in points: points[p] = 1
                    else: points[p] += 1
            elif line.isDia():
                ini = (line._x1, line._y1) if line._x1 < line._x2 else (line._x2, line._y2)
                fin = (line._x1, line._y1) if line._x1 > line._x2 else (line._x2, line._y2)
                p = []
                for idx, a in enumerate(range(ini[0], fin[0]+1)):
                    b = ini[1] + idx * (1 if ini[1] < fin[1] else -1)
                    p.append((a,b))
                for i in p:
                    if i not in points: points[i] = 1
                    else: points[i] += 1
    tot_points = 0
    for p, v in points.items():
        if v > 1:
            tot_points += 1
    print(tot_points)

def day6p1():
    with open('res\\input6', 'r') as opfile:
        fishes = list(map(lambda x: int(x), opfile.readline().strip().split(',')))
    for _ in range(80):
        fishes = list(map(lambda x: x-1, fishes))
        num_new = len(list(filter(lambda x: x==-1, fishes)))
        fishes = list(map(lambda x: 6 if x==-1 else x, fishes))
        fishes.extend([8]*num_new)
    print(len(fishes))

def day6p2():
    with open('res\\input6', 'r') as opfile:
        fishes = list(map(lambda x: int(x), opfile.readline().strip().split(',')))
    per_day = []
    for timer in range(9):
        per_day.append(len(list(filter(lambda x: x==timer, fishes))))
    for _ in range(256):
        to_gen = per_day[0]
        for i in range(1,len(per_day)):
            per_day[i-1] = per_day[i]
        per_day[6] += to_gen
        per_day[8] = to_gen
    print(sum(per_day))

import sys
eval('day' + sys.argv[1] + '()')