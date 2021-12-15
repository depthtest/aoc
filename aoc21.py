import re
from heapq import heapify, heappush, heappop

def day1p1():
    inc, prev = 0, 0
    with open('input1', 'r') as opfile:
        for i, line in enumerate(opfile):
            if i > 0 and int(line)>prev:
                inc += 1
            prev=int(line)
    print(inc)

def day1p2():
    inc = 0
    with open('input1', 'r') as opfile:
        lines = list(map(lambda x: int(x), opfile.readlines()))
        for i in range(1, len(lines)-2):
            if (lines[i]+lines[i+1]+lines[i+2]) > (lines[i-1]+lines[i]+lines[i+1]):
                inc += 1
    print(inc)

def day2p1():
    x, d = 0, 0
    with open('input2', 'r') as opfile:
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
    with open('input2', 'r') as opfile:
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
    with open('input3','r') as opfile:
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
    with open('input3','r') as opfile:
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
    with open('input4', 'r') as opfile:
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
    with open('input4', 'r') as opfile:
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
    with open('input5', 'r') as opfile:
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
    with open('input5', 'r') as opfile:
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
    with open('input6', 'r') as opfile:
        fishes = list(map(lambda x: int(x), opfile.readline().strip().split(',')))
    for _ in range(80):
        fishes = list(map(lambda x: x-1, fishes))
        num_new = len(list(filter(lambda x: x==-1, fishes)))
        fishes = list(map(lambda x: 6 if x==-1 else x, fishes))
        fishes.extend([8]*num_new)
    print(len(fishes))

def day6p2():
    with open('input6', 'r') as opfile:
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

def day7p1():
    with open('input7','r') as opfile:
        crabs = list(map(lambda x: int(x), opfile.readline().strip().split(',')))
    median = sorted(crabs)[len(crabs)//2]
    acc = 0
    for cr in crabs:
        acc += abs(cr-median)
    print(acc)

def day7p2():
    with open('input7','r') as opfile:
        crabs = list(map(lambda x: int(x), opfile.readline().strip().split(',')))
    mean = int(sum(crabs)/len(crabs))
    acc = 0
    for cr in crabs:
        dist = abs(cr-mean) + 1
        acc += sum([i for i in range(dist)])
    print(acc)

def day8p1():
    acc = 0
    with open('input8', 'r') as opfile:
        for line in opfile:
            outputs = line.strip().split(' | ')[1]

            for dig in outputs.split():
                if len(dig) < 5 or len(dig) == 7:
                    acc += 1
    print(acc)

def day8p2():
    def deduce(signals):
        # SEGMENTS
        #  aaaa
        # b    c
        # b    c
        #  dddd
        # e    f
        # e    f
        #  gggg
        signs = [set(ss) for ss in signals]
        segments = {}

        numbers = [-1] * 10
        numbers[1] = list(filter(lambda x: len(x)==2, signs))[0]
        numbers[4] = list(filter(lambda x: len(x)==4, signs))[0]
        numbers[7] = list(filter(lambda x: len(x)==3, signs))[0]
        numbers[8] = list(filter(lambda x: len(x)==7, signs))[0]

        signs.remove(numbers[1])
        signs.remove(numbers[4])
        signs.remove(numbers[7])
        signs.remove(numbers[8])

        found = False
        for signal in filter(lambda x: len(x)==6, signs):
            for el in numbers[1]:
                if el not in signal:
                    numbers[6] = signal
                    segments['c'] = el
                    found = True
                    break
            if found: break
        segments['f'] = list(filter(lambda x: x != segments['c'], numbers[1]))[0]
        signs.remove(numbers[6])

        for signal in filter(lambda x: len(x)==5, signs):
            five_lacking = list(filter(lambda x: x not in signal, numbers[6]))
            if len(five_lacking) == 1:
                segments['e'] = five_lacking[0]
                numbers[5] = signal
        signs.remove(numbers[5])

        for signal in signs:
            if segments['e'] in signal:
                if len(signal) == 5:
                    numbers[2] = signal
                if len(signal) == 6:
                    numbers[0] = signal
            if segments['f'] in signal and len(signal) == 5:
                numbers[3] = signal
        signs.remove(numbers[2])
        signs.remove(numbers[3])
        signs.remove(numbers[0])

        numbers[9] = signs[0]

        ret_dict = {}
        for idx, signal in enumerate(numbers):
            ret_dict[frozenset(signal)] = str(idx)
        return ret_dict

    def decode(outputs, segments):
        ret = ''
        for out in outputs:
            ret += segments[frozenset(out)]
        return int(ret)
    acc = 0
    with open('input8', 'r') as opfile:
        for line in opfile:
            linespl = line.split()
            ins = linespl[:10]
            outs = linespl[11:]
            segments = deduce(ins)
            acc += decode(outs, segments)
    print(acc)

def day9p1():
    with open('input9','r') as opfile:
        lines = list(map(lambda x: x.strip(), opfile.readlines()))
    acc = 0
    for idx, line in enumerate(lines):
        for jdx, ch in enumerate(line):
            isMin = True
            if idx > 0:
                isMin &= ch < lines[idx-1][jdx]
            if idx < (len(lines)-1):
                isMin &= ch < lines[idx+1][jdx]
            if jdx > 0:
                isMin &= ch < lines[idx][jdx-1]
            if jdx < (len(line)-1):
                isMin &= ch < lines[idx][jdx+1]
            if isMin:
                acc += 1 + int(ch)
    print(acc)

def day9p2():
    def growBasin(lines, i, j):
        visited = {}
        queue = [(i, j)]
        while queue:
            proc = queue.pop()
            if proc in visited: continue
            a, b = proc
            if a > 0 and lines[a-1][b] < '9':
                    queue.append((a-1, b))
            if a < len(lines)-1 and lines[a+1][b] < '9':
                    queue.append((a+1, b))
            if b > 0 and lines[a][b-1] < '9':
                    queue.append((a, b-1))
            if b < len(lines[0])-1 and lines[a][b+1] < '9':
                    queue.append((a, b+1))
            visited[proc] = lines[a][b]
        return len(visited)

    with open('input9','r') as opfile:
        lines = list(map(lambda x: x.strip(), opfile.readlines()))
    basins = []
    for idx, line in enumerate(lines):
        for jdx, ch in enumerate(line):
            isMin = True
            if idx > 0:
                isMin &= ch < lines[idx-1][jdx]
            if idx < (len(lines)-1):
                isMin &= ch < lines[idx+1][jdx]
            if jdx > 0:
                isMin &= ch < lines[idx][jdx-1]
            if jdx < (len(line)-1):
                isMin &= ch < lines[idx][jdx+1]
            if isMin:
                basins.append(growBasin(lines, idx, jdx))
    largest_basins = sorted(basins, reverse=True)[:3]
    acc = 1
    for bas in largest_basins:
        acc *= bas
    print(acc)

def day10p1():
    openers = {'(':')', '[':']', '{':'}', '<':'>'}
    closers_points = {')':3, ']':57, '}':1197, '>':25137}
    acc_points = 0
    with open('input10', 'r') as opfile:
        for line in opfile:
            check = []
            line = line.strip()
            for ch in line:
                if ch in openers:
                    check.append(ch)
                if ch in closers_points:
                    popped = check.pop()
                    if openers[popped] != ch:
                        #corrupted line
                        acc_points += closers_points[ch]
    print(acc_points)

def day10p2():
    openers = {'(':')', '[':']', '{':'}', '<':'>'}
    closers_points = {')':1, ']':2, '}':3, '>':4}
    scores = []
    with open('input10', 'r') as opfile:
        for line in opfile:
            check = []
            line = line.strip()
            corrupted = False
            for ch in line:
                if ch in openers:
                    check.append(ch)
                if ch in closers_points:
                    popped = check.pop()
                    if openers[popped] != ch:
                        corrupted= True
                        break
            if corrupted: continue

            line_score = 0
            while check:
                popped = check.pop()
                line_score = line_score*5 + closers_points[openers[popped]]
            scores.append(line_score)
    print(sorted(scores)[len(scores)//2])

def day11p1():
    with open('input11', 'r') as opfile:
        grid = [[int(ch) for ch in line.strip()] for line in opfile]

        total_flash = 0
        for step in range(100):

            for i in range(len(grid)):
                for j in range(len(grid[i])):
                    grid[i][j] = grid[i][j] + 1

            while True:
                flash_this_loop = 0

                for i in range(len(grid)):
                    for j in range(len(grid[i])):
                        if grid[i][j] > 9:
                            grid[i][j] = 0
                            flash_this_loop += 1

                            for x in range(-1,2):
                                for y in range(-1,2):
                                    if 0 <= i+x < len(grid) and 0 <= j+y < len(grid[i]):
                                        if grid[i+x][j+y] > 0:
                                            grid[i+x][j+y] += 1

                total_flash += flash_this_loop
                if flash_this_loop == 0:
                    break

    print(total_flash)

def day11p2():
    with open('input11', 'r') as opfile:
        grid = [[int(ch) for ch in line.strip()] for line in opfile]

        step = 0
        while True:
            step += 1

            for i in range(len(grid)):
                for j in range(len(grid[i])):
                    grid[i][j] = grid[i][j] + 1

            flash_this_step = 0
            while True:
                flash_this_loop = 0

                for i in range(len(grid)):
                    for j in range(len(grid[i])):
                        if grid[i][j] > 9:
                            grid[i][j] = 0
                            flash_this_loop += 1

                            for x in range(-1,2):
                                for y in range(-1,2):
                                    if 0 <= i+x < len(grid) and 0 <= j+y < len(grid[i]):
                                        if grid[i+x][j+y] > 0:
                                            grid[i+x][j+y] += 1

                flash_this_step += flash_this_loop
                if flash_this_loop == 0:
                    break

            if flash_this_step == len(grid) * len(grid[0]):
                print(step)
                return

class Day12:
    class Graph:
        def __init__(self, edges):
            self._adjacent = {}
            for v_a, v_b in edges:
                if v_a not in self._adjacent: self._adjacent[v_a] = set()
                if v_b not in self._adjacent: self._adjacent[v_b] = set()
                self._adjacent[v_a].add(v_b)
                self._adjacent[v_b].add(v_a)
        def get(self, vert):
            return self._adjacent[vert]

    def DFS_allpaths(graph, start, goal):
        paths = []
        this_path = [start]

        def rec_dfs(graph, goal):
            if this_path[-1] == goal:
                paths.append(this_path)
                return
            for child in graph.get(this_path[-1]):
                if child.isupper() or (child.islower() and child not in this_path):
                    this_path.append(child)
                    rec_dfs(graph, goal)
                    this_path.pop()

        rec_dfs(graph, goal)
        return paths

    def DFS_allpaths_v2(graph, start, goal):
        paths = []
        this_path = [start]

        def rec_dfs(graph, goal, one_small_rep_allowed=True):
            if this_path[-1] == goal:
                paths.append(this_path)
                return
            for child in graph.get(this_path[-1]):
                if child.isupper() or (child.islower() and (child not in this_path or one_small_rep_allowed)):
                    if child == 'start': continue
                    lower_and_isinpath = not(child.islower() and child in this_path)
                    this_path.append(child)
                    rec_dfs(graph, goal, one_small_rep_allowed and lower_and_isinpath)
                    this_path.pop()

        rec_dfs(graph, goal)
        return paths

def day12p1():
    with open('input12', 'r') as opfile:
        edg = list(map(lambda x: tuple(x.strip().split('-')), opfile.readlines()))
        graph = Day12.Graph(edg)

    paths = Day12.DFS_allpaths(graph, 'start', 'end')
    print(len(paths))

def day12p2():
    with open('input12', 'r') as opfile:
        edg = list(map(lambda x: tuple(x.strip().split('-')), opfile.readlines()))
        graph = Day12.Graph(edg)

    paths = Day12.DFS_allpaths_v2(graph, 'start', 'end')
    print(len(paths))

def day13p1():
    points = set()
    folds = []

    point_rex = re.compile(r'(?P<xcoord>\d+),(?P<ycoord>\d+)')
    fold_rex = re.compile(r'fold along (?P<coord>x|y)=(?P<howmuch>\d+)')
    with open('input13', 'r') as opfile:
        for line in opfile:
            if match := point_rex.match(line):
                points.add((int(match.groups()[0]), int(match.groups()[1])))
            elif match := fold_rex.match(line):
                folds.append((match.groups()[0], int(match.groups()[1])))

    folded = set(points)
    for idx, fold in enumerate(folds):
        if idx > 0: break

        fold_coord = 0 if fold[0] == 'x' else 1
        tomirror = list(filter(lambda x: x[fold_coord]>fold[1], folded))

        for p in tomirror:
            folded.remove(p)
            mirror_point = (
                (p[0] - 2*(p[0]-fold[1])) if fold_coord == 0 else p[0],
                (p[1] - 2*(p[1]-fold[1])) if fold_coord == 1 else p[1],
            )
            folded.add(mirror_point)
    print(len(folded))

def day13p2():
    points = set()
    folds = []

    point_rex = re.compile(r'(?P<xcoord>\d+),(?P<ycoord>\d+)')
    fold_rex = re.compile(r'fold along (?P<coord>x|y)=(?P<howmuch>\d+)')
    with open('input13', 'r') as opfile:
        for line in opfile:
            if match := point_rex.match(line):
                points.add((int(match.groups()[0]), int(match.groups()[1])))
            elif match := fold_rex.match(line):
                folds.append((match.groups()[0], int(match.groups()[1])))

    folded = set(points)
    for fold in folds:
        fold_coord = 0 if fold[0] == 'x' else 1
        tomirror = list(filter(lambda x: x[fold_coord]>fold[1], folded))

        for p in tomirror:
            folded.remove(p)
            mirror_point = (
                (p[0] - 2*(p[0]-fold[1])) if fold_coord == 0 else p[0],
                (p[1] - 2*(p[1]-fold[1])) if fold_coord == 1 else p[1],
            )
            folded.add(mirror_point)

    max_x = max(map(lambda x: x[0], folded))
    max_y = max(map(lambda x: x[1], folded))

    for j in range(max_y+1):
        str_to_scrn = ''
        for i in range(max_x+1):
            if (i,j) in folded:
                str_to_scrn += '#'
            else:
                str_to_scrn += '.'
        print(str_to_scrn)

def day14p1():
    template_rex = re.compile(r'^(?P<polymer>[A-Z]+)$')
    pair_ins_rex = re.compile(r'(?P<pair>[A-Z]+) -> (?P<ins>[A-Z])')
    with open('input14', 'r') as opfile:
        pairs = {}
        for line in opfile:
            if matches := template_rex.match(line):
                template = matches.groups()[0]
            elif matches := pair_ins_rex.match(line):
                pairs[matches.groups()[0]] = matches.groups()[1]

    cnts_idx = {}
    cnts = []

    def append(tmp, newchar):
        if newchar not in cnts_idx:
            cnts_idx[newchar] = len(cnts)
            cnts.append(1)
        else: cnts[cnts_idx[newchar]] += 1
        return tmp + newchar

    for _ in range(10):
        newtmp = ''
        cnts_idx.clear()
        cnts.clear()
        for idx in range(len(template)):
            if idx < len(template):
                newtmp = append(newtmp, template[idx])
            if template[idx:idx+2] in pairs:
                newtmp = append(newtmp, pairs[template[idx:idx+2]])
        template = newtmp

    max_char_cnt = max(cnts)
    min_char_cnt = min(cnts)
    print(max_char_cnt - min_char_cnt)

def day14p2():
    template_rex = re.compile(r'^(?P<polymer>[A-Z]+)$')
    pair_ins_rex = re.compile(r'(?P<pair>[A-Z]+) -> (?P<ins>[A-Z])')
    with open('input14', 'r') as opfile:
        pairs = {}
        for line in opfile:
            if matches := template_rex.match(line):
                template = matches.groups()[0]
            elif matches := pair_ins_rex.match(line):
                pairs[matches.groups()[0]] = matches.groups()[1]

    def add_to_cnts(mapcnt, elem, val=1):
        if elem not in mapcnt: mapcnt[elem] = val
        else: mapcnt[elem] += val

    cnts = {}
    for idx in range(len(template)-1):
        pair = template[idx:idx+2]
        add_to_cnts(cnts, pair)

    for _ in range(40):
        this_step_map = {}
        for pair in cnts:
            if pair in pairs:
                add_to_cnts(this_step_map, pair[0]+pairs[pair], cnts[pair])
                add_to_cnts(this_step_map, pairs[pair]+pair[1], cnts[pair])
        cnts = this_step_map

    letter_cnt = {}
    for k, v in cnts.items():
        add_to_cnts(letter_cnt, k[1], v)
    letter_cnt[template[0]] += 1

    max_ch = max(letter_cnt.items(), key=lambda x:x[1])
    min_ch = min(letter_cnt.items(), key=lambda x:x[1])

    print(max_ch[1] - min_ch[1])

def day15p1():
    with open('input15', 'r') as opfile:
        matrix = [list(map(lambda x: int(x), line.strip())) for line in opfile.readlines()]

    acc_matrix = [[1000000 for m_row_col in m_row] for m_row in matrix]
    acc_matrix[0][0] = 0

    neighs = [(1,0), (0,1), (-1, 0), (0, -1)]
    def neigh(curr, add):
        return tuple(map(lambda x: x[0]+x[1], zip(curr, add)))
    def valid_neigh(pos):
        return (-1 < pos[0] < len(matrix)) and (-1 < pos[1] < len(matrix[0]))

    queue = [(0, (0,0))]
    heapify(queue)
    while queue:
        _, this_pos = heappop(queue)
        for ne in neighs:
            nxt = neigh(this_pos, ne)
            if valid_neigh(nxt):
                possible_cost = acc_matrix[this_pos[0]][this_pos[1]] + matrix[nxt[0]][nxt[1]]
                actual_cost = acc_matrix[nxt[0]][nxt[1]]
                if possible_cost < actual_cost:
                    acc_matrix[nxt[0]][nxt[1]] = possible_cost
                    heappush(queue, (possible_cost, nxt))

    print(acc_matrix[-1][-1])

def day15p2():    
    with open('input15', 'r') as opfile:
        matrix = [list(map(lambda x: int(x), line.strip())) for line in opfile.readlines()]

    real_matrix = [[-1 for _ in range(len(matrix[0])*5)] for _ in range(len(matrix)*5)]
    for j5 in range(5):
        for i5 in range(5):
            for j in range(len(matrix)):
                for i in range(len(matrix[j])):
                    real_matrix[j5*len(matrix) + j][i5*len(matrix[j]) + i] = (matrix[j][i] + i5 + j5 - 1) % 9 + 1

    acc_matrix = [[1000000 for _ in real_matrix[0]] for _ in real_matrix]
    acc_matrix[0][0] = 0

    neighs = [(1,0), (0,1), (-1, 0), (0, -1)]
    def neigh(curr, add):
        return tuple(map(lambda x: x[0]+x[1], zip(curr, add)))
    def valid_neigh(pos):
        return (-1 < pos[0] < len(real_matrix)) and (-1 < pos[1] < len(real_matrix[0]))

    queue = [(0, (0,0))]
    heapify(queue)
    while queue:
        _, this_pos = heappop(queue)
        for ne in neighs:
            nxt = neigh(this_pos, ne)
            if valid_neigh(nxt):
                possible_cost = acc_matrix[this_pos[0]][this_pos[1]] + real_matrix[nxt[0]][nxt[1]]
                actual_cost = acc_matrix[nxt[0]][nxt[1]]
                if possible_cost < actual_cost:
                    acc_matrix[nxt[0]][nxt[1]] = possible_cost
                    heappush(queue, (possible_cost, nxt))

    print(acc_matrix[-1][-1])

import sys
eval('day' + sys.argv[1] + '()')
