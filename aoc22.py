import math
import re
from heapq import heapify, heappush, heappop

def parse_day1():
    deers = []
    curr_deer = []
    with open('input', 'r') as ff:
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

import sys
eval('day' + sys.argv[1] + '()')