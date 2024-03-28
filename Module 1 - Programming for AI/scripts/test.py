# def getr(specs):
#     if len(specs) == 1:
#         return [[spec] for spec in specs[0]]
#     else:
#         return [[x, *spec] for x in specs[0] for spec in getr(specs[1:])]
#
#
# print(getr([range(0, 3)]))
# print(getr([range(0, 3), range(3, 6)]))
# print(getr([range(0, 3), range(3, 6), range(6, 9)]))
# print(getr([(1,), range(3, 6), range(6, 9)]))

import traceback
import sys


def mefun(x):
    return len[x]

s = "test"
try:
    mefun(10)
except Exception as e:
    data = traceback.extract_tb(sys.exc_info()[2])
    print(data[-1][-1])
