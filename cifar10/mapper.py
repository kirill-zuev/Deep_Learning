#!/nn/bin/python3

import sys
import os
import numpy as np

cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def get_values(s, ws, i1):
    ans = []
    for w in ws:
        op = ''.join(['<',w,'>'])
        cl = ''.join(['</',w,'>'])
        i0 = s.find(op,i1)
        i1 = s.find(cl,i0)
        sub = s[i0+len(op):i1]
        ans.append(sub)
    return ans, i1

def parse(s):
    fname, i1 = get_values(s,["filename"],0)
    fname = fname[0]

    i1 = s.find("<object>",i1)
    objs = []
    i0 = s.find("<name>",i1)
    while i0 > 0:
        i1 = s.find("</name>",i0)
        nm = s[i0+6:i1]
        objs.append(nm)
        i0 = s.find("<name>",i1)
    data = {'fname': fname,
            'objs': objs}
    return data

lines = []
for line in sys.stdin:
    lines.append(line)

s = ''.join(lines)

data = parse(s)
objs = []
for obj in data['objs']:
    if obj in cifar_classes:
        objs.append(obj)

for obj in objs:
    print(obj, data['fname'], sep='\t')
