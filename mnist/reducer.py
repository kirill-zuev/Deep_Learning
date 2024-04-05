#!/nn/bin/python3

import sys

for line in sys.stdin:
    predicted = line.split('\t')
    print("predicted:", predicted[0])
