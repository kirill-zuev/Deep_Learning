#!/nn/bin/python3

import sys

for line in sys.stdin:
    pred, number = line.split('\t')
    print("numbers of Nearest Neighbors:", pred, "image number", number)
