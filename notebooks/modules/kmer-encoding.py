import itertools
nucleotides = "ATGC"

for i in range(5, 16):
    combinations = itertools.product(*itertools.repeat(nucleotides, i))
    for j in combinations:
        print(j)