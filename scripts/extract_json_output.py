import json
import sys

infile = sys.argv[1]
outfile = sys.argv[2]
assert infile.endswith(".json")

lines = []
with open(infile, "r") as f:
    for line in f:
        lines.append(json.loads(line))


with open(outfile, 'w') as f:
    for line in lines:
        f.write(f"{line['prediction']}\n")
