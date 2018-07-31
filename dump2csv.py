import numpy as np
import re
import pandas as pd


filename = 'test.txt'

with open(filename) as f:
    data = f.readlines()

out_data = []
for line in data:
    match = re.search('curr: (\d+)', line)
    if match:
        temp = int(match.group(1))
        out_data.append(temp)

np_array = np.asarray(out_data)
df = pd.DataFrame(np_array)
df.to_csv("out_data.csv", header=False, index=False)




