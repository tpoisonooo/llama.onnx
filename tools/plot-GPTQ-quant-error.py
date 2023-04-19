import matplotlib.pyplot as plt
import numpy as np
import sys
import os

print(len(sys.argv))
if len(sys.argv) < 2:
    print('usage {} log_from_GPTQ-for-llama'.format(sys.argv[0]))
    sys.exit(-1)

table = dict()

key = None
idx = -1
_type = None
with open(sys.argv[1]) as f:

    while True:
        line = f.readline()
        if not line:
            break

        if 'Quantizing' in line:
            key = line.split(' ')[1].split('.')[1]
            _type = 'append'

        if 'Optimizing' in line:
            idx = int(line.split(' ')[2])
            key = line.split(' ')[1].split('.')[1]
            _type = 'update'

        if ', error' in line:
            error = float(line.split(' ')[-1])

            if key not in table:
                table[key] = []

            if _type == 'append':
                table[key].append(error)
            else:
                table[key][idx] = error

            _type = None

color_map = {
    'k_proj': 'dimgray',
    'v_proj': 'lightcoral',
    'q_proj': 'chocolate',
    'o_proj': 'gold',
    'up_proj': 'olive',
    'gate_proj': 'cadetblue',
    'down_proj': 'darkviolet'
}

legends = []
for k, v in table.items():
    idxs = range(len(v))
    plt.plot(idxs, v, c=color_map[k])
    legends.append(k)

plt.title(os.path.basename(sys.argv[1]))
plt.xlabel('layerid')
plt.ylabel('error')
plt.legend(legends)

plt.show()