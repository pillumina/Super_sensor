'''
Author: Calvin
How to use the code: add or delete the device directory names in 'devs' list
                    (make sure the corresponding directory really exists!)
Outputs: create .csv files in corresponding directories and create plots(.png) files in current working directory.
'''

import numpy as np
import pandas as pd
from numpy import genfromtxt
from scipy.fftpack import fft
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import re
import os
from os.path import isfile, join



if_filter = False
devs = ['Dev1', 'Dev2', 'Dev3']

# Colors for plotting
# tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
#              (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
#              (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
#              (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
#              (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
# for i in range(len(tableau20)):
#     r, g, b = tableau20[i]
#     tableau20[i] = (r / 255., g / 255., b / 255.)

def dump2csv(filename):
    # file_name = filename.split("\\")[-1]
    name = filename.split('.')[0]

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
    df.to_csv(name + ".csv", header=False, index=False)

    # return name

def get_csv_name(dir):
    all_files = get_files(dir)
    csv_files = []
    for file in all_files:
        if file.split('.')[1] == 'csv':
            csv_files.append(file)

    return csv_files

def rd_csv(filename):
    my_data = genfromtxt(filename+".csv", delimiter=",")
    return my_data


def get_files(dir):
    cwd = os.getcwd()
    file_list = os.listdir(dir)
    file_path = [os.path.join(cwd, dir, x) for x in file_list]
    return file_path





if __name__ == '__main__':
    for i, device in enumerate(devs):
        files_path = get_files(device)
        for file in files_path:
            dump2csv(file)

        csv_files = get_csv_name(device)

        plt.figure(i)
        for j, file in enumerate(csv_files):
            # out_name = dump2csv(file)
            out_name = file.split('.')[0]
            data = rd_csv(out_name)
            data = data[data > 2000]

            x_axis = np.arange(1, len(data)+1)

            if if_filter:
                xf = np.arange(len(data))  # 频率
                yf = abs(fft(data))
                print(yf)
                xf2 = xf[range(int(len(x_axis) / 2))]  # 取一半区间
                yf1 = abs(fft(data)) / len(x_axis)  # 归一化处理
                yf2 = yf1[range(int(len(x_axis) / 2))]
                plt.plot(xf, yf, 'b',label=out_name.split('\\')[-1])
            else:
                plt.plot(x_axis, data, label=out_name.split('\\')[-1], linewidth=1.2, linestyle='-', alpha=0.7)

        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.legend()
        plt.xlabel('The frames')
        plt.ylabel('Output')
        plt.title(device + " " + "all scenarios measurement results")
        plt.savefig(device + " " + "plotting")





