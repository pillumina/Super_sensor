import numpy as np
import sys
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')


if __name__ == "__main__":


    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print('Invalid number of parameters! The first par is the name of directory and the second one is the '
              'name you wish to save for output figure!')
        exit(0)
    else:
        if len(sys.argv) == 3:
            print("Your directory name: %s " % (sys.argv[1]))
            print("Saved plot name: %s " % (sys.argv[2]))
            save_name = sys.argv[2][1:]
        elif len(sys.argv) == 2:
            print("Your directory name: %s " % (sys.argv[1]))
            save_name = sys.argv[1][1:]

    dir_name = sys.argv[1][1:]

    cwd = os.getcwd()
    dir = os.path.join(cwd, 'data')
    # obj_dirs = os.listdir(dir)
    # test_dir = obj_dirs[0]
    path = os.path.join(dir, dir_name)
    files_list = os.listdir(path)
    files_path = [os.path.join(path, x) for x in files_list]


    out_list = []
    for file in files_path:
        with open(file, 'r') as f:
            all_data = f.readlines()
            for line in all_data:
                try:
                    num = int(line.split('\n')[0])
                    # filter numbers smaller than 2000
                    if num > 2000:
                        out_list.append(int(line.split('\n')[0]))
                except:
                    continue
    out_array = np.array(out_list)

    # Plotting
    x_axis = np.arange(len(out_array))
    plt.plot(x_axis, out_array, linewidth=0.6, linestyle='-', alpha=0.7, color='mediumslateblue')
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel('The frames')
    plt.ylabel('Output')
    plt.title(dir_name + " " + "PIR data stream plot")
    if os.path.exists("figures") == False:
        os.mkdir("figures")
    save_path = os.path.join(cwd, "figures")
    plt.savefig(os.path.join(save_path,save_name))

