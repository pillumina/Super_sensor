import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os
import re


Dev1 = ['noone2min', 'static2min', 'walk2min', 'wave2min']
Dev2 = ['noone2min', 'static2min', 'typing2min', 'walk2min', 'wave2min']
Dev3 = ['noone2min', 'static2min', 'typing2min', 'walking2min', 'wave2min']
dir_lists = [Dev1, Dev2, Dev3]
dev_lists = ['Dev1', 'Dev2', 'Dev3']

def dump2txt(dirname, filename):
    # file_name = filename.split("\\")[-1]
    # name = filename.split('.')[0]
    cwd = os.getcwd()
    dir = os.path.join(cwd, 'data')
    path = os.path.join(dir, dirname)
    file = os.path.join(path, filename + '.txt')

    with open(file) as f:
        data = f.readlines()

    out_data = []
    for line in data:
        match = re.search('curr: (\d+)', line)
        if match:
            temp = int(match.group(1))
            out_data.append(temp)

    np_array = np.asarray(out_data)

    if os.path.exists(file) is False:
        print("The path of the file is not found, please check your spelling and make sure the file is in the directory.")
    else:
        new_dir = os.path.join(dir, dirname + "_" + filename)
        if os.path.exists(new_dir) is False:
            os.mkdir(new_dir)
        np.savetxt(os.path.join(new_dir, filename + '.txt'), np_array, '%d')


def data2arr(dir_name, plotting):
    if os.path.exists("data") == False:
        print("Please create a directory called 'data' under current working space "
              "and then put all your data directories in it.")
    else:
        save_name = dir_name
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
        if plotting:
            plt.figure()
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

        return out_array

def plot_arr(arr, filename, dirname):
    cwd = os.getcwd()
    plt.figure()
    x_axis = np.arange(len(arr))
    plt.plot(x_axis, arr, linewidth=0.6, linestyle='-', alpha=0.7, color='mediumslateblue')
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel('The frames')
    plt.ylabel('Output')
    plt.title(filename + " " + "PIR data stream plot")
    if os.path.exists(dirname) == False:
        os.mkdir(dirname)
    save_path = os.path.join(cwd, dirname)
    # plt.savefig(os.path.join(save_path, filename))
    plt.show()

def z_score(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    out_arr = (arr - mean) // std
    return out_arr


def z_plot(arr, filename):
    z_arr = z_score(arr)
    plot_arr(z_arr, filename, 'z_figures')

def elim_outliers(arr, filename, plotting):
    temp_arr = z_score(arr)
    args = np.argwhere((temp_arr > 3) | (temp_arr < -3))
    args = args.reshape((len(args), 1))
    out_arr = np.delete(arr, args, None)
    if plotting:
        plot_arr(out_arr, filename, 'no_outliers')
    # or using mask
    # mask = np.ones(len(arr), dtype=bool)
    # mask[args] = False
    # out_arr = arr[mask]
    return out_arr


def info_entropy(arr):
    """
    calculate the shanno entropy of the arr
    """
    x_value_list = set([arr[i] for i in range(arr.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(arr[arr == x_value].shape[0]) / arr.shape[0]
        logp = np.log2(p)
        ent -= p * logp

    return ent


def max_vote(arr):
    '''
    :param arr:  numpy array consists of 0/1
    :return: the result of max voting.
    '''
    counts = np.bincount(arr)
    out = np.argmax(counts)
    return out

def hot_coding(arr, threshold):
    '''
    :param arr: numpy array that used for encoding.
    :param threshold:  A tuple -- (lowerbound, upperbound)
    :return: encoded arr with the same length
    '''
    lo, hi = threshold
    bi_arr = np.ones_like(arr)
    bi_arr[(arr >= lo) & (arr <= hi)] = 0
    bi_arr = bi_arr.astype('int64')
    return bi_arr

def near_diff(arr):
    return np.diff(arr)

def slide_algo(arr, len_unit, slide_step, threshold):
    '''
    :param arr:  all data in one section
    :param len_unit:  the length of the unit window.
    :param slide_step:  the shift scale of the unit window.
    :param threshold: tuple (lo, hi) for hot encoding.
    :return: the max voting result of the final array.
    '''
    ls = len(arr)
    if slide_step > ls:
        slide_step = ls
    if len_unit > ls:
        len_unit = ls

    i = 0
    stop = False
    fn_list = []
    while i <= ls and stop is False:
        upp = i + len_unit + 1
        if upp > ls:
            upp = ls
            stop = True
        wd = arr[i:upp]
        wd_diff = near_diff(wd)
        wd_diff = np.abs(wd_diff)
        # print(wd_diff)
        wd_ecod = hot_coding(wd_diff, threshold)
        # print(wd_ecod)
        mx_vt = max_vote(wd_ecod)
        fn_list.append(mx_vt)
        i = i + slide_step
    fn_arr = np.asarray(fn_list)
    result = max_vote(fn_arr)
    return (result, fn_arr)




if __name__ == "__main__":
    # # check the slide algo
    # arr = np.asarray([1.2, 1.3, 1.7, 2.5, 2.7, 2.6, 1.8, 1.6, 1.6, 2.6, 2.7])
    # (result, fn_arr) = slide_algo(arr, 3, 1, (0.2, 0.4))
    # print("----------------------------------------------")
    # print(fn_arr)
    # print("----------------------------------------------")
    # print(result)

    # dir = 'PIR_C10_NO'
    # arr = data2arr(dir, False)
    # arr = elim_outliers(arr, dir, False)
    # # (result, fn_arr) =  slide_algo(arr, 500, 100, (0, 10))
    # # count = np.bincount(fn_arr)
    # # print(len(fn_arr))
    # # print(result)
    # arr_diff = near_diff(arr)
    # x_axis = np.arange(len(arr_diff))
    # plt.plot(x_axis, arr_diff)
    # plt.show()


    # plt.figure()
    dir1 = 'Dev2_wave2min'
    arr1 = data2arr(dir1, False)
    arr1 = elim_outliers(arr1, dir1, False)
    arr_diff1 = near_diff(arr1)
    x_axis1 = np.arange(len(arr_diff1))
    plt.plot(x_axis1, arr_diff1)
    plt.show()

    # dirnames = []
    # for i in range(3):
    #     dev = dev_lists[i]
    #     dirname_list = dir_lists[i]
    #     for dirname in dirname_list:
    #         temp_name = dev + '_' + dirname
    #         dirnames.append(temp_name)
    #
    # # print(dirnames[:5])
    # for dir in dirnames:
    #     out_arr = data2arr(dir)
    #     elim_outliers(out_arr, dir)

    # # check the entropy of various devices
    # ent = []
    # for dir in Dev3:
    #     dir_name = 'Dev3_' + dir
    #     out_arr = data2arr(dir_name, False)
    #     fn_arr = elim_outliers(out_arr, dir_name, False)
    #     entro = info_entropy(fn_arr)
    #     ent.append(entro)
    # print(ent)

    # ent = []
    # dirnames = ['PIR_C07_NO', 'PIR_C10_NO', 'PIR_C19_NO']
    # for dir in dirnames:
    #     out_arr = data2arr(dir, False)
    #     fn = elim_outliers(out_arr, dir, False)
    #     entro = info_entropy(fn)
    #     ent.append(entro)
    # print(ent)