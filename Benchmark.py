from utils import *
from SDF import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def new_data2arr(dir_name, plotting):
    import numpy as np
    import os
    import matplotlib.pyplot as plt

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
                        str_list = line.split(':"')[1]
                        num_list = str_list.split(',')
                        del num_list[-1]
                        num_arr = list(map(int, num_list))
                        tmp_arr = np.asarray(num_arr)
                        tmp_arr = tmp_arr[tmp_arr>2000]
                        tmp_list = tmp_arr.tolist()
                        out_list.extend(tmp_list)
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
            # plt.savefig(os.path.join(save_path,save_name))
            plt.show()

        return out_array


def slice_arr(arr, slice_Len):
    length = len(arr) // slice_Len
    B = arr[:(length * slice_Len)]
    C = np.reshape(B, (length, slice_Len))
    return C

def label_arr(arr, label):
    length = len(arr)
    temp = np.zeros(length)
    temp[:] =  label
    return temp

def eval_err(X_train, y_train, X_test, y_test, upper, plotting=False):

    error = []
    for i in range(1, upper):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))

    if plotting:
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
                 markerfacecolor='blue', markersize=10)
        plt.title('Error Rate K Value')
        plt.xlabel('K Value')
        plt.ylabel('Mean Error')
        plt.show()
    return error


def knn_algo(data, label, k_value):
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.10)
    classifier = KNeighborsClassifier(n_neighbors=k_value)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    print("Confusion Matrix:")

    print(confusion_matrix(y_test, y_pred))

    print('-----------------------------------')
    print(classification_report(y_test, y_pred))



def find_best_k(data, label, upper, repeats):
    k_lists=[]
    for i in range(repeats):
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.10)
        error = eval_err(X_train, y_train, X_test, y_test, upper, False)
        kk = range(1, upper)
        k_min = kk[error.index(min(error))]
        k_lists.append(k_min)
    k_arr = np.asarray(k_lists)
    freq_k = np.bincount(k_arr).argmax()
    return freq_k


'''
Cross Validation for different classifiers
'''

def knn_cross_val(data, label, k_value, n_folds, test_size=0.3):
    cv = ShuffleSplit(n_splits=n_folds, test_size= test_size, random_state=0)

    clf = KNeighborsClassifier(n_neighbors=k_value)
    scores = cross_val_score(clf, data, label, cv=cv)
    return scores


def svm_cross_val(data, label, C, n_folds, test_size=0.3):
    cv = ShuffleSplit(n_splits=n_folds, test_size= test_size, random_state=0)

    clf = svm.SVC(C=C, kernel='rbf', gamma=20, decision_function_shape='ovo')
    scores = cross_val_score(clf, data, label, cv=cv)
    return scores


def tree_cross_val(data, label, n_folds, test_size=0.3):
    cv = ShuffleSplit(n_splits=n_folds, test_size=test_size, random_state=0)

    clf = DecisionTreeClassifier(criterion='gini')
    scores = cross_val_score(clf, data, label, cv=cv)
    return scores


def rf_cross_val(data, label, n_estimator, n_folds, test_size=0.3):
    cv = ShuffleSplit(n_splits=n_folds, test_size=test_size, random_state=0)

    clf = RandomForestClassifier(n_estimators=n_estimator)
    scores = cross_val_score(clf, data, label, cv=cv)
    return scores



if __name__ == '__main__':
    # dir = 'tp1'
    # arr = new_data2arr(dir, False)
    # arr = elim_outliers(arr, dir, False)
    # print(arr.shape)
    # feature = sdf_features(arr, 6, False)
    # print(feature)
    #
    # dir1 = 'non_1'
    # arr1 = new_data2arr(dir1, False)
    # arr1 = elim_outliers(arr1, dir1, False)
    # feature1 = sdf_features(arr1, 6, False)
    # print(feature1)
    #
    # dir2 = 'tp_3'
    # arr2 = new_data2arr(dir2, False)
    # arr2 = elim_outliers(arr2, dir2, False)
    # feature2 = sdf_features(arr2, 6, False)
    # print(feature2)


    # C = slice_arr(arr, 20)
    # lb1 = label_arr(C, 1)


    # dirs = ['tp1','tp_1', 'wave_left_right1','wave_up_down1', 'stall']
    dirs = ['tp_1']
    arr_list = []
    for dir in dirs:
        arr = new_data2arr(dir, False)
        arr = elim_outliers(arr, dir, False)
        arr.tolist()
        arr_list.extend(arr)
    arrs = np.asarray(arr_list)
    C = slice_arr(arrs, 20)
    lb1 = label_arr(C, 1)


    dir1s = ['non1_30']
    arr_list1 = []
    for dir in dir1s:
        arr = new_data2arr(dir, False)
        arr = elim_outliers(arr, dir, False)
        arr.tolist()
        arr_list1.extend(arr)
    arrs1 = np.asarray(arr_list1)

    D = slice_arr(arrs1, 20)
    lb2 = label_arr(D, 0)

    # rst = np.apply_along_axis(sdf_features,1, C, 20)
    # rst1 = np.apply_along_axis(sdf_features, 1, D, 20)

    # rst_f = np.concatenate((rst, rst1), axis=0)


    C_D = np.concatenate((C, D), axis=0)

    lb = np.concatenate((lb1, lb2), axis=0)

    print(C_D.shape)
    print(lb.shape)
    # rst_f = np.apply_along_axis(sdf_features, 1, C_D, 20)
    # print(rst_f[0])


    #
    X_train, X_test, y_train, y_test = train_test_split(C_D, lb, test_size=0.30)
    # print(X_train.shape)
    # print(X_test.shape)

    X_train_sdf = np.apply_along_axis(sdf_features, 1, X_train, 20)
    classifier = KNeighborsClassifier(n_neighbors=50)
    classifier.fit(X_train_sdf, y_train)

    X_train = X_train.flatten()

    # print(np.reshape(X_train, (X_train.shape[0] * X_train.shape[1], 1)).shape)
    partition = max_entropy_partition(X_train,20)


    print(min(X_train))
    print(max(X_train))

    print(partition.shape)
    print(partition)

    features = []
    for i in range(len(X_test)):
        xtest = X_test[i]
        symbols = generate_symbol_sequence(xtest, partition)
        morph_matrix, pvec = analyze_symbol_sequence(symbols, 20, False)
        feature = pvec
        features.append(feature)
        # print(feature.shape)
    fs_arr = np.asarray(features)
    # print(fs_arr[:10])

    y_pred = classifier.predict(fs_arr)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))





    # print(find_best_k(C_D, lb, 100, 20))
    # knn_algo(rst_f, lb, 30)
    # print(knn_cross_val(rst_f, lb, 21, 10, 0.3).mean())
    # print(tree_cross_val(rst_f, lb, 20, 0.3).mean())
    # print(rf_cross_val(rst_f, lb, 30,  20, 0.3).mean())`
