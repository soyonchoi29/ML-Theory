import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import csv
import matplotlib.pyplot as plt

from get_favorite_data import *


if __name__ == "__main__":
    assert 1/2 == 0.5, "Are you sure you're using python 3?"
    print(f"Version of sklearn: {sklearn.__version__}")
    print("(It should be 1.4.1)")

    # Team Purple:
    l_p1 = KNeighborsClassifier(n_neighbors=5)
    l_p2 = DecisionTreeClassifier()
    l_p3 = SVC(kernel="linear")
    l_p4 = SVC(kernel="gaussian")

    # Team Orange:
    l_o1 = Ridge()
    l_o2 = KernelRidge(kernel="poly", degree=3)
    l_o3 = KernelRidge(kernel="poly", degree=10)
    l_o4 = KernelRidge(kernel="rbf")

    # number of samples to draw
    n = 1000
    mse_s = {}

    file = open('mse_purple.csv', 'w', newline='')
    writer = csv.writer(file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['MSE Scores', 'P1', 'P2', 'P3', 'P4'])

    X_1, y_1 = get_favorite_data_1(n)
    X_2, y_2 = get_favorite_data_2(n)
    X_3, y_3 = get_favorite_data_3(n)
    X_4, y_4 = get_favorite_data_4(n)

    # plot datasets (just bc i wanna see :))
    fig, axs = plt.subplots(2, 2, figsize=(15, 5))
    fig.set_tight_layout({"pad": 1})

    axs[0, 0].scatter(X_1, y_1)
    axs[0, 0].set_title('dataset 1')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')

    axs[1, 0].scatter(X_2, y_2)
    axs[1, 0].set_title('dataset 2')
    axs[1, 0].set_xlabel('x')
    axs[1, 0].set_ylabel('y')

    axs[0, 1].scatter(X_3, y_3)
    axs[0, 1].set_title('dataset 3')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('y')

    axs[1, 1].scatter(X_4, y_4)
    axs[1, 1].set_title('dataset 4')
    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_ylabel('y')

    # axs[1, 1].remove()
    # axs[1, 1] = fig.add_subplot(2, 2, 4, projection='3d')
    # axs[1, 1].scatter(X_4[:, 0], X_4[:, 1], y_4)
    # axs[1, 1].set_title('dataset 4 (just the first 2 dimensions)')
    # axs[1, 1].set_xlabel('x')
    # axs[1, 1].set_ylabel('y')
    # axs[1, 1].set_zlabel('z')

    plt.show()


    # split datasets into training and testing sets
    X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(X_1, y_1, test_size=0.25, random_state=42)
    X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_2, y_2, test_size=0.25, random_state=42)
    X_3_train, X_3_test, y_3_train, y_3_test = train_test_split(X_3, y_3, test_size=0.25, random_state=42)
    X_4_train, X_4_test, y_4_train, y_4_test = train_test_split(X_4, y_4, test_size=0.25, random_state=42)

    print('got all the data!')

    # fit the learners
    l_o1.fit(X_1_train, y_1_train)
    l_o1_pred_1 = l_o1.predict(X_1_test)
    l_o1.fit(X_2_train, y_2_train)
    l_o1_pred_2 = l_o1.predict(X_2_test)
    l_o1.fit(X_3_train, y_3_train)
    l_o1_pred_3 = l_o1.predict(X_3_test)
    l_o1.fit(X_4_train, y_4_train)
    l_o1_pred_4 = l_o1.predict(X_4_test)

    mse_s['learner 1'] = []
    mse_s['learner 1'].append(mean_squared_error(y_1_test, l_o1_pred_1))
    mse_s['learner 1'].append(mean_squared_error(y_2_test, l_o1_pred_2))
    mse_s['learner 1'].append(mean_squared_error(y_3_test, l_o1_pred_3))
    mse_s['learner 1'].append(mean_squared_error(y_4_test, l_o1_pred_4))
    writer.writerow(['Learner 1'] + mse_s['learner 1'])

    print('done with learner 1!')

    l_o2.fit(X_1_train, y_1_train)
    l_o2_pred_1 = l_o2.predict(X_1_test)
    l_o2.fit(X_2_train, y_2_train)
    l_o2_pred_2 = l_o2.predict(X_2_test)
    l_o2.fit(X_3_train, y_3_train)
    l_o2_pred_3 = l_o2.predict(X_3_test)
    l_o2.fit(X_4_train, y_4_train)
    l_o2_pred_4 = l_o2.predict(X_4_test)

    mse_s['learner 2'] = []
    mse_s['learner 2'].append(mean_squared_error(y_1_test, l_o2_pred_1))
    mse_s['learner 2'].append(mean_squared_error(y_2_test, l_o2_pred_2))
    mse_s['learner 2'].append(mean_squared_error(y_3_test, l_o2_pred_3))
    mse_s['learner 2'].append(mean_squared_error(y_4_test, l_o2_pred_4))
    writer.writerow(['Learner 2'] + mse_s['learner 2'])

    print('done with learner 2!')

    l_o3.fit(X_1_train, y_1_train)
    l_o3_pred_1 = l_o3.predict(X_1_test)
    l_o3.fit(X_2_train, y_2_train)
    l_o3_pred_2 = l_o3.predict(X_2_test)
    l_o3.fit(X_3_train, y_3_train)
    l_o3_pred_3 = l_o3.predict(X_3_test)
    l_o3.fit(X_4_train, y_4_train)
    l_o3_pred_4 = l_o3.predict(X_4_test)

    mse_s['learner 3'] = []
    mse_s['learner 3'].append(mean_squared_error(y_1_test, l_o3_pred_1))
    mse_s['learner 3'].append(mean_squared_error(y_2_test, l_o3_pred_2))
    mse_s['learner 3'].append(mean_squared_error(y_3_test, l_o3_pred_3))
    mse_s['learner 3'].append(mean_squared_error(y_4_test, l_o3_pred_4))
    writer.writerow(['Learner 3'] + mse_s['learner 3'])

    print('done with learner 3!')

    l_o4.fit(X_1_train, y_1_train)
    l_o4_pred_1 = l_o4.predict(X_1_test)
    l_o4.fit(X_2_train, y_2_train)
    l_o4_pred_2 = l_o4.predict(X_2_test)
    l_o4.fit(X_3_train, y_3_train)
    l_o4_pred_3 = l_o4.predict(X_3_test)
    l_o4.fit(X_4_train, y_4_train)
    l_o4_pred_4 = l_o4.predict(X_4_test)

    mse_s['learner 4'] = []
    mse_s['learner 4'].append(mean_squared_error(y_1_test, l_o4_pred_1))
    mse_s['learner 4'].append(mean_squared_error(y_2_test, l_o4_pred_2))
    mse_s['learner 4'].append(mean_squared_error(y_3_test, l_o4_pred_3))
    mse_s['learner 4'].append(mean_squared_error(y_4_test, l_o4_pred_4))
    writer.writerow(['Learner 4'] + mse_s['learner 4'])

    print('done with learner 4!')
    print('done :)')