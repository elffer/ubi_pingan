# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

path_train = "data/dm/train.csv"  # 训练文件
path_test = "data/dm/test.csv"  # 测试文件

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。


def read_csv_test(train_path, test_path):
    """
    文件读取模块，头文件见columns.
    :return: x_train, y_train, x_test, y_test
    """
    features = ["TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                        "CALLSTATE"]

    # train data
    train_df = pd.read_csv(train_path)
    train_df.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                        "CALLSTATE", "Y"]
    x_train = train_df[features].values
    y_train = train_df['Y']

    # test data
    test_df = pd.read_csv(test_path)
    test_df.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                        "CALLSTATE", "Y"]
    x_test = test_df[features].values
    y_test = test_df['Y']
    return x_train, x_test, y_train, y_test


def split_demo_csv(train_path, test_path, train_frac):
    """
    文件读取模块，头文件见columns.
    :return: x_train, y_train, x_test, y_test
    """
    # for filename in os.listdir(path_train):
    data = pd.read_csv(train_path)
    data.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                    "CALLSTATE", "Y"]
    train_df, test_df = train_validate_test_split(data, train_percent=train_frac, seed=None)

    train_df.to_csv(train_path)
    test_df.to_csv(test_path)


def read_predict_data(data_path):
    """
    文件读取模块，头文件见columns.
    :return:
    """
    # for filename in os.listdir(path_train):
    data_df = pd.read_csv(data_path)
    data_df.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                        "CALLSTATE"]
    features = ["TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                        "CALLSTATE"]
    x_pred = data_df[features].values
    id = data_df['TERMINALNO']
    return id, x_pred


def train_validate_test_split(df, train_percent=.7, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    train = df.ix[perm[:train_end]]
    test = df.ix[perm[train_end:]]
    return train, test


def write_csv(data_df, save_path):
    data_df.to_csv(save_path, columns=['Id', 'Pred'], index=False, header=False)


def process():
    """
    处理过程，在示例中，使用随机方法生成结果，并将结果文件存储到预测结果路径下。
    :return: 
    """

    with open(path_test) as lines:
        with(open(os.path.join(path_test_out, "test.csv"), mode="w")) as outer:
            writer = csv.writer(outer)
            i = 0
            ret_set = set([])
            for line in lines:
                if i == 0:
                    i += 1
                    writer.writerow(["Id", "Pred"])  # 只有两列，一列Id为用户Id，一列Pred为预测结果(请注意大小写)。
                    continue
                item = line.split(",")
                if item[0] in ret_set:
                    continue
                # 此处使用随机值模拟程序预测结果
                writer.writerow([item[0], np.random.rand()]) # 随机值
                
                ret_set.add(item[0])  # 根据赛题要求，ID必须唯一。输出预测值时请注意去重


def train():
    x_train, x_test, y_train, y_test = read_csv_test(path_train, train_frac=0.8)
    model = LinearRegression()
    model.fit(x_train, y_train)

    # predict off-line
    y_predict = model.predict(x_test)

    # predict on-line
    y_pred_df = pd.DataFrame(y_predict)
    y_pred_df.reset_index()

    print('y_predict = ', y_pred_df.head(n=10))

if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    # train()
