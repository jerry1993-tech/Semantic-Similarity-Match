# -*- coding: utf-8 -*-

import pandas as pd


def process_sentence(sentence):
    pass


def write_data_csv_to_seq_in(data_csv_path):
    data_csv = pd.read_csv(data_csv_path)

    data_new = {
        'sentence1': data_csv['sentence1'],
        'sentence2': data_csv['sentence2'],
        'label': data_csv['label']
    }

    data_df = pd.DataFrame(data_new)
    # shuffle
    data_csv_new = data_df.sample(frac=1).reset_index(drop=True)

    result = []
    for index, row in data_csv_new.iterrows():
        result.append([row['sentence1'], row['sentence2'], row['label']])

    return result


if __name__ == '__main__':

    train_csv_path = './data_csv/train/train.csv'
    valid_csv_path = './data_csv/valid/valid.csv'
    test_csv_path = './data_csv/test/test.csv'

    f1 = open('./data/train/train.seq1.in', 'a+')
    f2 = open('./data/train/train.seq2.in', 'a+')
    f3 = open('./data/train/train.label', 'a+')

    f4 = open('./data/valid/valid.seq1.in', 'a+')
    f5 = open('./data/valid/valid.seq2.in', 'a+')
    f6 = open('./data/valid/valid.label', 'a+')

    f7 = open('./data/test/test.seq1.in', 'a+')
    f8 = open('./data/test/test.seq2.in', 'a+')
    f9 = open('./data/test/test.label', 'a+')

    result_train = write_data_csv_to_seq_in(train_csv_path)
    result_valid = write_data_csv_to_seq_in(valid_csv_path)
    result_test = write_data_csv_to_seq_in(test_csv_path)

    for i in range(len(result_train)):
        f1.write(str(result_train[i][0]))
        f1.write('\n')
        f2.write(str(result_train[i][1]))
        f2.write('\n')
        f3.write(str(result_train[i][2]))
        f3.write('\n')

    for i in range(len(result_valid)):
        f4.write(str(result_valid[i][0]))
        f4.write('\n')
        f5.write(str(result_valid[i][1]))
        f5.write('\n')
        f6.write(str(result_valid[i][2]))
        f6.write('\n')

    for i in range(len(result_test)):
        f7.write(str(result_test[i][0]))
        f7.write('\n')
        f8.write(str(result_test[i][1]))
        f8.write('\n')
        f9.write(str(result_test[i][2]))
        f9.write('\n')

