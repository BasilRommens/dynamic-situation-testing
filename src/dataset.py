import numpy as np
import pandas as pd


def create_gaussian(dim, n):
    anchor_point = np.random.randint(low=-100, high=100, size=dim)
    new_points = list()
    for _ in range(n):
        new_point = np.random.normal(size=dim)
        new_point += anchor_point
        new_points.append(new_point)
    return new_points, anchor_point


def prepare_german_credit():
    # set the fnames of the original and kaggle datasets
    og_data_fname = 'data/german.data'
    kaggle_data_fname = 'data/german_credit_data.csv'

    # load the original dataset
    og_data = np.loadtxt(og_data_fname, dtype=str)

    # take the last column of the dataset which represents the class
    og_data_class = og_data[:, -1]

    # load the kaggle dataset
    kaggle_data = pd.read_csv(kaggle_data_fname)

    # remove the 'idx' column from the dataset
    kaggle_data = kaggle_data.drop(columns=['idx'])

    # add the class column to the kaggle dataset
    kaggle_data['Class'] = og_data_class

    # replace all the class values of 2 with 0 and make all numeric
    kaggle_data['Class'] = kaggle_data['Class'].replace('2', '0').astype(int)

    # save the new dataset
    kaggle_data.to_csv('data/german_credit_data_processed.csv', index=False)


def prepare_COMPAS():
    df = pd.read_csv('data/compas-scores-raw.csv')

    # convert the screening data column to a datetime object
    df['Screening_Date'] = pd.to_datetime(df['Screening_Date'])

    # convert the DateOfBirth column to a datetime object
    df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'])

    # change the Screening_Date column and the DateOfBirth column to a numeric value
    df['Screening_Date'] = df['Screening_Date'].astype(np.int64)
    df['DateOfBirth'] = df['DateOfBirth'].astype(np.int64)

    # write the compas data to a csv file
    df.to_csv('data/compas-scores-raw-processed.csv', index=False)


if __name__ == '__main__':
    # pts = create_gaussian(6, 100)
    # prepare_german_credit()
    prepare_COMPAS()
