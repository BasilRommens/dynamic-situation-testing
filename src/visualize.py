import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from inout import read_data
from knn import calc_dist_mat, knn_situation
from dataclasses import dataclass
from process import process_all


@dataclass
class Visualize:
    title: str
    y_axis: str
    x_axis: str


def visualize_tuples(tuples, visualization_param):
    # strip indices and keeping the diff values
    clean_tuples = map(lambda x: x[1], tuples)

    # count the number of times each diff occurs
    diff_count = list(dict(Counter(clean_tuples)).items())

    # sort the diff count for applying a cumsum
    sorted_tuples = sorted(diff_count, key=lambda x: x[0], reverse=True)

    # calculate the cumulative sum of the diff counter values
    cumsum_nr_values = np.cumsum(list(map(lambda x: x[1], sorted_tuples)))
    # det
    nr_tuples = len(tuples)
    percentage_values = cumsum_nr_values / nr_tuples

    diff_values = list(map(lambda x: x[0], sorted_tuples))

    # determine x and y values for plotting
    x = diff_values
    y = percentage_values
    # add a (max t, 0) point
    # x = np.insert(x, 0, max(x))
    # y = np.insert(y, 0, 0)

    # set figure size before plotting
    plt.figure(figsize=(7, 4.8))

    # plot the results
    ax = sns.lineplot(x=x, y=y)

    # specify labels
    ax.set(xlabel=visualization_param.x_axis, ylabel=visualization_param.y_axis,
           title=visualization_param.title, xlim=(-1, 1), ylim=(-0.1, 1.1))
    # adding a grid to the plot
    ax.grid(visible=True, which='major', color='black', linewidth=1.0,
            linestyle=":")
    ax.grid(visible=True, which='minor', color='black', linewidth=0.5,
            linestyle=":")
    # adding ticks on the correct spots
    ax.set_xticks(np.arange(-1, 1.1, 0.5))
    ax.set_yticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1])

    plt.show()


def visualize_data(data, x, y, color):
    # plot the results
    ax = sns.scatterplot(x=x, y=y, data=data, hue=color)

    # specify labels
    ax.set(xlabel=x, ylabel=y, title='Scatter plot')

    plt.show()


def visualize_variable_kde(data, var_name, color):
    """
    Visualize the heatmap of the data for the variable name passed
    :param data: the data to visualize
    :param var_name: the variable name to visualize
    :return: nothing
    """
    sns.kdeplot(data=data, x='x', y='y', weights=var_name, fill=True,
                alpha=0.1, color=color).set(title=var_name)
    plt.show()


def visualize_kde(arr):
    """
    visualizes an array in the form of a kde plot
    :param arr: a 1d array containing numeric values
    :return: nothing
    """
    sns.kdeplot(data=arr, fill=True).set(title="kde plot")
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv("data/german_credit_data_class.csv")
    visualize_data(df, 'Credit amount', 'Age', 'Class')
    # read the data from the csv and json file
    r = read_data('data/german_credit_data.json', 'data/german_credit_data_class.csv')

    # process the data
    tuples, ranked_values, decision_attribute = process_all(r)
    # determine the distances
    protected_attributes = {"Sex": ["female"]}
    dist_mat = calc_dist_mat(tuples, ranked_values, r.attribute_types,
                             decision_attribute, protected_attributes)

    # apply the situation testing algorithm with knn
    k = 32
    valid_tuples = knn_situation(k, tuples, dist_mat,
                                 protected_attributes, decision_attribute)
    # TODO: add some formatting for using the right visualization parameters
    visualization_param = Visualize(
        "dataset=german credit - dec = class=0 - protected = female",
        "Fract. of tuples with diff >= t", "t")
    visualize_tuples(valid_tuples, visualization_param)
