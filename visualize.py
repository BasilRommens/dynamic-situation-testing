import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from inout import read_data
from knn import determine_distances, knn_situation
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

    # plot the results
    ax = sns.lineplot(x, y)

    # specify labels
    ax.set(xlabel=visualization_param.x_axis, ylabel=visualization_param.y_axis,
           title=visualization_param.title, xlim=(-1, 1), ylim=(0, 1))

    plt.show()


if __name__ == '__main__':
    # read the data from the csv and json file
    all_tuples, attributes, attribute_types, ordinal_attribute_values, attributes_to_ignore, decision_attribute = read_data(
        'german_credit_data.json', 'german_credit_data_class.csv')

    # process the data
    tuples, ranked_values, attributes, decision_attribute = process_all(
        all_tuples, attributes,
        attribute_types,
        ordinal_attribute_values,
        attributes_to_ignore,
        decision_attribute)
    # determine the distances
    protected_attributes = {"Sex": ["female"]}
    distance_dict = determine_distances(tuples, ranked_values, attribute_types,
                                        attributes, decision_attribute,
                                        protected_attributes)

    # apply the situation testing algorithm with knn
    k = 32
    valid_tuples = knn_situation(k, tuples, attributes, distance_dict,
                                 protected_attributes, decision_attribute)
    # TODO: add some formatting for using the right visualization parameters
    visualization_param = Visualize(
        "dataset=german credit - dec = class=0 - protected = female",
        "Fract. of tuples with diff >= t", "t")
    visualize_tuples(valid_tuples, visualization_param)
