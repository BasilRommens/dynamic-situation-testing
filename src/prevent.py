import copy

from discover import discover_disc_situation
from inout import read_data
from knn import calc_dist_mat, knn_situation
from process import process_all


def prevent_situation(tuples, discriminated_tuples, decision_attribute,
                       decision_attr_type=None):
    attributes = list(tuples.columns)
    decision_attribute_name = list(decision_attribute.keys())[0]
    decision_attribute_idx = attributes.index(decision_attribute_name)
    decision_attribute_val = decision_attribute[decision_attribute_name]

    # the value to switch to for discrimination prevention
    to_val = 0
    if decision_attr_type is None:
        to_val = 1 if decision_attribute_val == 0 else 0

    # get all the indices of the truly discriminated tuples
    discriminated_tuples_neg_idx = [discriminated_tuple[0] for
                                    discriminated_tuple in discriminated_tuples
                                    if discriminated_tuple[1]]

    non_discriminated_tuples = copy.deepcopy(tuples)
    for idx, tuple in enumerate(tuples):
        if idx in discriminated_tuples_neg_idx:
            tuple[decision_attribute_idx] = to_val
        non_discriminated_tuples.append(tuple)

    return non_discriminated_tuples


if __name__ == "__main__":
    # read the data from the csv and json file
    all_tuples,  attribute_types, ordinal_attribute_values, attributes_to_ignore, decision_attribute = read_data(
        'german_credit_data.json', 'german_credit_data_class.csv')

    # process the data
    tuples, ranked_values,  decision_attribute = process_all(
        all_tuples,
        attribute_types,
        ordinal_attribute_values,
        attributes_to_ignore,
        decision_attribute)
    # determine the distances
    protected_attributes = {"Sex": ["male"]}
    dist_mat = calc_dist_mat(tuples, ranked_values, attribute_types,
                             decision_attribute,
                             protected_attributes)

    # apply the situation testing algorithm with knn
    k = 16
    valid_tuples = knn_situation(k, tuples,  dist_mat,
                                 protected_attributes, decision_attribute)

    # discover the discriminated tuples among the valid tuples
    discriminated_tuples = discover_disc_situation(valid_tuples, threshold=0.05)
    discriminated_tuple_count = len(
        [discriminated_tuple for discriminated_tuple in discriminated_tuples if
         discriminated_tuple[1]])
    print("Discriminated tuples: {}".format(discriminated_tuple_count))

    # prevent discrimination in all the tuples
    non_discriminated_tuples = prevent_situation(tuples, discriminated_tuples,
                                                 decision_attribute, attributes)

    # apply the situation testing algorithm again knn, with the bias 'removed'
    dist_mat = calc_dist_mat(non_discriminated_tuples, ranked_values,
                             attribute_types,
                             decision_attribute,
                             protected_attributes)
    valid_tuples = knn_situation(k, non_discriminated_tuples,
                                 dist_mat, protected_attributes,
                                 decision_attribute)

    # discover the discriminated tuples among the valid tuples
    discriminated_tuples = discover_disc_situation(valid_tuples, threshold=0.05)
    discriminated_tuple_count = len(
        [discriminated_tuple for discriminated_tuple in discriminated_tuples if
         discriminated_tuple[1]])
    print("Discriminated tuples: {}".format(discriminated_tuple_count))

    # manipulate all the tuples
    non_discriminated_tuples = prevent_situation(tuples, discriminated_tuples,
                                                 decision_attribute, attributes)

    # apply the situation testing algorithm again knn, with the bias 'removed'
    dist_mat = calc_dist_mat(non_discriminated_tuples, ranked_values,
                             attribute_types,
                             decision_attribute,
                             protected_attributes)
    valid_tuples = knn_situation(k, non_discriminated_tuples,
                                 dist_mat, protected_attributes,
                                 decision_attribute)

    # discover the discriminated tuples among the valid tuples
    discriminated_tuples = discover_disc_situation(valid_tuples, threshold=0.05)
    discriminated_tuple_count = len(
        [discriminated_tuple for discriminated_tuple in discriminated_tuples if
         discriminated_tuple[1]])
    print("Discriminated tuples: {}".format(discriminated_tuple_count))
    # manipulate all the tuples
    non_discriminated_tuples = prevent_situation(tuples, discriminated_tuples,
                                                 decision_attribute, attributes)

    # apply the situation testing algorithm again knn, with the bias 'removed'
    dist_mat = calc_dist_mat(non_discriminated_tuples, ranked_values,
                             attribute_types,
                             decision_attribute,
                             protected_attributes)
    valid_tuples = knn_situation(k, non_discriminated_tuples,
                                 dist_mat, protected_attributes,
                                 decision_attribute)

    # discover the discriminated tuples among the valid tuples
    discriminated_tuples = discover_disc_situation(valid_tuples, threshold=0.05)
    discriminated_tuple_count = len(
        [discriminated_tuple for discriminated_tuple in discriminated_tuples if
         discriminated_tuple[1]])
    print("Discriminated tuples: {}".format(discriminated_tuple_count))
