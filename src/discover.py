from inout import read_data
from knn import determine_distances, knn_situation
from process import process_all


def discover_disc_situation(idx_disc_tuples, threshold):
    # list containing (idx, disc) tuples, where the first value is the index of
    # the tuple and the second if it has been discriminated against according
    # to the threshold
    discriminated_tuples = list()

    # determine the number of tuples that have been discriminated against
    for idx_disc_tuple in idx_disc_tuples:
        # if tuple disc factor exceeds threshold, tuple has been discriminated
        if idx_disc_tuple[1] > threshold:
            discriminated_tuples.append((idx_disc_tuple[0], True))
        else:
            discriminated_tuples.append((idx_disc_tuple[0], False))

    return discriminated_tuples


if __name__ == "__main__":
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
    protected_attributes = {"Sex": ["male"]}
    distance_dict = determine_distances(tuples, ranked_values, attribute_types,
                                        attributes, decision_attribute,
                                        protected_attributes)

    # apply the situation testing algorithm with knn
    k = 16
    valid_tuples = knn_situation(k, tuples, attributes, distance_dict,
                                 protected_attributes, decision_attribute)
    discriminated_tuples = discover_disc_situation(valid_tuples, threshold=0.05)
    print(discriminated_tuples)
