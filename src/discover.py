from inout import read_data
from knn import calc_dist_mat, knn_situation
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
    r = read_data('german_credit_data.json', 'german_credit_data_class.csv')

    # process the data
    tuples, ranked_values, decision_attribute = process_all(r)
    # determine the distances
    protected_attributes = {"Sex": ["male"]}
    dist_mat = calc_dist_mat(tuples, ranked_values, r.attribute_types,
                             decision_attribute,
                             protected_attributes)

    # apply the situation testing algorithm with knn
    k = 16
    valid_tuples = knn_situation(k, tuples, dist_mat,
                                 protected_attributes, decision_attribute)

    # discover the discriminated tuples among the valid tuples
    discriminated_tuples = discover_disc_situation(valid_tuples, threshold=0.05)
    print(discriminated_tuples)
