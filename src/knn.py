import json

from distance import total_distance
from inout import read_data
from process import process_all


def determine_distances(tuples, ranked_values, attribute_types, attributes,
                        decision_attribute, protected_attributes):
    distance_dict = dict()
    for idx_1, t_1 in enumerate(tuples):
        for idx_2, t_2 in enumerate(tuples):
            distance = total_distance(t_1, t_2, tuples, attributes,
                                      attribute_types, ranked_values,
                                      decision_attribute, protected_attributes)
            distance_dict[str((idx_1, idx_2))] = distance
    return distance_dict


def check_part_of_protected_group(tuple, protected_attributes, attributes):
    for attribute, values in protected_attributes.items():
        attribute_idx = attributes.index(attribute)
        if tuple[attribute_idx] not in values:
            return False
    return True


def knn_group(k, tuple_idx, group_idxs, distance_dict):
    idx_distance = list()
    for other_tuple_idx in group_idxs:
        # skip the tuple that has the same index as the one we are determining
        # the k nearest neighbors for
        if other_tuple_idx == tuple_idx:
            continue

        pair_str = str((tuple_idx, other_tuple_idx))
        distance = distance_dict[pair_str]

        idx_distance.append((other_tuple_idx, distance))

    idx_distance = sorted(idx_distance, key=lambda x: x[1])

    k_idx_distance = idx_distance[:k]

    return k_idx_distance


def knn_situation(k, tuples, attributes, distance_dict, protected_attributes,
                  decision_attribute):
    # in knn for situation testing first we need to construct the protected and
    # unprotected groups
    protected_tuple_idxs = set()
    unprotected_tuple_idxs = set()
    for idx, tuple in enumerate(tuples):
        if check_part_of_protected_group(tuple, protected_attributes,
                                         attributes):
            protected_tuple_idxs.add(idx)
        else:
            unprotected_tuple_idxs.add(idx)

    # now we need to find the k nearest neighbors for each tuple in each group
    decision_attribute_name = list(decision_attribute.keys())[0]
    decision_attribute_value = decision_attribute[decision_attribute_name]
    decision_attribute_idx = attributes.index(decision_attribute_name)
    valid_tuples = list()
    for idx, tuple in enumerate(tuples):
        # continue if the idx is in the unprotected group
        if idx in unprotected_tuple_idxs:
            continue
        # if decision of idx is not same as decision attribute then continue
        if tuple[decision_attribute_idx] != decision_attribute_value:
            continue

        _diff = diff(decision_attribute_idx, distance_dict, idx, k,
                     protected_tuple_idxs, tuple, tuples,
                     unprotected_tuple_idxs)
        valid_tuples.append((idx, _diff))
    return valid_tuples


def diff(decision_attribute_idx, distance_dict, idx, k, protected_tuple_idxs,
         tuple, tuples, unprotected_tuple_idxs):
    # find the k nearest neighbors in the (un)protected group
    idx_group_prot = knn_group(k, idx, protected_tuple_idxs, distance_dict)
    idx_group_unprot = knn_group(k, idx, unprotected_tuple_idxs,
                                 distance_dict)
    # calculate the probability of the tuple having the same outcome as the
    # k nearest neighbors in the (un)protected group
    p_1 = calculate_prop_dec_group(decision_attribute_idx, idx_group_prot,
                                   k, tuple, tuples)
    p_2 = calculate_prop_dec_group(decision_attribute_idx, idx_group_unprot,
                                   k, tuple, tuples)
    _diff = p_1 - p_2
    return _diff


def calculate_prop_dec_group(decision_attribute_idx, idx_group, k, tuple,
                             tuples):
    p = sum(1 for idx, _ in idx_group if
            tuples[idx][decision_attribute_idx] == tuple[
                decision_attribute_idx]) / k
    return p


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
    # write json
    json_data = json.dumps(distance_dict)
    open('../data/distance_dict.json', 'w').write(json_data)

    # read the same json
    distance_dict = json.loads(open('../data/distance_dict.json').read())

    # apply the situation testing algorithm with knn
    k = 16
    valid_tuples = knn_situation(k, tuples, attributes, distance_dict,
                                 protected_attributes, decision_attribute)
    print(valid_tuples)
