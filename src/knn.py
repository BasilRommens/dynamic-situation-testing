import time

import numpy as np
import scipy.spatial as scs
import scipy.stats
import sklearn.metrics
from sklearn import preprocessing

from distance import total_distance, ppdist, Metric, transform_tuples
from inout import read_data
from process import process_all
import matplotlib.pyplot as plt
import seaborn as sns


def attr_type_idx(attr_type):
    if attr_type == 'interval':
        return 0
    elif attr_type == 'nominal':
        return 1
    elif attr_type == 'ordinal':
        return 2


def calc_dist_mat(tuples, ranked_values, attribute_types, decision_attribute,
                  protected_attributes):
    _tuples = tuples.copy()  # copy tuples to prevent side effects

    attributes = list(_tuples.columns)
    m_dict = dict()
    s_dict = dict()
    for idx, attribute in enumerate(attributes):
        attribute_type = attribute_types[
            attributes[idx]]  # determine attribute type
        if attribute_type != 'interval':
            continue

        all_interval = _tuples[attributes[idx]].values

        # remove all nan values
        all_interval = all_interval[~np.isnan(all_interval)]

        m_dict[idx] = np.average(all_interval)
        s_dict[idx] = np.std(all_interval)

    if decision_attribute:
        decision_attr_idx = attributes.index(list(decision_attribute.keys())[0])
    else:
        decision_attr_idx = []
    if protected_attributes:
        protected_attr_idcs = [attributes.index(protected_attribute)
                           for protected_attribute in
                           protected_attributes.keys()]
    else:
        protected_attr_idcs = []

    # label encode the entire dataset
    le = preprocessing.LabelEncoder()

    # only select columns that contain non-numerical nominal and ordinal values
    object_tuples = _tuples.select_dtypes(include=['object'])
    object_tuples.replace({None: 'NA'}, inplace=True)
    unique_tuple_values = np.unique(object_tuples.values)
    le.fit(unique_tuple_values)
    _tuples[object_tuples.columns] = object_tuples.apply(le.transform)

    # find the index of the NA value and replace all None values with it
    na_idx = np.where(unique_tuple_values == 'NA')[0]
    if len(na_idx):
        na_idx = na_idx[0]
        _tuples.replace({na_idx: 0}, inplace=True)

    _tuples = transform_tuples(_tuples, _tuples.columns, attribute_types,
                               ranked_values, m_dict, s_dict, decision_attr_idx,
                               protected_attr_idcs)

    if type(na_idx) == int:
        _tuples.replace({na_idx: None}, inplace=True)

    attribute_types = [attribute_types[attribute] for attribute in attributes]
    attribute_types = list(map(attr_type_idx, attribute_types))
    metric = Metric(_tuples.columns, attribute_types, ranked_values, m_dict,
                    s_dict, decision_attr_idx, protected_attr_idcs)

    # convert all columns to float
    temp_tuples = _tuples.values.tolist()
    if np.dtype('float64') in _tuples.dtypes.values:
        temp_tuples = [[None if np.isnan(v) else v for v in t] for t in
                       temp_tuples]
    _tuples = temp_tuples

    start = time.time()
    flat_dist_mat = ppdist(_tuples, metric=metric)
    print(f'ppdist took {time.time() - start} seconds')

    dist_mat = scs.distance.squareform(flat_dist_mat)

    return dist_mat


def is_part_of_protected_group(tuple, protected_attributes, attributes):
    for attribute, values in protected_attributes.items():
        attribute_idx = attributes.index(attribute)
        if tuple[attribute_idx] not in values:
            return False
    return True


def knn_group(k, tuple_idx, group_idxs, dist_mat):
    idx_distance = list()
    for other_tuple_idx in group_idxs:
        # skip the tuple that has the same index as the one we are determining
        # the k nearest neighbors for
        if other_tuple_idx == tuple_idx:
            continue

        distance = dist_mat[tuple_idx, other_tuple_idx]

        idx_distance.append((other_tuple_idx, distance))

    idx_distance = sorted(idx_distance, key=lambda x: x[1])

    k_idx_distance = idx_distance[:k]

    return k_idx_distance


def knn_situation(k, tuples, dist_mat, protected_attributes,
                  decision_attribute):
    attributes = list(tuples.columns)
    # in knn for situation testing first we need to construct the protected and
    # unprotected groups
    protected_tuple_idxs = set()
    unprotected_tuple_idxs = set()
    for idx, tuple in enumerate(tuples.values):
        if is_part_of_protected_group(tuple, protected_attributes,
                                      attributes):
            protected_tuple_idxs.add(idx)
        else:
            unprotected_tuple_idxs.add(idx)

    # now we need to find the k nearest neighbors for each tuple in each group
    decision_attribute_name = list(decision_attribute.keys())[0]
    decision_attribute_value = decision_attribute[decision_attribute_name]
    decision_attribute_idx = attributes.index(decision_attribute_name)
    valid_tuples = list()
    for idx, tuple in enumerate(tuples.values):
        # continue if the idx is in the unprotected group
        if idx in unprotected_tuple_idxs:
            continue
        # if decision of idx is not same as decision attribute then continue
        if tuple[decision_attribute_idx] != decision_attribute_value:
            continue

        _diff, idx_group_prot, idx_group_unprot = diff(decision_attribute_idx,
                                                       dist_mat, idx, k,
                                                       protected_tuple_idxs,
                                                       tuple, tuples,
                                                       unprotected_tuple_idxs)
        valid_tuples.append((idx, _diff, idx_group_prot, idx_group_unprot))
    return valid_tuples


def diff(decision_attribute_idx, dist_mat, idx, k, protected_tuple_idxs,
         tuple, tuples, unprotected_tuple_idxs):
    # find the k nearest neighbors in the (un)protected group
    idx_group_prot = knn_group(k, idx, protected_tuple_idxs, dist_mat)
    idx_group_unprot = knn_group(k, idx, unprotected_tuple_idxs,
                                 dist_mat)
    # calculate the probability of the tuple having the same outcome as the
    # k nearest neighbors in the (un)protected group
    p_1 = calc_prop_dec_group(decision_attribute_idx, idx_group_prot,
                              k, tuple, tuples)
    p_2 = calc_prop_dec_group(decision_attribute_idx, idx_group_unprot,
                              k, tuple, tuples)
    _diff = p_1 - p_2
    return _diff, idx_group_prot, idx_group_unprot


def calc_prop_dec_group(decision_attribute_idx, idx_group, k, tuple, tuples):
    p = sum(1 for idx, _ in idx_group if
            tuples.iloc[idx][decision_attribute_idx] == tuple[
                decision_attribute_idx]) / k
    return p


if __name__ == "__main__":
    # read the data from the csv and json file
    r = read_data('data/german_credit_data.json',
                  'data/german_credit_data_class.csv')

    # process the data
    tuples, ranked_values, decision_attribute = process_all(r)
    # determine the distances
    protected_attributes = {"Sex": ["female"]}
    dist_mat = calc_dist_mat(tuples, ranked_values, r.attribute_types,
                             decision_attribute,
                             protected_attributes)
    # write dump
    dist_mat.dump('data/dist_mat.dump')
    # read the same dump
    dist_mat = np.load('data/dist_mat.dump', allow_pickle=True)

    # apply the situation testing algorithm with knn
    k = 4
    valid_tuples = knn_situation(k, tuples, dist_mat, protected_attributes,
                                 decision_attribute)
    print(valid_tuples)
    sns.kdeplot()
    plt.show()
