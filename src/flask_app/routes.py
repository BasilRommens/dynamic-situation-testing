import numpy as np
from flask import current_app as app
from flask import render_template, request

from inout import read_data
from knn import calc_dist_mat, knn_situation
from process import process_all


@app.route('/')
def home():
    global valid_tuples
    headers = sorted(list(tuples.columns))
    headers.insert(0, 'Score')

    new_tuples = list()
    # create a new tuple with the score and the k nearest neighbors along with
    # its own values
    for tuple_idx, score, k_prot, k_unprot in valid_tuples:
        # sort the tuple by the attribute name and then only take the values
        new_tuple = list(map(lambda x: x[1], sorted(
            list(tuples.iloc[tuple_idx].to_dict().items()),
            key=lambda x: x[0])))
        # insert the score at the beginning of the tuple
        new_tuple.insert(0, score)

        # sort the tuple by the attribute name and then only take the values
        # but now for the k nearest neighbors of the protected group
        new_k_prot = list()
        for prot_idx, _ in k_prot:
            new_prot_el = list(map(lambda x: x[1], sorted(
                list(tuples.iloc[prot_idx].to_dict().items()),
                key=lambda x: x[0])))
            new_prot_el.insert(0, 'protected')
            new_k_prot.append(new_prot_el)

        # sort the tuple by the attribute name and then only take the values
        # but now for the k nearest neighbors of the unprotected group
        new_k_unprot = list()
        for unprot_idx, _ in k_unprot:
            new_unprot_el = list(map(lambda x: x[1], sorted(
                list(tuples.iloc[unprot_idx].to_dict().items()),
                key=lambda x: x[0])))
            new_unprot_el.insert(0, 'unprotected')
            new_k_unprot.append(new_unprot_el)

        # create a triple tuple
        new_tuple = (new_tuple, new_k_prot, new_k_unprot)

        # add the new tuple
        new_tuples.append(new_tuple)

    return render_template('home.html', items=new_tuples, headers=headers)


@app.route('/explore')
def explore():
    return render_template('explore.html')


@app.route('/discriminate')
def discriminate():
    return render_template('discriminate.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/dynamic')
def dynamic():
    attrs = list(tuples.columns)
    return render_template('dynamic.html', attrs=attrs)


@app.route('/dynamic', methods=['POST'])
def dynamic_post():
    global r
    form = request.form
    # TODO add check that there is no overlap in decision attribute and sensitive attributes
    # TODO add check that there is no overlap in ignore attributes and sensitive/decision attributes
    k, protected_attrs = handle_situation_form(form, r)

    # process the data
    tuples, ranked_values, decision_attr = process_all(r)
    # determine the distances
    dist_mat = calc_dist_mat(tuples, ranked_values, r.attribute_types,
                             decision_attr, protected_attrs)

    # apply the situation testing algorithm with knn
    valid_tuples = knn_situation(k, tuples, dist_mat, protected_attrs,
                                 decision_attr)

    headers = sorted(list(tuples.columns))
    headers.insert(0, 'Score')

    new_tuples = list()
    # create a new tuple with the score and the k nearest neighbors along with
    # its own values
    for tuple_idx, score, k_prot, k_unprot in valid_tuples:
        # sort the tuple by the attribute name and then only take the values
        new_tuple = list(map(lambda x: x[1], sorted(
            list(tuples.iloc[tuple_idx].to_dict().items()),
            key=lambda x: x[0])))
        new_tuple.insert(0, score)  # insert the score at the beginning

        # get the k closest neighbors of both protected and unprotected group
        new_k_prot = get_knn_els(k_prot, tuples, 'protected')
        new_k_unprot = get_knn_els(k_unprot, tuples, 'unprotected')

        # create a triple tuple
        new_tuple = (new_tuple, new_k_prot, new_k_unprot)

        # add the new tuple
        new_tuples.append(new_tuple)

    return render_template('home.html', items=new_tuples, headers=headers)


def get_knn_els(knn_els, tuples, el_name):
    # sort the tuple by the attribute name and then only take the values
    # for the k nearest neighbors
    new_knn_els = list()
    for knn_el_idx, _ in knn_els:
        knn_el = list(map(lambda x: x[1], sorted(
            list(tuples.iloc[knn_el_idx].to_dict().items()),
            key=lambda x: x[0])))
        knn_el.insert(0, el_name)
        new_knn_els.append(knn_el)
    return new_knn_els


def handle_situation_form(form, r):
    k = None
    decision_attr = [None, None]
    sensitive_attr_vars = list()
    sensitive_attr_vals = list()
    ignore_attrs = list()
    for key in form.keys():
        # k nearest neighbors
        if key == 'k' and form[key]:
            k = int(form[key])
        # decision attribute
        elif key == 'decisionAttrVar':
            decision_attr[0] = form[key]
        elif key == 'decisionAttrVal':
            # do type check of the decision attribute
            attr_type = r.attribute_types[decision_attr[0]]
            decision_attr[1] = int(form[key]) if attr_type == 'interval' else \
                form[key]
        # protected attribute
        elif 'sensitiveAttrVar' in key:
            sensitive_attr_vars.append(form[key])
        elif 'sensitiveAttrVal' in key:
            # do type check of the decision attribute
            attr_type = r.attribute_types[sensitive_attr_vars[-1]]
            sensitive_attr_val = int(form[key]) if attr_type == 'interval' else \
                form[key]
            sensitive_attr_vals.append([sensitive_attr_val])
        # attributes to ignore
        elif key == 'ignoreAttr':
            ignore_attrs = form[key]

    # set the remaining return values
    r.decision_attribute = {decision_attr[0]: decision_attr[1]}
    r.attributes_to_ignore = ignore_attrs
    protected_attrs = dict(zip(sensitive_attr_vars, sensitive_attr_vals))

    return k, protected_attrs


# read the data from the csv and json file
r = read_data('data/german_credit_data.json',
              'data/german_credit_data_class.csv')

# process the data
tuples, ranked_values, decision_attribute = process_all(r)
# determine the distances
protected_attributes = {"Sex": ["female"]}
dist_mat = calc_dist_mat(tuples, ranked_values, r.attribute_types,
                         decision_attribute, protected_attributes)
# write dump
dist_mat.dump('data/dist_mat.dump')
# read the same dump
dist_mat = np.load('data/dist_mat.dump', allow_pickle=True)

# apply the situation testing algorithm with knn
k = 4
valid_tuples = knn_situation(k, tuples, dist_mat,
                             protected_attributes,
                             decision_attribute)
