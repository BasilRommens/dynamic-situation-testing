# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
import numpy as np
from flask import Flask, render_template, redirect, url_for, request

from inout import read_data
from knn import calc_dist_mat, knn_situation
from process import process_all

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__, template_folder='templates')


# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
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
    k = None
    decision_attribute = [None, None]
    sensitive_attribute_var = list()
    sensitive_attribute_val = list()
    ignore_attributes = list()
    # TODO add type conversion for decision and sensitive attributes
    # TODO add check that there is no overlap in decision attribute and sensitive attributes
    # TODO add check that there is no overlap in ignore attributes and sensitive/decision attributes
    for key in form.keys():
        # k nearest neighbors
        if key == 'k' and form[key]:
            k = int(form[key])
        # decision attribute
        elif key == 'decisionAttrVar':
            decision_attribute[0] = form[key]
        elif key == 'decisionAttrVal':
            decision_attribute[1] = form[key]
        # protected attribute
        elif 'sensitiveAttrVar' in key:
            sensitive_attribute_val.append(form[key])
        elif 'sensitiveAttrVar' in key:
            sensitive_attribute_val.append(form[key])
        # attributes to ignore
        elif key == 'ignoreAttr':
            ignore_attributes = form[key]

    r.decision_attribute = {decision_attribute[0]: decision_attribute[1]}
    r.protected_attributes = dict(zip(sensitive_attribute_var, sensitive_attribute_val))
    r.attributes_to_ignore = ignore_attributes

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
    valid_tuples = knn_situation(k, tuples, dist_mat, protected_attributes,
                                 decision_attribute)

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


with app.app_context():
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
    valid_tuples = knn_situation(k, tuples, dist_mat, protected_attributes,
                                 decision_attribute)

# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()
