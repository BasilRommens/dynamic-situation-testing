import os

import numpy as np
import json

import pandas as pd
from flask import current_app as app, redirect
from flask import render_template, request

from globals import colors_string
from helper import create_segment_name
from inout import read_data
from knn import calc_dist_mat, knn_situation
from process import process_all
from setup import calc_fig, data, get_contour_html
from werkzeug.utils import secure_filename
from viz import dynamic


@app.route('/')
def home():
    return redirect('/upload')

    global data

    valid_tuples = data['valid_tuples']
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


@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_post():
    global data

    csv_f = request.files['csv-file']
    csv_fname = secure_filename(csv_f.filename)
    csv_path_fname = os.path.join(app.config['UPLOAD_FOLDER'], csv_fname)
    csv_f.save(csv_path_fname)

    json_f = request.files['json-file']
    json_fname = secure_filename(json_f.filename)
    json_path_fname = os.path.join(app.config['UPLOAD_FOLDER'], json_fname)
    json_f.save(json_path_fname)

    # TODO make this cleaner
    # read the data from the csv and json file
    data['r'] = read_data(json_path_fname, csv_path_fname)
    data['r'].df = data['r'].df.head(data['size_lim'])

    # process the data
    data['tuples'], data['ranked_values'], data['decision_attribute'] = \
        process_all(data['r'])
    tuples = data['tuples']

    # determine the distances
    protected_attr_name = tuples.columns[0]
    protected_attr_val = tuples[protected_attr_name][0]
    data['protected_attributes'] = {protected_attr_name: [protected_attr_val]}
    data['dist_mat'] = calc_dist_mat(data['tuples'], data['ranked_values'],
                                     data['r'].attribute_types,
                                     data['decision_attribute'],
                                     data['protected_attributes'])

    # apply the situation testing algorithm with knn
    data['k'] = 4
    data['valid_tuples'] = knn_situation(data['k'], data['tuples'],
                                         data['dist_mat'],
                                         data['protected_attributes'],
                                         data['decision_attribute'])
    data['csv_fname'] = csv_fname
    data['path'] = app.config['UPLOAD_FOLDER']

    # remove unused elements in UI
    if 'neighbor_tbl' in data:
        del data['neighbor_tbl']
    if 'neighbor_col' in data:
        del data['neighbor_col']

    data['plot'] = False

    return redirect('/dynamic')


@app.route('/new_contour', methods=['POST'])
def add_contour():
    global data
    # read the form
    form = request.form

    # if the form is empty return
    if not len(form):
        return redirect('/dashapp')

    # feature name
    feat_name = list(form.keys())[0]

    # get the data type of the new contour
    data_type = data['feature_types_dict'][feat_name]

    contour = None
    if data_type == 'nominal':
        val = form[feat_name]
        # create the contour in dict form
        contour = {feat_name: val}

        # get the name of the new contour
        contour_name = create_segment_name(contour)
        data['contour_names'].append(contour_name)
    elif data_type == 'ordinal':
        val = form[feat_name]
        # create the contour in dict form
        contour = {feat_name: val}

        # get the ordinal attribute dict
        ordinal_attr_dict = data['ordinal_attr_dict']
        # invert the dict
        inverted_dict = {v: k for k, v in ordinal_attr_dict[feat_name].items()}

        # get an inverted dict to convert the value to readable value
        # get the name of the new contour
        contour_name = create_segment_name({feat_name: inverted_dict[int(val)]})
        data['contour_names'].append(contour_name)
    elif data_type == 'interval':
        _range = list()
        for range_val in form.values():
            _range.append(int(range_val))
        _range = sorted(_range)
        # create the contour in dict form
        contour = {feat_name: _range}

        # get the name of the new contour
        contour_name = create_segment_name(contour)
        data['contour_names'].append(contour_name)

    # add the contour to the contours in data
    data['contours'].append(contour)

    # create the contour html for contour removal
    data['contour_html'] = get_contour_html(data['contour_names'])

    # create the segments using seaborn and translate to plotly
    # all possible colors
    used_colors_idcs = set(data['used_colors_idcs'])
    all_colors = set([i for i in range(len(colors_string))])
    possible_colors = list(all_colors.difference(used_colors_idcs))
    new_idx = possible_colors[0]
    used_colors_idcs.add(new_idx)
    data['used_colors_idcs'] = list(used_colors_idcs)

    # create the plotly segment
    kde_segment = dynamic.kde_segment(data['data_pts'].copy(), contour,
                                      colors_string[new_idx], contour_name)

    # prepend the contour to the figure
    new_fig = dynamic.combine_plots(kde_segment, data['fig'])

    # set plotly white theme
    data['fig'] = dynamic.set_theme(new_fig, 'plotly_white')

    return redirect('/dashapp')


@app.route('/dynamic')
def dynamic_site():
    attrs = list(data['tuples'].columns)
    return render_template('situation-testing-form.html', attrs=attrs)


@app.route('/dynamic', methods=['POST'])
def dynamic_post():
    global data
    r = data['r']
    path = data['path']
    csv_fname = data['csv_fname']
    form = request.form
    k, protected_attrs, decision_attr, attrs_to_ignore = handle_situation_form(
        form, r)

    # create new json file
    new_json_data = {"attribute_types":
                         {k: r.attribute_types[k]
                          for k in r.attribute_types.keys()},
                     "ordinal_attribute_values":
                         {k: r.ordinal_attribute_values[k]
                          for k in r.ordinal_attribute_values.keys()},
                     "attributes_to_ignore":
                         attrs_to_ignore,
                     "decision_attribute":
                         decision_attr,
                     }

    json_fname = '~temp.json'
    with open(path + json_fname, 'w') as f:
        json.dump(new_json_data, f)

    fig, data_pts, valid_tuples, table_ls, contour_names, contours, contour_html, \
        contour_form, contour_form_options, features, attr_types_dict, \
        used_colors_idcs, ordinal_attr_dict = calc_fig(path, json_fname,
                                                       csv_fname,
                                                       protected_attrs,
                                                       attrs_to_ignore, k)

    data['table'] = table_ls
    data['fig'] = fig
    data['data_pts'] = data_pts
    data['contour_names'] = contour_names
    data['contours'] = contours
    data['contour_html'] = contour_html
    data['contour_form'] = contour_form
    data['contour_form_options'] = contour_form_options
    data['used_colors_idcs'] = used_colors_idcs
    data['features'] = features
    data['feature_types_dict'] = attr_types_dict
    data['ordinal_attr_dict'] = ordinal_attr_dict
    data['valid_tuples'] = valid_tuples
    data['click_shapes'] = list()
    data['plot'] = True

    return redirect('/dashapp')


@app.route('/remove_contour/<int:contour_id>', methods=['POST'])
def remove_contours(contour_id: int):
    global data

    # contour name
    contour_name = data['contour_names'][contour_id]

    # remove the contours such that they aren't in a list anymore
    data['contour_names'].pop(contour_id)
    data['contours'].pop(contour_id)

    # remove the contours from the figure
    idx_list = list()
    for figure_el_idx, figure_el in enumerate(data['fig']['data']):
        if 'legendgroup' not in figure_el:
            continue
        if figure_el['legendgroup'] == contour_name:
            color_idx = colors_string.index(figure_el['fillcolor'])
            if color_idx in data['used_colors_idcs']:
                data['used_colors_idcs'].remove(color_idx)
            idx_list.append(figure_el_idx)

    # get the max index from all the figure elements
    max_idx = len(data['fig']['data'])
    # only keep the elements that are not in the index list
    new_fig = list()
    for idx in range(max_idx):
        if idx in idx_list:
            continue
        new_fig.append(data['fig']['data'][idx])

    # store the new figure
    data['fig']['data'] = tuple(new_fig)

    # set plotly white theme
    data['fig'] = dynamic.set_theme(data['fig'], 'plotly_white')

    # regenerate the list of buttons in the dash app
    data['contour_html'] = get_contour_html(data['contour_names'])

    # redirect to /dashapp to show the figure
    return redirect('/dashapp')


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
    decision_attr = {decision_attr[0]: decision_attr[1]}
    attrs_to_ignore = ignore_attrs
    protected_attrs = dict(zip(sensitive_attr_vars, sensitive_attr_vals))

    return k, protected_attrs, decision_attr, attrs_to_ignore


# read the data from the csv and json file
path = 'data/'
json_fname = 'german_credit_data.json'
csv_fname = 'german_credit_data_class.csv'
r = read_data(path + json_fname, path + csv_fname)
data['r'] = r

# process the data
tuples, ranked_values, decision_attribute = process_all(r)
data['tuples'] = tuples
data['ranked_values'] = ranked_values
data['decision_attribute'] = decision_attribute

# determine the distances
protected_attributes = {"Sex": ["female"]}
data['protected_attributes'] = protected_attributes
dist_mat = calc_dist_mat(tuples, ranked_values, r.attribute_types,
                         decision_attribute, protected_attributes)
data['dist_mat'] = dist_mat

# apply the situation testing algorithm with knn
k = 4
data['k'] = k
valid_tuples = knn_situation(k, tuples, dist_mat, protected_attributes,
                             decision_attribute)
data['valid_tuples'] = valid_tuples
