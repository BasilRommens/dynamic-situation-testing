import time

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc

from dimred import dimred_mds, dimred_pca, dimred_umap
from globals import colors_string
from helper import create_segment_name
from inout import read_data
from knn import calc_dist_mat, knn_situation
from matrix import Matrix
from process import process_all, make_numeric
from tuples import get_tuple_discrimination_type, get_tuples_with_attr
from viz import dynamic

click_shapes = list()

data = dict()
data['size_lim'] = 1000


def process_fig_data(path, json_fname, csv_fname, protected_attributes,
                     ignore_cols, k=4):
    # READ AND BASIC PROCESSING
    # read the data from the csv and json file
    r = read_data(path + json_fname, path + csv_fname)
    r.df = r.df.head(data['size_lim'])
    print('read data')

    # process the data
    # TODO add NA type values
    tuples, ranked_values, decision_attribute = process_all(r)
    print('processed data')

    # KNN SITUATION TESTING
    # determine the distances
    start = time.time()
    dist_mat = calc_dist_mat(tuples, ranked_values, r.attribute_types,
                             decision_attribute, protected_attributes)
    print(f'calculated distance matrix in {time.time() - start} seconds')
    print('calculated distance matrix')

    # apply the situation testing algorithm with knn
    valid_tuples = knn_situation(k, tuples, dist_mat, protected_attributes,
                                 decision_attribute)
    print('calculated knn situation testing tuples')

    return valid_tuples, tuples, protected_attributes, ignore_cols, r, dist_mat, \
        decision_attribute


def calc_pre_plotting(valid_tuples, tuples, protected_attributes, ignore_cols,
                      r, dist_mat, decision_attribute, path, csv_fname):
    # get the markers for the different tuple types according to the situation
    # testing algorithm
    all_tuple_markers = get_tuple_discrimination_type(valid_tuples, tuples)

    # read the complete dataset in original form
    df = pd.read_csv(path + csv_fname)
    df = df.head(data['size_lim'])
    og_df = df.copy()

    # get the tuples with the sensitive attributes
    sensitive_tuple_idxs = get_tuples_with_attr(df, protected_attributes)
    decision_attr = list(decision_attribute.keys())[0]
    og_decision_attr_vals = df[decision_attr].values

    # convert all values to string
    og_decision_attr_vals = [str(x) for x in og_decision_attr_vals]

    df[decision_attr] = df[decision_attr].apply(
        lambda x: 0 if x == decision_attribute[decision_attr] else 1)
    df = make_numeric(df, r.ordinal_attribute_values)
    class_col = df[decision_attr]
    class_colors = [colors_string[1] if c == 1 else colors_string[2] for c in
                    class_col]

    # create a zip of the class colors and the decision attribute values
    class_color_names = zip(class_colors, og_decision_attr_vals)

    # get only the unique pairs to construct a dict
    class_color_names_dict = dict(set(class_color_names))

    df = df.drop(columns=ignore_cols)

    # bounds for different columns
    segment_dict_ls = []

    # symbol map
    symbol_map = {'negative discrimination': 'line-ew-open',
                  'neutral': 'circle',
                  'positive discrimination': 'cross-thin-open',
                  'sensitive': 'circle'}

    # construct a distance matrix
    m = Matrix(df, heatmap_viz=False, feat_names=df.columns, DD=dist_mat)
    dist_mat = m.merged_matrix()

    # dimensionality reduction using distance matrix
    n_d_pts = len(df)

    # different dimensionality reduction techniques
    dim_red_samples = dimred_mds(dist_mat)
    # dimensionality reduce first with PCA
    # dim_red_samples = dimred_pca(dist_mat, dims=8)
    # print(f'dimred in {time.time() - start} seconds')
    # dim_red_samples = dimred_umap(dim_red_samples, dense=True)
    # print(f'dimred in {time.time() - start} seconds')

    n_feat = dim_red_samples.shape[0] - n_d_pts

    # construct the scatter plot from data points and feature points
    # data points + extra info
    data_pts = pd.DataFrame(dim_red_samples[:n_d_pts], columns=['x', 'y'])
    data_pts['type'] = 0
    data_pts = pd.concat([data_pts, og_df], axis=1)

    # feature points
    feat_pts = pd.DataFrame(dim_red_samples[n_d_pts:], columns=['x', 'y'])
    feat_pts['type'] = 2

    return all_tuple_markers, df, og_df, sensitive_tuple_idxs, segment_dict_ls, \
        symbol_map, n_d_pts, n_feat, data_pts, feat_pts, class_colors, class_color_names_dict


def sort_color_marker_pair(pair):
    match pair[0]:
        case 'positive discrimination':
            return 0
        case 'negative discrimination':
            return 1
        case 'sensitive':
            return 2
        case 'neutral':
            return 3


def calc_plot(all_tuple_markers, df, og_df, sensitive_tuple_idxs,
              segment_dict_ls, symbol_map, n_d_pts, n_feat, data_pts, feat_pts,
              class_colors, class_colors_name_dict):
    # figure to which to add all the scatter plot data
    scatter_data = go.FigureWidget()

    # split the data points according to a pair of class colors and symbols
    colors_decision = list(
        map(lambda x: class_colors_name_dict[x], class_colors))
    rev_class_colors_name_dict = {v: k for k, v in
                                  class_colors_name_dict.items()}
    markers_colors_zip = list(zip(all_tuple_markers, colors_decision))
    unique_marker_color_pairs = list(set(markers_colors_zip))

    # sort unique marker color pairs as it makes it easier in the plot
    unique_marker_color_pairs = sorted(unique_marker_color_pairs,
                                       key=sort_color_marker_pair)

    for unique_marker_color_pair in unique_marker_color_pairs:
        # determine the name of the marker color pair
        marker_color_pair_name = ",".join(unique_marker_color_pair)

        # find the indices of all the tuples with the same marker and color
        indices = [i for i, x in enumerate(markers_colors_zip) if
                   x == unique_marker_color_pair]

        # get the data points with the same marker and color
        data_pts_same_marker_color = data_pts.iloc[indices]

        # get the colors with the same marker and color
        colors_same_marker_color = [rev_class_colors_name_dict[
                                        unique_marker_color_pair[1]]] * len(
            indices)

        # get the markers with the same marker and color
        markers_same_marker_color = [unique_marker_color_pair[0]] * len(indices)

        # create the scatter plot for that unique marker color pair
        _scatter_data = dynamic.scatter_plot(data_pts_same_marker_color,
                                             hover_data=list(
                                                 data_pts_same_marker_color.columns),
                                             symbol=markers_same_marker_color,
                                             symbol_map=symbol_map,
                                             color=np.array(
                                                 colors_same_marker_color),
                                             size=[10] * len(indices), width=2,
                                             name=marker_color_pair_name)
        # add this to the scatter data figure
        scatter_data = dynamic.combine_plots(scatter_data, _scatter_data)

    # # create a basic scatter plot of data points
    # scatter_data = dynamic.scatter_plot(data_pts,
    #                                     hover_data=list(og_df.columns),
    #                                     symbol=all_tuple_markers,
    #                                     symbol_map=symbol_map,
    #                                     color=np.array(class_colors),
    #                                     size=[10] * n_d_pts, width=2,
    #                                     name='Tuples')
    # scatter plot of feature points
    scatter_feature = dynamic.scatter_plot(feat_pts, text=list(df.columns),
                                           size=[20] * n_feat,
                                           text_position='top center',
                                           color=['black'] * n_feat,
                                           name='Features')
    # scatter plot of sensitive tuples
    scatter_sensitive = dynamic.scatter_plot(
        data_pts.iloc[sensitive_tuple_idxs],
        symbol=['sensitive'] * len(sensitive_tuple_idxs),
        symbol_map={'sensitive': 'circle'},
        color=['grey'] * len(sensitive_tuple_idxs),
        size=[20] * len(sensitive_tuple_idxs),
        name='Sensitive Tuples')

    # combine the two scatter plots
    scatter_final = dynamic.combine_plots(scatter_data, scatter_feature)
    scatter_final = dynamic.combine_plots(scatter_sensitive, scatter_final)

    # create the segments using seaborn and translate to plotly
    init_segment = go.FigureWidget()
    # bw_weigths = calc_bw_weights(data_pts.values, 1)
    contour_names = list()
    used_colors_idcs = list()
    for i, segment_dict in enumerate(segment_dict_ls):
        # add the idx to used colors
        used_colors_idcs.append(i)
        # create the name of the segment
        segment_name = create_segment_name(segment_dict)
        contour_names.append(segment_name)

        # create the plotly segment
        kde_segment = dynamic.kde_segment(data_pts.copy(), segment_dict,
                                          colors_string[i], segment_name)

        # combine the segments
        init_segment = dynamic.combine_plots(init_segment, kde_segment)

    # combine the segments with the scatter plot
    scatter_final = dynamic.combine_plots(init_segment, scatter_final)

    # set the theme of the plot
    scatter_final = dynamic.set_theme(scatter_final, 'plotly_white')

    return scatter_final, contour_names, segment_dict_ls, used_colors_idcs


def calc_german_credit_fig():
    global data

    # german credit dataset
    path = 'data/'
    data['path'] = path
    json_fname = 'german_credit_data.json'
    csv_fname = 'german_credit_data_class.csv'
    data['csv_fname'] = csv_fname
    protected_attributes = {"Sex": ["female"]}
    ignore_cols = ['Class']
    k = 4

    return calc_fig(path, json_fname, csv_fname, protected_attributes,
                    ignore_cols, k)


def calc_adult_fig():
    global data

    # german credit dataset
    path = 'data/'
    data['path'] = path
    json_fname = 'adult.json'
    csv_fname = 'adult.csv'
    data['csv_fname'] = csv_fname
    protected_attributes = {"sex": ["Female"]}
    ignore_cols = ['class', 'education']
    k = 4

    return calc_fig(path, json_fname, csv_fname, protected_attributes,
                    ignore_cols, k)


def calc_table(data_pts, valid_tuples):
    """
    Create a table of the valid tuples
    :param data_pts: all data points
    :param valid_tuples: all the valid tuple values
    :return: a dict that represents the table
    """
    # table list
    table_ls = list()

    # iterate over all valid tuples
    for i, valid_tuple in enumerate(valid_tuples):
        data_pt_idx = valid_tuple[0]
        score = valid_tuple[1]
        # TODO do something with this information
        protected = valid_tuple[2]
        unprotected = valid_tuple[3]

        # get the data point
        data_pt = data_pts.iloc[data_pt_idx].to_dict()

        # add score
        data_pt['score'] = score

        table_ls.append(data_pt)

    return table_ls


def get_contour_html(contour_names):
    title = html.H3('Remove Contours', className='mx-2')
    contour_html = [html.Form(dbc.Input(id=f'contour_{i}',
                                        value=contour_name,
                                        class_name="btn btn-danger mr-2 my-2",
                                        type="submit"),
                              action=f'/remove_contour/{i}',
                              method='POST',
                              id=f'form_contour_{i}', )
                    for i, contour_name in enumerate(contour_names)]

    if len(contour_html):
        contour_html.insert(0, title)
    else:
        contour_html = [title, html.P('No contours to remove',
                                      className='mx-2 text-muted')]

    return contour_html


def get_feature_form(feature, feature_value, feature_type, ordinal_attr_dict):
    input_id = feature + '_input'
    if feature_type == 'interval':
        # get the minimum and the maximum value of the slider
        _min = min(feature_value)
        _max = max(feature_value)

        # get the step size
        return html.Div([dbc.Input(id=input_id + '1', type='number', value=_min,
                                   min=_min, max=_max, name=feature),
                         dbc.Input(id=input_id + '2', type='number', value=_max,
                                   min=_min, max=_max, name=feature + '2')])
    if feature_type == 'ordinal':
        inverted_dict = {v: k for k, v in ordinal_attr_dict[feature].items() if
                         not np.isnan(v)}
        new_options = list()
        for option in feature_value:
            if option not in inverted_dict:
                continue
            new_option = {'label': inverted_dict[option], 'value': option}
            new_options.append(new_option)
        return dbc.Select(id=input_id, options=new_options,
                          value=feature_value[0], name=feature)
    if feature_type == 'nominal':
        return dbc.Select(id=input_id, options=feature_value,
                          value=feature_value[0], name=feature)


def construct_contour_form(features, feature_values, feature_types,
                           ordinal_attr_dict):
    options = [{'label': feature, 'value': feature} for feature in features]
    option_forms = [
        html.Div(
            id=feature,
            children=[
                html.Label(feature, htmlFor=feature + '_input'),
                get_feature_form(feature, feature_values[idx],
                                 feature_types[idx], ordinal_attr_dict),
            ]) for idx, feature in enumerate(features)
    ]
    return [
        html.Div(dbc.Select(id='feature', options=options, ), className='mb-2'),
        html.Div(id='featureOptions'),
        html.Button('Add', type='submit',
                    className='btn btn-success mt-2')], option_forms


def calc_fig(path, json_fname, csv_fname, protected_attributes, ignore_cols, k):
    # process the data for the figure
    fig_data = process_fig_data(path, json_fname, csv_fname,
                                protected_attributes, ignore_cols, k)
    print('processed fig data')

    # calculate the data needed for the plot
    preplotting_data = calc_pre_plotting(*fig_data, path, csv_fname)
    print('calculated preplotting data')

    # plot the plot
    scatter_final, contour_names, contours, used_colors_idcs = calc_plot(
        *preplotting_data)
    print('calculated plot')

    # get the contour html
    contour_html = get_contour_html(contour_names)
    print('fetched contour html')

    # generate the form for the contours
    # only take features that are not ignored
    # (aka not in protected attr)
    df = fig_data[1].copy()
    df.drop(fig_data[2].keys(), axis=1, inplace=True)

    ordinal_attr_dict = fig_data[4].ordinal_attribute_values
    features = df.columns
    feature_values = [df[c].unique() for c in features]
    attr_types_dict = fig_data[4].attribute_types
    feature_types = [attr_types_dict[feature] for feature in features]
    contour_form, contour_form_options = construct_contour_form(features,
                                                                feature_values,
                                                                feature_types,
                                                                ordinal_attr_dict)
    print('generated contour form')

    table_ls = calc_table(preplotting_data[8], fig_data[0])
    print('calculated table')

    return scatter_final, preplotting_data[8], fig_data[
        0], table_ls, contour_names, contours, contour_html, contour_form, \
        contour_form_options, list(
        features), attr_types_dict, used_colors_idcs, ordinal_attr_dict


# calculate the figure to be used
fig, data_pts, valid_tuples, table_ls, contour_names, contours, contour_html, \
    contour_form, contour_form_options, features, attr_types_dict, \
    used_colors_idcs, ordinal_attr_dict = calc_german_credit_fig()
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
data['csv_fname'] = None
