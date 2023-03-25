import itertools

import plotly.graph_objects as go
import seaborn as sns
import networkx as nx
import os
import scipy.spatial as scs
import numpy as np
import pandas as pd
from dash import Dash, html, dcc, Output, Input, no_update, State
from sklearn.manifold import MDS, TSNE
import matplotlib.pyplot as plt
import time

from dataset import create_gaussian
from globals import colors_string
from helper import create_segment_name
from inout import read_data
from knn import calc_dist_mat, knn_situation
from matrix import Matrix
from process import make_numeric, process_all
from stress import total_stress, total_knn_stress
from tuples import get_tuple_discrimination_type, get_tuples_with_attr
from visualize import visualize_kde
from viz import dynamic


def dimred_glimmer(dist_mat):
    """
    dimensionality reduce the distance matrix using glimmer
    :param dist_mat: the distance matrix
    :return: all the dim reduced points
    """
    # convert the distance matrix to a csv file
    df = pd.DataFrame(dist_mat)
    # add the necessary rows of text to the file
    n_dims = len(df.columns)

    # create the initial string
    col_name_str = ','.join([str(i) for i in range(n_dims)])
    type_str = ','.join(['DOUBLE'] * n_dims)
    header_str = col_name_str + '\n' + type_str + '\n'

    path = 'data/'
    ifname = '~in.csv'
    ofname = '~out.csv'
    glimmer_fname = 'glimmer'
    # write header string to ~in.csv
    with open(path + ifname, 'w+') as f:
        f.write(header_str)
    df.to_csv('data/~in.csv', index=False, header=False, mode='a+')

    # dimensionality reduce matrix with glimmer
    os.system(f'{path + glimmer_fname} {path + ifname} {path + ofname} csv')

    # read matrix in pandas
    df = pd.read_csv(path + ofname, skiprows=1)
    # then convert to numpy array
    arr = df.to_numpy()

    return arr


def dimred_graph(dist_mat):
    """
    dimensionality reduce the distance matrix using a force-directed graph
    layout algorithm
    :param dist_mat: the distance matrix to use for edge strengths
    :return: the dimensionality reduced points
    """
    # convert the distance matrix into a graph and determine the layout
    new_dist_mat = np.reciprocal(dist_mat)  # low dist = high weight
    new_dist_mat = np.nan_to_num(new_dist_mat, copy=True, nan=0.0, posinf=None,
                                 neginf=None)  # replace nan with 0

    G = nx.from_numpy_array(new_dist_mat)  # create the graph
    pos = nx.spring_layout(G, iterations=1_000)  # determine the node positions

    # create a proper numpy array out of the positions
    arr = np.array([val for val in pos.values()])

    return arr


def dimred_mds(dist_mat):
    df = pd.DataFrame(dist_mat)
    model = MDS(n_components=2, dissimilarity='precomputed')

    arr = model.fit_transform(df)
    print(model.stress_)  # print the stress to show how stressed the layout is

    return arr


def dimred_tsne(dist_mat):
    df = pd.DataFrame(dist_mat)
    model = TSNE(n_components=2, perplexity=20)
    arr = model.fit_transform(df)
    return arr


def calc_bw_weights(data_pts, bw):
    """
    calculate the weights for the kernel density estimation per point
    :param data_pts: the data points to use
    :param bw: the fixed bandwidth to use
    :return: a list of bandwidth weights
    """
    # calculate the local density of each point
    f_p_all = np.zeros(len(data_pts))
    for i, data_pt in enumerate(data_pts):
        f_p = 0
        for data_pt2 in data_pts:
            f_p += np.linalg.norm(data_pt - data_pt2)
        f_p = f_p / len(data_pts)
        f_p_all[i] = f_p

    # calculate the geometric mean of the local densities
    G = np.exp(np.mean(np.log(f_p_all)))
    # calculate smoothing parameter for each point
    lambda_p = np.power(G / f_p_all, 2)

    # calculate the bandwidth weights
    weights = bw * lambda_p

    return weights


if __name__ == '__main__':
    # german credit dataset
    path = 'data/'
    json_fname = 'german_credit_data.json'
    csv_fname = 'german_credit_data_class.csv'
    protected_attributes = {"Sex": ["female"]}
    ignore_cols = ['Class', 'Sex']
    # # adult dataset
    # path = 'data/'
    # json_fname = 'adult.json'
    # csv_fname = 'adult.csv'
    # protected_attributes = {"sex": ["female"]}
    # ignore_cols = ['class', 'sex']

    # read the data from the csv and json file
    r = read_data(path + json_fname, path + csv_fname)
    print('read data')

    # process the data
    # TODO add NA type values
    tuples, ranked_values, decision_attribute = process_all(r)
    print('processed data')
    # determine the distances
    start = time.time()
    dist_mat = calc_dist_mat(tuples, ranked_values, r.attribute_types,
                             decision_attribute, protected_attributes)
    print(f'calculated distance matrix in {time.time() - start} seconds')
    print('calculated distance matrix')
    # write dump
    dist_mat.dump('data/dist_mat.dump')
    # read the same dump
    dist_mat = np.load('data/dist_mat.dump', allow_pickle=True)

    # apply the situation testing algorithm with knn
    k = 4
    valid_tuples = knn_situation(k, tuples, dist_mat, protected_attributes,
                                 decision_attribute)
    print('calculated knn situation testing')

    # get the markers for the different tuple types according to the situation
    # testing algorithm
    all_tuple_markers = get_tuple_discrimination_type(valid_tuples, tuples)

    # german credit dataset
    df = pd.read_csv('data/german_credit_data_class.csv')
    og_df = df.copy()
    sensitive_tuple_idxs = get_tuples_with_attr(df, protected_attributes)
    df = make_numeric(df, r.ordinal_attribute_values)
    class_col = df['Class']
    class_colors = [colors_string[1] if c == 1.0 else colors_string[2] for c in
                    class_col]
    df = df.drop(columns=ignore_cols)
    # bounds for different columns
    segment_dict_ls = [{'Credit amount': (6000, 20_000)}]
    # symbol map
    symbol_map = {'negative discrimination': 'line-ew-open',
                  'neutral': 'circle',
                  'positive discrimination': 'cross-thin-open',
                  'sensitive': 'circle'}

    # # auto mpg data set
    # ignore_cols = ['displacement']
    # df = pd.read_csv('data/auto-mpg.csv')
    # print(df)
    # df = make_numeric(df)
    # og_df = df.copy()
    # df = df.drop(columns=ignore_cols)
    # # bounds for different columns
    # segment_dict_ls = [{'weight': (4000, 6000)},
    #                    {'mpg': (43, 46), 'horsepower': (0, 100)},
    #                    {'year': (84, 90)}]

    # # gaussian data set
    # gaussian_sets = list()
    # np.random.seed(42)
    # for i in range(6):
    #     gaussian_set, anchor_point = create_gaussian(6, 100)
    #     gaussian_sets.extend(gaussian_set)
    # gaussian_sets = np.array(gaussian_sets)
    # df = pd.DataFrame(gaussian_sets)
    # results = [colors_string[dim] for dim in range(6) for _ in range(100)]

    # construct a distance matrix
    m = Matrix(df, heatmap_viz=False, feat_names=df.columns, DD=dist_mat)
    dist_mat = m.merged_matrix()

    # show kde plot of flattened distance matrix
    visualize_kde(dist_mat.flatten())

    # dimensionality reduction using distance matrix
    n_d_pts = len(df)

    # different dimensionality reduction techniques
    # dim_red_samples = dimred_glimmer(dist_mat)
    # dim_red_samples = dimred_tsne(dist_mat)
    # dim_red_samples = dimred_graph(dist_mat)
    dim_red_samples = dimred_mds(dist_mat)

    # calculate the reciprocal distance matrix if graph matrix
    reciprocal_dist_mat = np.reciprocal(dist_mat)
    np.fill_diagonal(reciprocal_dist_mat, 0)

    # create the distance matrix for the dimensionality reduced points
    flat_mat = scs.distance.pdist(dim_red_samples)
    sq_mat = scs.distance.squareform(flat_mat)

    # determine the total stress of the distance matrices
    total_stress(dist_mat, sq_mat, n_d_pts, n_d_pts)
    # determine the total knn stress of the distance matrices
    knn_els = list()
    for valid_tuple in valid_tuples:
        prot_knn = list(
            itertools.product([valid_tuple[0]], dict(valid_tuple[2]).keys()))
        unprot_knn = list(
            itertools.product([valid_tuple[0]], dict(valid_tuple[3]).keys()))
        knn_els += prot_knn + unprot_knn
    # calculate the total knn stress
    total_knn_stress(dist_mat, sq_mat, n_d_pts, n_d_pts, knn_els)

    n_feat = sq_mat.shape[0] - n_d_pts

    # show a heatmap of the distance matrix
    sns.heatmap(dist_mat)
    plt.show()
    # show a kde distribution plot of the distance matrix

    # construct the scatter plot from data points and feature points
    # data points + extra info
    data_pts = pd.DataFrame(dim_red_samples[:n_d_pts], columns=['x', 'y'])
    data_pts['type'] = 0
    data_pts = pd.concat([data_pts, og_df], axis=1)
    # feature points
    feat_pts = pd.DataFrame(dim_red_samples[n_d_pts:], columns=['x', 'y'])
    feat_pts['type'] = 2

    # create a basic scatter plot of data points
    scatter_data = dynamic.scatter_plot(data_pts,
                                        hover_data=list(og_df.columns),
                                        symbol=all_tuple_markers,
                                        symbol_map=symbol_map,
                                        color=np.array(class_colors),
                                        # color=np.array(results),
                                        size=[10] * n_d_pts, width=2,
                                        name='Tuples')
    # scatter plot of feature points
    scatter_feature = dynamic.scatter_plot(feat_pts, text=list(df.columns),
                                           size=[20] * n_feat,
                                           text_position='top center',
                                           color=['black'] * n_feat,
                                           # color=[colors_string[i] for i in range(6)],
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
    for i, segment_dict in enumerate(segment_dict_ls):
        # create the name of the segment
        segment_name = create_segment_name(segment_dict)

        # create the plotly segment
        kde_segment = dynamic.kde_segment(data_pts.copy(), segment_dict,
                                          colors_string[i], segment_name)

        # combine the segments
        init_segment = dynamic.combine_plots(init_segment, kde_segment)

    # combine the segments with the scatter plot
    scatter_final = dynamic.combine_plots(init_segment, scatter_final)

    # set the theme of the plot
    scatter_final = dynamic.set_theme(scatter_final, 'plotly_white')
    fig = scatter_final

    fig2 = fig.data[0]


    def update_point(trace, points, selector):
        c = list(fig2.marker.color)
        s = list(fig2.marker.size)
        for i in points.point_inds:
            c[i] = '#bae2be'
            s[i] = 20
            with fig2.batch_update():
                fig2.marker.color = c
                fig2.marker.size = s


    app = Dash(__name__)

    app.layout = html.Div(
        children=[
            dcc.Graph(
                id="graph",
                figure=fig,
                clear_on_unhover=True,
                style={'width': '100%', 'height': '100vh',
                       'display': 'inline-block'}),
            dcc.Store(id='graph_store_layout'),
            dcc.Store(id='graph_store_style')
        ]
    )

    click_shapes = list()


    @app.callback(
        Output('graph_store_style', 'data'),
        Input('graph', 'restyleData')
    )
    def update_restyle_store(restyle_data):
        return restyle_data


    @app.callback(
        Output('graph_store_layout', 'data'),
        Input('graph', 'relayoutData'),
    )
    def update_layout_store(relayout_data):
        return relayout_data


    @app.callback(
        Output("graph", "figure", allow_duplicate=True),
        Input("graph", "clickData"),
        prevent_initial_call=True
    )
    def update_click(clickData):
        global click_shapes

        if clickData is None:
            return no_update

        pt = clickData["points"][0]
        if 'hovertext' not in pt.keys():
            return no_update

        pt_coords = pt['x'], pt['y']
        pt_idx = data_pts[(data_pts['x'] == pt_coords[0]) & (
                data_pts['y'] == pt_coords[1])].index[0]

        # find all the neighbors of the point, both protected and unprotected
        prot_pts = list()
        unprot_pts = list()
        for valid_tuple in valid_tuples:
            if valid_tuple[0] == pt_idx:
                prot_pts = list(dict(valid_tuple[2]).keys())
                unprot_pts = list(dict(valid_tuple[3]).keys())

        # if the point isn't a sensitive tuple, return the figure
        if not len(prot_pts) and not len(unprot_pts):
            return no_update

        # clear all the line shapes
        fig['layout']['shapes'] = list()
        # find the coordinates of the protected points
        get_coords = lambda x: data_pts.iloc[x][['x', 'y']].values
        prot_coords = list(map(get_coords, prot_pts))
        unprot_coords = list(map(get_coords, unprot_pts))

        # create the protected lines
        temp_fig = go.Figure()
        for prot_coord in prot_coords:
            temp_fig.add_shape(type="line",
                               x0=pt_coords[0], y0=pt_coords[1],
                               x1=prot_coord[0], y1=prot_coord[1],
                               line=dict(width=1, color='red'))
        # create the unprotected lines
        for unprot_coord in unprot_coords:
            temp_fig.add_shape(type="line",
                               x0=pt_coords[0], y0=pt_coords[1],
                               x1=unprot_coord[0], y1=unprot_coord[1],
                               line=dict(width=1, color='green'))

        # add the lines to the figure
        fig['layout']['shapes'] = temp_fig['layout']['shapes']
        # add the lines to the figure
        click_shapes = temp_fig['layout']['shapes']

        return fig


    @app.callback(
        Output("graph", "figure"),
        Input("graph", "hoverData"),
        [State("graph_store_layout", "data"),
         State("graph_store_style", "data")],
    )
    def update_hover(hoverData, graph_store_layout, graph_store_style):
        global click_shapes

        # clear all the line shapes, apart from the ones created by clicking
        fig['layout']['shapes'] = click_shapes

        # keep the zoom level and everything when hovering using graph_store
        if graph_store_layout:
            if 'xaxis.autorange' in graph_store_layout or 'autosize' in graph_store_layout:
                fig['layout']['xaxis']['autorange'] = True
            else:
                fig['layout']['xaxis']['autorange'] = False
            if 'yaxis.autorange' in graph_store_layout or 'autosize' in graph_store_layout:
                fig['layout']['yaxis']['autorange'] = True
            else:
                fig['layout']['yaxis']['autorange'] = False

            for axis_name in ['axis', 'axis2']:
                if f'x{axis_name}.range[0]' in graph_store_layout:
                    print("yeet")
                    fig['layout'][f'x{axis_name}']['range'] = [
                        graph_store_layout[f'x{axis_name}.range[0]'],
                        graph_store_layout[f'x{axis_name}.range[1]']
                    ]
                if f'y{axis_name}.range[0]' in graph_store_layout:
                    fig['layout'][f'y{axis_name}']['range'] = [
                        graph_store_layout[f'y{axis_name}.range[0]'],
                        graph_store_layout[f'y{axis_name}.range[1]']
                    ]

        # keep hidden traces hidden and show traces that weren't hidden
        if graph_store_style:
            for idx, trace_idx in enumerate(graph_store_style[1]):
                fig.data[trace_idx].update(
                    visible=graph_store_style[0]['visible'][idx])

        if hoverData is None:
            return fig

        pt = hoverData["points"][0]
        if 'hovertext' not in pt.keys():
            return fig

        pt_coords = pt['x'], pt['y']
        try:
            pt_idx = data_pts[(data_pts['x'] == pt_coords[0]) & (
                data_pts['y'] == pt_coords[1])].index[0]
        except IndexError:
            print('no point index found')
            return fig

        # find all the neighbors of the point, both protected and unprotected
        prot_pts = list()
        unprot_pts = list()
        for valid_tuple in valid_tuples:
            if valid_tuple[0] == pt_idx:
                prot_pts = list(dict(valid_tuple[2]).keys())
                unprot_pts = list(dict(valid_tuple[3]).keys())

        # if the point isn't a sensitive tuple, return the figure
        if not len(prot_pts) and not len(unprot_pts):
            return fig

        # find the coordinates of the protected points
        get_coords = lambda x: data_pts.iloc[x][['x', 'y']].values
        prot_coords = list(map(get_coords, prot_pts))
        unprot_coords = list(map(get_coords, unprot_pts))

        # create the protected lines
        for prot_coord in prot_coords:
            fig.add_shape(type="line",
                          x0=pt_coords[0], y0=pt_coords[1],
                          x1=prot_coord[0], y1=prot_coord[1],
                          line=dict(width=1, color='rgb(256, 170, 170)'))
        # create the unprotected lines
        for unprot_coord in unprot_coords:
            fig.add_shape(type="line",
                          x0=pt_coords[0], y0=pt_coords[1],
                          x1=unprot_coord[0], y1=unprot_coord[1],
                          line=dict(width=1, color='rgb(170, 256, 170)'))

        return fig


    app.run_server(debug=True)
