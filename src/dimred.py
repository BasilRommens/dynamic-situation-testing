import itertools

import plotly.graph_objects as go
import seaborn as sns
import networkx as nx
import os
import scipy.spatial as scs
import numpy as np
import pandas as pd
from dash import Dash, html, dcc, Output, Input, no_update, State
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
import matplotlib.pyplot as plt
import time

import os

# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from umap import UMAP

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
    model = MDS(n_components=2, dissimilarity='precomputed', n_jobs=-1)

    arr = model.fit_transform(dist_mat)
    print(model.stress_)  # print the stress to show how stressed the layout is

    return arr


def dimred_pca(dist_mat, dims=8):
    model = PCA(n_components=dims)
    arr = model.fit_transform(dist_mat)
    return arr


def dimred_umap(dist_mat, dense=True):
    model = UMAP(n_components=2, n_neighbors=100, densmap=dense)
    arr = model.fit_transform(dist_mat)
    return arr


def dimred_tsne(dist_mat):
    model = TSNE(n_components=2, perplexity=20)
    arr = model.fit_transform(dist_mat)
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
    csv_fname = 'german_credit_data_processed.csv'
    protected_attributes = {"Sex": ["female"]}
    class_col = 'Class'
    ignore_cols = []
    # # adult dataset
    # path = 'data/'
    # json_fname = 'adult.json'
    # csv_fname = 'adult.csv'
    # protected_attributes = {"sex": ["Female"]}
    # class_col = 'class'
    # ignore_cols = ['native-country']

    # read the data from the csv and json file
    r = read_data(path + json_fname, path + csv_fname)
    # r.df = r.df.head(200)
    r.df = r.df.drop(columns=ignore_cols)
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
    df = pd.read_csv(path + csv_fname)
    # df = df.head(200)
    og_df = df.copy()
    sensitive_tuple_idxs = get_tuples_with_attr(df, protected_attributes)
    df = make_numeric(df, r.ordinal_attribute_values)
    class_col = df[class_col]
    class_colors = [colors_string[1] if c == 1.0 else colors_string[2] for c in
                    class_col]
    df = df.drop(columns=ignore_cols)

    # bounds for different columns
    segment_dict_ls = []

    # symbol map
    symbol_map = {'negative discrimination': 'line-ew',
                  'positive discrimination': 'cross-thin',
                  'neutral': 'circle',
                  'protected': 'circle'}

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

    # take the features neither in the ignore list, the class column, nor the
    # protected attributes
    feat_names = [col for col in df.columns if col not in ignore_cols and
                  col != list(decision_attribute.keys())[
                      0] and col not in protected_attributes.keys()]

    # construct a distance matrix
    m = Matrix(df, heatmap_viz=False, feat_names=feat_names, DD=dist_mat,
               attr_types=r.attribute_types, attr_corr=False)
    dist_mat = m.merged_matrix()

    # show kde plot using the flattened distance matrix
    visualize_kde(dist_mat.flatten())

    # dimensionality reduction using distance matrix
    n_d_pts = len(df)

    # different dimensionality reduction techniques
    # dim_red_samples = dimred_glimmer(dist_mat)
    # dim_red_samples = dimred_tsne(dist_mat)
    # dim_red_samples = dimred_umap(dist_mat, True)
    # dim_red_samples = dimred_graph(dist_mat)
    dim_red_samples = dimred_mds(dist_mat)

    # calculate the reciprocal distance matrix if graph matrix
    # reciprocal_dist_mat = np.reciprocal(dist_mat)
    # np.fill_diagonal(reciprocal_dist_mat, 0)

    # create the distance matrix for the dimensionality reduced points
    flat_mat = scs.distance.pdist(dim_red_samples)
    sq_mat = scs.distance.squareform(flat_mat)

    # determine the total stress of the distance matrices
    print("regular stress")
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
    print("knn stress")
    total_knn_stress(dist_mat, sq_mat, n_d_pts, n_d_pts, knn_els)

    # show a heatmap of the distance matrix
    sns.heatmap(dist_mat)
    plt.show()

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
    scatter_feature = dynamic.scatter_plot(feat_pts, text=feat_names,
                                           size=[20] * n_feat,
                                           text_position='top center',
                                           color=['black'] * n_feat,
                                           # color=[colors_string[i] for i in range(6)],
                                           name='Features')

    # scatter plot of sensitive tuples
    scatter_sensitive = dynamic.scatter_plot(
        data_pts.iloc[sensitive_tuple_idxs],
        symbol=['protected'] * len(sensitive_tuple_idxs),
        symbol_map={'protected': 'circle'},
        color=['grey'] * len(sensitive_tuple_idxs),
        size=[20] * len(sensitive_tuple_idxs),
        name='Protected Tuples')

    # combine the two scatter plots
    scatter_final = dynamic.combine_plots(scatter_data, scatter_feature)
    scatter_final = dynamic.combine_plots(scatter_sensitive, scatter_final)

    # create the segments using seaborn and translate to plotly
    init_segment = go.Figure()
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
    fig.show()
