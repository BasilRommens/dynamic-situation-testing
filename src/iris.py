import pandas as pd
import plotly.express as px
import src.cmap as cmap
from dimred import dimred_mds, dimred_umap
from matrix import Matrix
from viz import dynamic


def plot_splom():
    # set plotly theme to white
    px.defaults.template = 'plotly_white'

    # cols to visualize
    cols_to_viz = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    # load the iris dataset
    iris_df = pd.read_csv('data/iris.csv')

    # get the unique species
    species = iris_df.species.unique()

    # create the color map for the species
    species_cmap = {specie: color for specie, color in
                    zip(species, cmap.okabe_tl)}

    # add additional column which changes the species name to a color
    iris_df['color'] = iris_df['species'].map(species_cmap)

    # give all the species a species id
    species_imap = {specie: i for i, specie in enumerate(species)}

    # add additional column which changes the species name to a species id
    iris_df['species_id'] = iris_df['species'].map(species_imap)

    # visualize the iris dataset with the scatterplot matrix from plotly
    fig = px.scatter_matrix(iris_df, dimensions=cols_to_viz, color='species',
                            symbol='species', color_discrete_map=species_cmap,
                            labels={col: col.replace('_', ' ') for col in
                                    cols_to_viz})
    fig.update_traces(diagonal_visible=False)  # remove the diagonal plots

    # save the figure as a png
    fig.write_image('out/iris_scatterplot_matrix.png')


def plot_parallel_coordinates():
    # set plotly theme to white
    px.defaults.template = 'plotly_white'

    # cols to visualize
    cols_to_viz = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    # load the iris dataset
    iris_df = pd.read_csv('data/iris.csv')

    # get the unique species
    species = iris_df.species.unique()

    # create the color map for the species
    species_cmap = {specie: color for specie, color in
                    zip(species, cmap.okabe_tl)}

    # add additional column which changes the species name to a color
    iris_df['color'] = iris_df['species'].map(species_cmap)

    # give all the species a species id
    species_imap = {specie: i for i, specie in enumerate(species)}

    # add additional column which changes the species name to a species id
    iris_df['species_id'] = iris_df['species'].map(species_imap)

    # visualize the iris dataset with the parallel coordinates from plotly
    color_continuous_scale = cmap.okabe_tl[:len(species)]
    fig = px.parallel_coordinates(iris_df, dimensions=cols_to_viz,
                                  color='species_id',
                                  color_continuous_scale=color_continuous_scale,
                                  labels={col: col.replace('_', ' ') for col in
                                          cols_to_viz})

    # make the figure wider
    fig.update_layout(width=1200)

    # save the figure as a png
    fig.write_image('out/iris_parallel_coordinates.png')


def plot_data_context_map():
    # set plotly theme to white
    px.defaults.template = 'plotly_white'

    # cols to visualize
    cols_to_viz = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    # load the iris dataset
    iris_df = pd.read_csv('data/iris.csv')

    # get the unique species
    species = iris_df.species.unique()

    # create the color map for the species
    species_cmap = {specie: color for specie, color in
                    zip(species, cmap.okabe_tl)}

    # add additional column which changes the species name to a color
    iris_df['color'] = iris_df['species'].map(species_cmap)

    # visualize the iris dataset with the data context map and exclude the class
    # construct a distance matrix
    m = Matrix(iris_df, heatmap_viz=False, feat_names=cols_to_viz,
               attr_types={col_to_viz: 'interval' for col_to_viz in
                           cols_to_viz},
               attr_corr=True)
    dist_mat = m.merged_matrix()

    # dimensionality reduction using distance matrix
    n_d_pts = len(iris_df)
    n_feat = len(cols_to_viz)

    # dimensionality reduce with MDS
    dim_red_samples = dimred_mds(dist_mat)

    # construct the scatter plot from data points and feature points
    # data points + extra info
    data_pts = pd.DataFrame(dim_red_samples[:n_d_pts], columns=['x', 'y'])
    data_pts['type'] = 0
    data_pts = pd.concat([data_pts, iris_df], axis=1)

    # feature points
    feat_pts = pd.DataFrame(dim_red_samples[n_d_pts:], columns=['x', 'y'])
    feat_pts['type'] = 2

    # split the data points by species
    data_pts_split = [data_pts[data_pts['species'] == specie] for specie in
                      species]

    # a symbol map for the species
    species_sym_map = {'Iris-setosa': 'circle', 'Iris-versicolor': 'diamond',
                       'Iris-virginica': 'square'}

    # create a basic scatter plot of data points
    scatter_data_ls = list()
    for data_pts_specie in data_pts_split:
        scatter_data = dynamic.scatter_plot(data_pts_specie,
                                            color=data_pts_specie['color'],
                                            symbol=data_pts_specie['species'],
                                            symbol_map=species_sym_map,
                                            size=[10] * n_d_pts, width=2,
                                            name=
                                            data_pts_specie['species'].unique()[
                                                0])
        scatter_data_ls.append(scatter_data)

    scatter_data = scatter_data_ls[0]
    for scatter_data_new in scatter_data_ls[1:]:
        scatter_data = dynamic.combine_plots(scatter_data, scatter_data_new)

    # scatter plot of feature points
    scatter_feature = dynamic.scatter_plot(feat_pts, text=cols_to_viz,
                                           size=[20] * n_feat,
                                           text_position='top center',
                                           color=['black'] * n_feat,
                                           name='Features')

    # combine the feature and data points
    fig = dynamic.combine_plots(scatter_data, scatter_feature)

    # set the theme of the plot to plotly_white
    fig.update_layout(template='plotly_white')

    # remove the axis ticks
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    # move the legend to the top middle
    fig.update_layout(legend=dict(orientation='h', yanchor='top',
                                  y=1.1, xanchor='center', x=0.5))
    # set the margins lower
    fig.update_layout(margin=dict(l=10, r=10, b=10, t=10))

    # save the figure as a png
    fig.write_image('out/iris_data_context_map.png')


if __name__ == '__main__':
    plot_splom()
    plot_parallel_coordinates()
    plot_data_context_map()
