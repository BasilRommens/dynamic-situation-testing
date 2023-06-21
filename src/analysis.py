import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
import scipy.spatial as scs
import seaborn as sns
import plotly.graph_objects as go

import cmap
from dataset import create_gaussian
from dimred import dimred_mds, dimred_umap, dimred_pca
from helper import create_segment_name
from knn import calc_dist_mat
from matrix import Matrix
from process import make_numeric
from stress import total_stress, total_knn_stress, get_knn_filter
from viz import dynamic


def analyze_usability_study():
    # read the usability study data
    df = pd.read_csv('data/usability_study.csv')

    # remove all the textual responses from the usability study
    df = df.drop(columns=['Timestamp', 'Give Me a Point - Discriminated',
                          'Give Me a Point - Maybe Discriminated',
                          'Give Me a Point - Not Discriminated',
                          'Is the Sample Discriminated?',
                          'What parts do you think were easy to use?',
                          'What parts do you think were difficult to use?',
                          'I would like to have more information on?',
                          'What functionality you felt was missing?',
                          ])

    # dict of question groups related to the usability study
    question_groups = {
        'SYSUSE': [
            'Overall, I am satisfied with how easy it is to use this system.',
            'It was simple to use this system.',
            'I was able to complete the tasks and scenarios quickly using this system.',
            'I felt comfortable using this system.',
            'It was easy to learn to use this system.',
            'I believe I could become productive quickly using this system.'
        ],
        'INFOQUAL': [
            'The information (such as online help, on-screen messages, and other documentation) provided with this system was clear.',
            'It was easy to find the information I needed.',
            'The information was effective in helping me complete the tasks and scenarios.',
            'The organization of information on the system screens was clear.'
        ],
        'INTERQUAL': [
            'The interface of this system was pleasant.',
            'I liked using the interface of this system.',
            'This system has all the functions and capabilities I expect it to have.',
            'Overall, I am satisfied with this system.'
        ]
    }

    # add the index as a column
    df = df.reset_index()

    # change the index column name to 'user'
    df = df.rename(columns={'index': 'user'})

    # melt the table such that we have a column for the question, a column for
    # the answer, and a column for the user
    df = df.melt(id_vars=['user'], var_name='question', value_name='answer')

    # add a column for the question group
    df['question_group'] = df['question'].apply(
        lambda x: [k for k, v in question_groups.items() if x in v][0])

    # copy the dataframe and set the question_group to all questions
    df_all = df.copy()
    df_all['question_group'] = 'ALL'

    # append the dataframe with all questions to the original dataframe
    df = pd.concat([df, df_all])

    # draw a boxplot per question group and add a boxplot for all the questions
    # together
    sns.set_theme(style="whitegrid")
    fig = sns.boxplot(x="question_group", y="answer", data=df,
                      palette=cmap.okabe_tl, showfliers=False)
    sns.swarmplot(x="question_group", y="answer", data=df,
                  color=cmap.okabe_tl[5], s=2)
    fig.set(xlabel='Question Group', ylabel='Answer Value (lower is better)')

    plt.savefig('out/usability_study.png', dpi=300, bbox_inches='tight')


def stress(df, attr_types, cols_to_use, DD, dim_red_method=dimred_mds,
           dimred_name='MDS', to_file='out/stress.csv', dataset_name='NA'):
    # visualize the iris dataset with the data context map and exclude the class
    # construct a distance matrix
    m = Matrix(df, heatmap_viz=False, feat_names=cols_to_use,
               attr_types=attr_types, DD=DD, attr_corr=True)
    full_dist_mat = m.merged_matrix()

    # dimensionality reduce with MDS
    dim_red_samples = dim_red_method(full_dist_mat)

    # create the distance matrix for the dimensionality reduced points
    flat_mat = scs.distance.pdist(dim_red_samples)
    sq_mat = scs.distance.squareform(flat_mat)

    # the number of data points in the dataset
    n_d_pts = len(df)

    # calculate the regular stress
    print("regular stress")
    E_DD, E_VD, E_DV, E_VV, E_A = total_stress(full_dist_mat, sq_mat, n_d_pts,
                                               n_d_pts)

    # convert the stress results to a pandas dataframe
    stress_results = pd.DataFrame(
        {'dataset': [dataset_name] * 5,
         'matrix': ['DD', 'VD', 'DV', 'VV', 'All'],
         'stress': [E_DD, E_VD, E_DV, E_VV, E_A],
         'dimred_technique': [dimred_name] * 5, 'stress_type': ['regular'] * 5})

    # determine the 10 nearest neighbors for all the elements
    knn_els = list()
    for row_idx, row in enumerate(DD):
        # get the indices of the top 10 lowest values
        indices = row.argsort()[:10]

        # combine the indices with the row index
        knn_els.extend([(row_idx, idx) for idx in indices])

    # calculate the KNN stress
    print("KNN stress")
    E_DD, E_VD, E_DV, E_VV, E_A = total_knn_stress(full_dist_mat, sq_mat,
                                                   n_d_pts, n_d_pts, knn_els)

    # convert the stress results to a pandas dataframe
    stress_results_knn = pd.DataFrame(
        {'dataset': [dataset_name] * 5,
         'matrix': ['DD', 'VD', 'DV', 'VV', 'All'],
         'stress': [E_DD, E_VD, E_DV, E_VV, E_A],
         'dimred_technique': [dimred_name] * 5, 'stress_type': ['knn'] * 5})

    # combine the regular and knn stress results
    all_stress_results_df = pd.concat([stress_results, stress_results_knn])

    # write the stress results to a file
    all_stress_results_df.to_csv(to_file, index=False, mode='a')


def stress_auto(dim_red_method=dimred_mds, dimred_name='MDS'):
    # cols to include in distance measure
    cols_to_use = ['mpg', 'cylinders', 'horsepower', 'weight',
                   'acceleration', 'year', 'origin']

    # set the attribute types
    attr_types = {col: 'interval' if col != 'origin' else 'nominal' for col in
                  cols_to_use}

    # load the iris dataset
    df = pd.read_csv('data/auto-mpg.csv', na_values='?')

    # only keep the columns we want to use
    df = df[cols_to_use]

    # calculate the dd matrix using the situation testing distances
    DD = calc_dist_mat(df, {}, attr_types, {}, {})

    # calculate the stress
    stress(df, attr_types, cols_to_use, DD, dim_red_method, dimred_name,
           dataset_name='auto MPG')


def stress_gaussian():
    # cols to include in distance measure
    cols_to_use = [0, 1, 2, 3, 4, 5]

    # set the attribute types
    attr_types = {col: 'interval' for col in cols_to_use}

    gaussian_sets = list()
    np.random.seed(42)
    for i in range(6):
        gaussian_set, anchor_point = create_gaussian(6, 100)
        gaussian_sets.extend(gaussian_set)
    gaussian_sets = np.array(gaussian_sets)
    df = pd.DataFrame(gaussian_sets)

    # calculate the dd matrix using the situation testing distances
    DD = calc_dist_mat(df, {}, attr_types, {}, {})

    # calculate the stress
    stress(df, attr_types, cols_to_use, DD, dimred_mds, 'MDS',
           'out/stress_gaussian.csv', dataset_name='Gaussian')


def stress_german_credit(dim_red_method=dimred_mds, dimred_name='MDS'):
    # set the attribute types
    attr_types = {
        "Age": "interval",
        "Sex": "nominal",
        "Job": "ordinal",
        "Housing": "nominal",
        "Saving accounts": "ordinal",
        "Checking account": "ordinal",
        "Credit amount": "interval",
        "Duration": "interval",
        "Purpose": "nominal",
    }

    # the ranked attribute types
    ranked_attr_values = {
        "Job": {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3
        },
        "Saving accounts": {
            "little": 0,
            "moderate": 1,
            "quite rich": 2,
            "rich": 3
        },
        "Checking account": {
            "little": 0,
            "moderate": 1,
            "quite rich": 2,
            "rich": 3
        }
    }

    # read the german credit dataset
    df = pd.read_csv('data/german_credit_data_processed.csv')
    df = df.drop(['Class'], axis=1)

    # cols to include in distance measure
    cols_to_use = df.columns

    # calculate the dd matrix using the situation testing distances
    DD = calc_dist_mat(df, ranked_attr_values, attr_types, {}, {})

    # make the values numeric in the dataframe
    df = make_numeric(df, ranked_attr_values)

    # calculate the stress
    stress(df, attr_types, cols_to_use, DD, dim_red_method, dimred_name,
           dataset_name='german credit')


def stress_adult(dim_red_method=dimred_mds, dimred_name='MDS'):
    # set the attribute types
    attr_types = {
        "age": "interval",
        "workclass": "nominal",
        "fnlwgt": "interval",
        "education": "ordinal",
        "education-num": "ordinal",
        "marital-status": "nominal",
        "occupation": "nominal",
        "relationship": "nominal",
        "race": "nominal",
        "sex": "nominal",
        "capital-gain": "interval",
        "capital-loss": "interval",
        "hours-per-week": "interval",
        "native-country": "nominal",
        "class": "nominal"
    }

    # the ranked attribute types
    ranked_attr_values = {
        "education": {
            "Preschool": 0,
            "1st-4th": 1,
            "5th-6th": 2,
            "7th-8th": 3,
            "9th": 4,
            "10th": 5,
            "11th": 6,
            "12th": 7,
            "HS-grad": 8,
            "Some-college": 9,
            "Assoc-voc": 10,
            "Assoc-acdm": 11,
            "Bachelors": 12,
            "Masters": 13,
            "Prof-school": 14,
            "Doctorate": 15
        },
        "education-num": {
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "10": 10,
            "11": 11,
            "12": 12,
            "13": 13,
            "14": 14,
            "15": 15,
            "16": 16
        }
    }

    # read the german credit dataset
    df = pd.read_csv('data/adult.csv')
    df = df.drop(['class'], axis=1)

    # limit to 5.000 first entries
    df = df.head(5_000)

    # cols to include in distance measure
    cols_to_use = df.columns

    # calculate the dd matrix using the situation testing distances
    DD = calc_dist_mat(df, ranked_attr_values, attr_types, {}, {})

    # make the values numeric in the dataframe
    df = make_numeric(df, ranked_attr_values)

    # calculate the stress
    stress(df, attr_types, cols_to_use, DD, dim_red_method, dimred_name,
           dataset_name='adult')


def stress_COMPAS(dim_red_method=dimred_mds, dimred_name='MDS'):
    # set the attribute types
    attr_types = {
        "Person_ID": "nominal",
        "AssessmentID": "nominal",
        "CaseID": "nominal",
        "Agency_Text": "nominal",
        "FirstName": "nominal",
        "MiddleName": "nominal",
        "Sex_Code_Text": "nominal",
        "Ethnic_Code_Text": "nominal",
        "DateOfBirth": "interval",
        "ScaleSet_ID": "nominal",
        "ScaleSet": "nominal",
        "AssessmentReason": "nominal",
        "Language": "nominal",
        "LegalStatus": "nominal",
        "CustodyStatus": "nominal",
        "MaritalStatus": "nominal",
        "Screening_Date": "interval",
        "RecSupervisionLevel": "ordinal",
        "RecSupervisionLevelText": "ordinal",
        "Scale_ID": "nominal",
        "DisplayText": "nominal",
        "RawScore": "interval",
        "DecileScore": "ordinal",
        "ScoreText": "ordinal",
        "AssessmentType": "nominal",
        "IsCompleted": "nominal",
        "IsDeleted": "nominal"
    }

    # the ranked attribute types
    ranked_attr_values = {
        "RecSupervisionLevel": {
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4
        },
        "RecSupervisionLevelText": {
            "Low": 1,
            "Medium": 2,
            "Medium with Override Consideration": 3,
            "High": 4
        },
        "DecileScore": {
            "-1": -1,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "10": 10
        },
        "ScoreText": {
            "Low": 1,
            "Medium": 2,
            "High": 3
        }
    }

    # read the german credit dataset
    df = pd.read_csv('data/compas-scores-raw-processed.csv',
                     na_values=['N/A', 'NULL'])

    # columns to drop
    cols_to_drop = ["IsCompleted", "IsDeleted", "Person_ID", "AssessmentID",
                    "Case_ID", "FirstName", "MiddleName", "RawScore",
                    "DecileScore"]

    # drop the columns
    df = df.drop(cols_to_drop, axis=1)

    # limit to 5.000 first entries
    df = df.head(5_000)

    # cols to include in distance measure
    cols_to_use = df.columns

    # calculate the dd matrix using the situation testing distances
    DD = calc_dist_mat(df, ranked_attr_values, attr_types, {}, {})

    # make the values numeric in the dataframe
    df = make_numeric(df, ranked_attr_values)

    # calculate the stress
    stress(df, attr_types, cols_to_use, DD, dim_red_method, dimred_name,
           dataset_name='COMPAS')


def plot_gaussian():
    # cols to include in distance measure
    cols_to_use = [0, 1, 2, 3, 4, 5]

    # set the attribute types
    attr_types = {col: 'interval' for col in cols_to_use}

    # gaussian data set
    gaussian_sets = list()
    np.random.seed(42)
    for i in range(6):
        gaussian_set, anchor_point = create_gaussian(6, 100)
        gaussian_sets.extend(gaussian_set)
    gaussian_sets = np.array(gaussian_sets)
    all_pts = pd.DataFrame(gaussian_sets)
    results = [cmap.okabe_tl[dim] for dim in range(6) for _ in range(100)]

    # visualize the gaussians with the data context map and exclude the class
    # construct a distance matrix
    m = Matrix(all_pts, heatmap_viz=False, feat_names=cols_to_use,
               attr_types=attr_types, attr_corr=False)
    dist_mat = m.merged_matrix(VV_weigth=1.0, DD_weight=1.0, VD_weight=1.0,
                               DV_weight=1.0)

    # dimensionality reduction using distance matrix
    n_d_pts = len(all_pts)
    n_feat = len(cols_to_use)

    # dimensionality reduce with MDS
    dim_red_samples = dimred_mds(dist_mat)

    # construct the scatter plot from data points and feature points
    # data points + extra info
    data_pts = pd.DataFrame(dim_red_samples[:n_d_pts], columns=['x', 'y'])
    data_pts['type'] = 0
    data_pts = pd.concat([data_pts], axis=1)

    # feature points
    feat_pts = pd.DataFrame(dim_red_samples[n_d_pts:], columns=['x', 'y'])
    feat_pts['type'] = 2

    # create a basic scatter plot of data points
    scatter_data = dynamic.scatter_plot(data_pts,
                                        color=np.array(results),
                                        size=[10] * n_d_pts, width=2,
                                        name='Tuples')
    # scatter plot of feature points
    scatter_feature = dynamic.scatter_plot(feat_pts,
                                           text=[i for i in range(1, 7)],
                                           size=[20] * 6,
                                           text_position='top center',
                                           color=cmap.okabe_tl[:6],
                                           name='Features')

    # combine the two scatter plots
    fig = dynamic.combine_plots(scatter_data, scatter_feature)

    # set the theme of the plot to plotly_white
    fig.update_layout(template='plotly_white')

    # remove the axis ticks
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    # remove the legend
    fig.update_layout(showlegend=False)

    # set the margins lower
    fig.update_layout(margin=dict(l=10, r=10, b=10, t=10))

    # save the figure as a png
    fig.write_image('out/gaussian.png')


def plot_auto():
    # cols to include in distance measure
    cols_to_use = ['mpg', 'cylinders', 'horsepower', 'weight',
                   'acceleration', 'year', 'origin']

    # set the attribute types
    attr_types = {col: 'interval' if col != 'origin' else 'nominal' for col in
                  cols_to_use}

    # auto mpg data set
    df = pd.read_csv('data/auto-mpg.csv', na_values='?')

    # keep only the columns to use
    df = df[cols_to_use]

    # calculate the dd matrix using the situation testing distances
    DD = calc_dist_mat(df, {}, attr_types, {}, {})

    # visualize the gaussians with the data context map and exclude the class
    # construct a distance matrix
    m = Matrix(df, heatmap_viz=False, feat_names=cols_to_use,
               attr_types=attr_types, DD=DD, attr_corr=True)
    dist_mat = m.merged_matrix()

    # dimensionality reduction using distance matrix
    n_d_pts = len(DD)
    n_feat = len(cols_to_use)

    # dimensionality reduce with MDS
    dim_red_samples = dimred_mds(dist_mat)

    # construct the scatter plot from data points and feature points
    # data points + extra info
    data_pts = pd.DataFrame(dim_red_samples[:n_d_pts], columns=['x', 'y'])
    data_pts['type'] = 0
    data_pts = pd.concat([data_pts, df], axis=1)

    # feature points
    feat_pts = pd.DataFrame(dim_red_samples[n_d_pts:], columns=['x', 'y'])
    feat_pts['type'] = 2

    # set colors per origin
    colors = [cmap.okabe_tl[int(origin) - 1]
              for origin in df['origin']]

    # create a basic scatter plot of data points
    scatter_data = dynamic.scatter_plot(data_pts,
                                        color=colors,
                                        size=[10] * n_d_pts, width=2,
                                        name='Tuples')
    # scatter plot of feature points
    scatter_feature = dynamic.scatter_plot(feat_pts,
                                           text=cols_to_use,
                                           size=[20] * n_feat,
                                           text_position='top center',
                                           color=['black'] * n_feat,
                                           name='Features')

    # combine the two scatter plots
    fig = dynamic.combine_plots(scatter_data, scatter_feature)

    # determine the contours
    segment_dict_ls = [{'weight': [1500, 2000]}, {'acceleration': [20, 25]},
                       {'horsepower': [80, 110]}]

    # create the segments using seaborn and translate to plotly
    init_segment = go.Figure()
    for i, segment_dict in enumerate(segment_dict_ls):
        # create the name of the segment
        segment_name = create_segment_name(segment_dict)

        # create the plotly segment
        kde_segment = dynamic.kde_segment(data_pts.copy(), segment_dict,
                                          cmap.okabe_tl[i], segment_name)

        # combine the segments
        init_segment = dynamic.combine_plots(init_segment, kde_segment)

    # add the kde contours to the scatter plots
    fig = dynamic.combine_plots(init_segment, fig)

    # set the theme of the plot to plotly_white
    fig.update_layout(template='plotly_white')

    # remove the axis ticks
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    # remove the legend
    fig.update_layout(showlegend=False)

    # set the margins lower
    fig.update_layout(margin=dict(l=10, r=10, b=10, t=10))

    # save the figure as a png
    fig.write_image('out/auto.png')


def plot_stress(stress_dataset='data/tidy-stress.csv', dimred_technique='MDS',
                knn_ofile='out/knn-stress.png',
                regular_ofile='out/regular-stress.png'):
    # read the stress dataset
    df = pd.read_csv(stress_dataset)

    # remove the gaussian cloud results
    df = df[df['dataset'] != 'gaussian']

    # only take the results with specified dimred technique
    df = df[df['dimred_technique'] == dimred_technique]

    # get the regular stress df
    df_regular = df[df['stress_type'] == 'regular']

    # create a grouped bar chart of the matrix
    fig = px.bar(df_regular, x='matrix', y='stress', color='dataset',
                 color_discrete_sequence=cmap.okabe_tl, barmode='group')

    # set the theme of the plot to plotly_white
    fig.update_layout(template='plotly_white')

    # move the legend to the top middle
    fig.update_layout(legend=dict(orientation='h', yanchor='top',
                                  y=1.1, xanchor='center', x=0.5))

    # make the legend title bold
    fig.update_layout(legend_title_text='<b>Dataset:</b>')

    # save the figure as png
    fig.write_image(regular_ofile)

    # get the knn stress df
    df_knn = df[df['stress_type'] == 'knn']

    # remove the auto MPG dataset from the KNN results
    df_knn = df_knn[df_knn['dataset'] != 'auto MPG']

    # create a grouped bar chart of the matrix
    fig = px.bar(df_knn, x='matrix', y='stress', color='dataset',
                 color_discrete_sequence=cmap.okabe_tl[1:], barmode='group')

    # set the theme of the plot to plotly_white
    fig.update_layout(template='plotly_white')

    # move the legend to the top middle
    fig.update_layout(legend=dict(orientation='h', yanchor='top',
                                  y=1.1, xanchor='center', x=0.5))

    # make the legend title bold
    fig.update_layout(legend_title_text='<b>Dataset:</b>')

    # save the figure as png
    fig.write_image(knn_ofile)


def plot_presentation(stress_dataset='data/stress.csv'):
    regular_ofile = 'out/regular-presentation.png'
    knn_ofile = 'out/knn-presentation.png'

    # read the stress dataset
    df = pd.read_csv(stress_dataset)

    # remove the gaussian cloud results
    df = df[df['dataset'] != 'gaussian']

    # only take the results with specified dimred technique
    df = df[df['dimred_technique'].isin(['MDS', 'PCA'])]

    # get only all stress results
    df = df[df['matrix'] == 'All']

    # get the regular stress df
    df_regular = df[df['stress_type'] == 'regular']

    # create a grouped bar chart of the matrix
    fig = px.bar(df_regular, x='dimred_technique', y='stress', color='dataset',
                 color_discrete_sequence=cmap.okabe_tl, barmode='group',
                 labels={'stress_type': 'Stress Type', 'stress': 'Stress'})

    # set the theme of the plot to plotly_white
    fig.update_layout(template='plotly_white')

    # move the legend to the top middle
    fig.update_layout(legend=dict(orientation='h', yanchor='top',
                                  y=1.1, xanchor='center', x=0.5))

    # make the legend title bold
    fig.update_layout(legend_title_text='<b>Dataset:</b>')

    # save the figure as png
    fig.write_image(regular_ofile)

    # get only the MDS knn results
    df_knn = df[df['dimred_technique'] == 'MDS']

    # remove the auto MPG dataset from the KNN results
    df_knn = df_knn[df_knn['dataset'] != 'auto MPG']

    # create a grouped bar chart of the matrix
    fig = px.bar(df_knn, x='stress_type', y='stress', color='dataset',
                 color_discrete_sequence=cmap.okabe_tl[1:], barmode='group',
                 labels={'stress_type': 'Stress Type', 'stress': 'Stress'})

    # set the theme of the plot to plotly_white
    fig.update_layout(template='plotly_white')

    # move the legend to the top middle
    fig.update_layout(legend=dict(orientation='h', yanchor='top',
                                  y=1.1, xanchor='center', x=0.5))

    # make the legend title bold
    fig.update_layout(legend_title_text='<b>Dataset:</b>')

    # save the figure as png
    fig.write_image(knn_ofile)


if __name__ == '__main__':
    # analyze_usability_study()
    # plot_auto()
    # dimred_func_ls = [dimred_mds, dimred_umap, dimred_pca]
    # dimred_name_ls = ['MDS', 'UMAP', 'PCA']
    # for dimred_func, dimred_name in zip(dimred_func_ls, dimred_name_ls):
    #     stress_auto(dimred_func, dimred_name)
    #     stress_german_credit(dimred_func, dimred_name)
    #     stress_adult(dimred_func, dimred_name)
    #     stress_COMPAS(dimred_func, dimred_name)
    # plot_stress('data/stress.csv', 'MDS', knn_ofile='out/mds-stress-knn.png',
    #             regular_ofile='out/mds-stress-regular.png')
    # plot_stress('data/stress.csv', 'UMAP', knn_ofile='out/umap-stress-knn.png',
    #             regular_ofile='out/umap-stress-regular.png')
    # plot_stress('data/stress.csv', 'PCA', knn_ofile='out/pca-stress-knn.png',
    #             regular_ofile='out/pca-stress-regular.png')
    plot_presentation()
