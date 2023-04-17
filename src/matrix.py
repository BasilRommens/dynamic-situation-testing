import pandas as pd
import numpy as np
import scipy.spatial as scs

from process import make_numeric
import seaborn as sns
import matplotlib.pyplot as plt
import time

eps = 1e-3


class Matrix:
    def __init__(self, elements, heatmap_viz=False, DD=None, VV=None, VD=None,
                 DV=None, feat_names=[]):
        self.elements = elements
        self.heatmap_viz = heatmap_viz
        self.DD = DD
        self.VV = VV
        self.VD = VD
        self.DV = DV
        self.feat_names = feat_names
        self.matrix = self._construct_matrix(DD, VV, VD, DV)

    def _construct_matrix(self, DD, VV, VD, DV) -> np.ndarray:
        # First we will construct parts of the matrix and then merge them
        DD_mat = self._construct_DD_mat() if DD is None else DD
        VV_mat = self._construct_VV_mat() if VV is None else VV
        VD_mat = self._construct_VD_mat() if VD is None else VD
        DV_mat = self._construct_DV_mat() if DV is None else DV

        # check similarity of VD and DV matrix
        if not np.array_equal(VD_mat, DV_mat.T):
            # determine the max norm matrix
            max_norm = np.argmax(
                [np.linalg.norm(VD_mat), np.linalg.norm(DV_mat)])
            if max_norm == 0:
                DV_mat = VD_mat.T
            else:
                VD_mat = DV_mat.T

        # merge the matrices in a single matrix
        return self._matrix_fusion(DD_mat, VV_mat, DV_mat, VD_mat)

    def _construct_DD_mat(self) -> np.ndarray:
        """
        calculates the distance between each pair of tuples in the dataset
        :return: data to data point matrix
        """
        # get all the elements in a list
        elements_ls = self.elements.values
        # scale all the elements to be within the range [0, 1]
        # elements_ls = (elements_ls - elements_ls.min()) / (
        #         elements_ls.max() - elements_ls.min())

        # normalize the elements list
        elements_ls = elements_ls / np.array(
            [np.linalg.norm(elements_ls, axis=1)]).T

        # use pdist to calculate the distances
        distances = scs.distance.pdist(elements_ls, 'euclidean')

        # convert the distances to a square matrix
        distances = scs.distance.squareform(distances)
        return distances

    def _construct_VV_mat(self) -> np.ndarray:
        """
        calculates the distance between each feature value vector
        :return: feature to feature matrix
        """
        # create the feature vectors in list form
        feature_ls = [self.elements[col].values
                      for col in self.elements.columns]

        # calculate the distances between the V vectors
        distances = scs.distance.pdist(feature_ls, 'correlation')

        # convert the distances to a square matrix
        distances = scs.distance.squareform(distances)

        return distances

    def _construct_VD_mat(self) -> np.ndarray:
        """
        suppose we have a unit vector Vi for feature i with length equal to the
        number of features and where the feature i is a 1. The data points Di
        stay the same. We will use these two vectors to calculate the distances.
        :return: feature to data point matrix
        """
        # get all the instances in a vector list
        instances_ls = -self.elements.values
        # calculate the highest and lowest feature value for each feature
        max_instances = np.max(instances_ls, axis=0)
        min_instances = np.min(instances_ls, axis=0)
        interval_size_instances = max_instances - min_instances
        instances_ls -= min_instances
        instances_ls /= interval_size_instances
        return instances_ls.T

        # legacy code
        # create the unit vectors of the feature vectors
        feature_ls = np.array([[1 if i == col_idx else 0
                                for i in range(len(self.elements.columns))]
                               for col_idx in
                               range(len(self.elements.columns))])

        # get all the instances in a vector list
        instances_ls = -self.elements.values
        # calculate the highest and lowest feature value for each feature
        max_instances = np.max(instances_ls, axis=0)
        min_instances = np.min(instances_ls, axis=0)
        interval_size_instances = max_instances - min_instances
        instances_ls -= min_instances
        instances_ls /= interval_size_instances

        # calculate the distances between the feature vectors and the instances
        one_value = lambda u, v: v[np.where(u == 1)]
        distances = scs.distance.cdist(feature_ls, instances_ls, one_value)

        return distances

    def _construct_DV_mat(self) -> np.ndarray:
        """
        suppose we have a unit vector Di for data point i with length equal to
        the number of data points and where the data point i is a 1. The feature
        vectors Vi contain the values of all the datapoints for feature i. We
        will use these two vectors to calculate the distances.
        :return: data point to feature matrix
        """
        # get all the instances in a vector list
        instances_ls = -self.elements.values
        # calculate the highest and lowest feature value for each feature
        max_instances = np.max(instances_ls, axis=0)
        min_instances = np.min(instances_ls, axis=0)
        interval_size_instances = max_instances - min_instances
        instances_ls -= min_instances
        instances_ls /= interval_size_instances

        return instances_ls

        # legacy code
        # create the unit vectors for the instances
        instances_ls = np.array([[1 if i == row_idx else 0
                                  for i in range(len(self.elements.values))]
                                 for row_idx in
                                 range(len(self.elements.values))])

        # create the feature vectors in list form
        feature_ls = -np.array([self.elements[col].tolist()
                                for col in self.elements.columns])

        # create a min max interval for each feature to scale the features
        # to be inside the [0,1] interval
        min_features = np.min(feature_ls, axis=1)
        max_features = np.max(feature_ls, axis=1)
        interval_size_features = max_features - min_features
        feature_ls -= np.array([min_features]).T
        feature_ls /= np.array([interval_size_features]).T

        # calculate the distances between the data point vectors and features
        one_value = lambda u, v: v[np.where(u == 1)]
        distances = scs.distance.cdist(instances_ls, feature_ls, one_value)

        return distances

    def merged_matrix(self, DD_weight=1.0, VV_weigth=1.0, DV_weight=1.0,
                      VD_weight=1.0):
        weighted_DD = self.DD * DD_weight
        weighted_VV = self.VV * VV_weigth
        weighted_DV = self.DV * DV_weight
        weighted_VD = self.VD * VD_weight

        # merge the matrices in a single matrix
        upper_mat = np.concatenate((weighted_DD, weighted_DV), axis=1)
        lower_mat = np.concatenate((weighted_VD, weighted_VV), axis=1)

        full_mat = np.concatenate((upper_mat, lower_mat), axis=0)
        if self.heatmap_viz:
            sns.heatmap(full_mat).set(title='Full Matrix')
            plt.show()

        # add a small epsilon (assumption: no negative distance values)
        full_mat += eps
        return full_mat

    def _matrix_fusion(self, DD, VV, DV, VD) -> np.ndarray:
        """
        fuses the matrices together to create the final matrix
        :param DD: data point to data point distance matrix
        :param VV: feature vector to feature vector distance matrix
        :param DV: data point to feature unit vector distance matrix
        :param VD: feature value vector to data point unit vector distance
        matrix
        :return: the fused matrix
        """
        if self.heatmap_viz:
            sns.heatmap(DD).set(title='DD pre-norm')
            plt.show()
            sns.heatmap(DV).set(title='DV pre-norm',
                                xticklabels=self.feat_names)
            plt.show()
            sns.heatmap(VD).set(title='VD pre-norm',
                                yticklabels=self.feat_names)
            plt.show()
            sns.heatmap(VV).set(title='VV pre-norm',
                                xticklabels=self.feat_names,
                                yticklabels=self.feat_names)
            plt.show()

        # find the weights with which we scale all the matrices
        # find the mean of all the matrices
        DD_mean = np.average(DD)
        VV_mean = np.average(VV)
        DV_mean = np.average(DV)
        VD_mean = np.average(VD)

        # find the max mean
        max_mean = max(DD_mean, VV_mean, DV_mean, VD_mean)

        # determine the weights
        DD_weight = max_mean / DD_mean
        VV_weight = max_mean / VV_mean
        DV_weight = max_mean / DV_mean
        VD_weight = max_mean / VD_mean

        # scale the matrices and store
        self.DD = DD * DD_weight
        self.VV = VV * VV_weight
        self.DV = DV * DV_weight
        self.VD = VD * VD_weight

        if self.heatmap_viz:
            sns.heatmap(self.DD).set(title='DD post-norm')
            plt.show()
            sns.heatmap(self.DV).set(title='DV post-norm',
                                     xticklabels=self.feat_names)
            plt.show()
            sns.heatmap(self.VD).set(title='VD post-norm',
                                     yticklabels=self.feat_names)
            plt.show()
            sns.heatmap(self.VV).set(title='VV post-norm',
                                     xticklabels=self.feat_names,
                                     yticklabels=self.feat_names)
            plt.show()

        # merge the matrices in a single matrix
        upper_mat = np.concatenate((self.DD, self.DV), axis=1)
        lower_mat = np.concatenate((self.VD, self.VV), axis=1)

        full_mat = np.concatenate((upper_mat, lower_mat), axis=0)
        if self.heatmap_viz:
            sns.heatmap(full_mat).set(title='Full Matrix')
            plt.show()

        # add a small epsilon (assumption: no negative distance values)
        full_mat += eps

        return full_mat


if __name__ == '__main__':
    df = pd.read_csv('data/german_credit_data.csv')
    df = make_numeric(df)
    m = Matrix(df)

    print(len(df))
    print(m.matrix.shape)
