import numpy as np

if __name__ == '__main__':
    pass


def stress(M1, M2):
    numerator = sum(np.power(M1[i, j] - M2[i, j], 2)
                    for j in range(M1.shape[1])
                    for i in range(M1.shape[0]))
    denominator = sum(np.power(M1[i, j], 2)
                      for j in range(M1.shape[1])
                      for i in range(M1.shape[0]))
    return np.sqrt(numerator / denominator)


def total_stress(og_mat, dimred_mat, width_DD_mat, height_DD_mat, beta_DD=1 / 7,
                 beta_VV=4 / 7, beta_DV=1 / 7, beta_VD=1 / 7):
    # determine the distances from the original distance matrix
    orig_DD = og_mat[:height_DD_mat, :width_DD_mat]
    orig_VD = og_mat[height_DD_mat:, :width_DD_mat]
    orig_DV = og_mat[:height_DD_mat, width_DD_mat:]
    orig_VV = og_mat[height_DD_mat:, width_DD_mat:]

    # determine the distances from the dimensionality reduced distance matrix
    DD = dimred_mat[:height_DD_mat, :width_DD_mat]
    VD = dimred_mat[height_DD_mat:, :width_DD_mat]
    DV = dimred_mat[:height_DD_mat, width_DD_mat:]
    VV = dimred_mat[height_DD_mat:, width_DD_mat:]

    # determine the stress for each submatrix and the total stress
    E_DD = stress(orig_DD, DD)
    E_VD = stress(orig_VD, VD)
    E_DV = stress(orig_DV, DV)
    E_VV = stress(orig_VV, VV)
    E_A = beta_DD * E_DD + beta_DV * E_DV + beta_VD * E_VD + beta_VV * E_VV

    # print all the stress values
    print(f'{E_DD=}')
    print(f'{E_VD=}')
    print(f'{E_DV=}')
    print(f'{E_VV=}')
    print(f'{E_A=}')

    return E_DD, E_VD, E_DV, E_VV, E_A


def total_knn_stress(og_mat, dimred_mat, width_DD_mat, height_DD_mat, knn_els,
                     beta_DD=1 / 7, beta_VV=4 / 7, beta_DV=1 / 7,
                     beta_VD=1 / 7):
    # filter out the elements in both the original matrix and the dimred matrix
    # that are not in the knn
    # filter matrix of DD matrix dimensions
    filter_mat = np.zeros((height_DD_mat, width_DD_mat))
    for row, col in knn_els:
        filter_mat[row, col] = 1

    # filter out the elements in both the original matrix and the dimred matrix
    og_mat[:height_DD_mat, :width_DD_mat] = og_mat[:height_DD_mat,
                                            :width_DD_mat] * filter_mat
    dimred_mat[:height_DD_mat, :width_DD_mat] = dimred_mat[:height_DD_mat,
                                                :width_DD_mat] * filter_mat

    return total_stress(og_mat, dimred_mat, width_DD_mat, height_DD_mat,
                        beta_DD, beta_VV, beta_DV, beta_VD)
