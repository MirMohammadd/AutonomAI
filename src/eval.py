from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics.cluster import adjusted_rand_score as ari
import numpy as np

def try_eval(y_tru, y_prd):
    dimension_1 = np.unique(y_tru).shape[0]
    dimension_2 = np.unique(y_prd).shape[0]

    mat = np.zeros(dimension_1,dimension_2)
    for yi, pi in zip(y_tru, y_prd):
        mat[pi,yi] += 1
    