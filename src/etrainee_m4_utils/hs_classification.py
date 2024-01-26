#
#------------------------------------------------------------------------------
# Copyright (c) 2013-2014, Christian Therien
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#------------------------------------------------------------------------------
#


"""
SAM_classifier, SID_classifier functions
Spectral Angle Mapper class
Spectral Information Divergence class
"""


import numpy as np


def _normalize(M):
    """
    Normalizes M to be in range [0, 1].

    Parameters:
      M: `numpy array`
          1D, 2D or 3D data.

    Returns: `numpy array`
          Normalized data.
    """

    minVal = np.min(M)
    maxVal = np.max(M)

    Mn = M - minVal;

    if maxVal == minVal:
        return np.zeros(M.shape);
    else:
        return Mn / (maxVal-minVal)


def _single_value_min(data, threshold):
    """
    Use a threshold to extract the minimum value along
    the data y axis.
    """
    min_vec = np.min(data, axis=1)
    amin = np.min(data)
    amax = np.max(data)
    limit = amin + (amax - amin) * threshold
    min_mask = min_vec < limit
    argmin = np.argmin(data, axis=1)
    return (argmin + 1) * min_mask


# For the SAM_classifier function, if you have these errors messages:
"""
C:\somepath\pysptools\classifiers\hs_classification.py:98: RuntimeWarning: invalid value encountered in arccos
  angles = np.arccos(np.round(sum_T_R / mul_T_R, _round_threshold))
C:\somepath\pysptools\classifiers\hs_classification.py:19: RuntimeWarning: invalid value encountered in less
  min_mask = min_vec < limit
"""
# ajust the _round_threshold parameter accordingly,
# a smaller value avoid these errors.
_round_threshold = 7


def SAM_classifier(M, E, threshold=0.1):
    """
    Classify a HSI cube M using the spectral angle mapper
    and a spectral library E
    This function is part of the SAM class.

    Parameters
        M : numpy array
          a HSI array (m x p)

        E : numpy array
          a spectral library (N x p)

        threshold : float
          threshold for classification (0-1)

    Returns : numpy array
          a class map (m)
    """
    def norm_array(m):
        res = np.zeros(m.shape[0])
        for i in range(m.shape[0]):
            res[i] = np.dot(m[i], m[i])
        return np.sqrt(res)

    M = _normalize(M)
    E = _normalize(E)

    TNA = norm_array(M)
    RNA = norm_array(E)
    sum_T_R = np.dot(E, M.T).T
    mul_T_R = np.ndarray((TNA.shape[0], RNA.shape[0]), dtype=float)
    for i in range(TNA.shape[0]):
        mul_T_R[i] = np.multiply(TNA[i],RNA)
    # read above for _round_threshold
    angles = np.arccos(np.round(sum_T_R / mul_T_R, _round_threshold))
    if isinstance(threshold, float):
        cmap = _single_value_min(angles, threshold)
    else:
        return np.argmin(angles, axis=1), angles
    return cmap


def SID_classifier(M, E, threshold=0.1):
    """
    Classify a HSI cube M using the spectral information divergence
    and a spectral library E.

    Parameters
        M : numpy array
          a HSI array (m x p)

        E : numpy array
          a spectral library (N x p)

        threshold : float
          threshold for classification (0-1)

    Returns : numpy array
          a class map (m)
    """
    def prob_vector_array(m):
        pv_array = np.ndarray(shape=m.shape, dtype=float)
        sum_m = np.sum(m, axis=1)
        pv_array[:] = (m.T / sum_m).T
        return pv_array + np.spacing(1)

    mn = M.shape[0]
    N = E.shape[0]
    p = prob_vector_array(M)
    q = prob_vector_array(E)
    sid = np.ndarray((mn, N), dtype=float)
    for i in range(mn):
        pq = q[0:,:] * np.log(q[0:,:] / p[i,:])
        pp = p[i,:] * np.log(p[i,:] / q[0:,:])
        sid[i,:] = np.sum(pp[0:,:] + pq[0:,:], axis=1)
    if isinstance(threshold, float):
        cmap = _single_value_min(sid, threshold)
    else:
        return np.argmin(sid, axis=1), sid
    return cmap
