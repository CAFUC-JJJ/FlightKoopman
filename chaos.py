"""""
by lujing
lujing_cafuc@nuaa.edu.cn

"""
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import distance
import sklearn.linear_model
from sklearn import utils

# from nolitsa

xs, ys, zs = [], [], []


def lorenz_attractor():
    mpl.rcParams["legend.fontsize"] = 10
    fig = plt.figure()
    ax = Axes3D(fig)
    plt.ion()
    a, b, c = 10.0, 28.0, 8.0 / 3.0         # a为普兰特数,b为规范化的瑞利数,c与几何形状相关
    h = 0.01                                # 微分迭代步长
    x0, y0, z0 = 0.1, 0, 0
    for i in range(10000):
        x1 = x0 + h * a * (y0 - x0)
        y1 = y0 + h * (x0 * (b - z0) - y0)
        z1 = z0 + h * (x0 * y0 - c * z0)
        x0, y0, z0 = x1, y1, z1
        xs.append(x0)
        ys.append(y0)
        zs.append(z0)
        plt.cla()
        ax.set_xlabel('X')
        ax.set_xlim(-35, 35)
        ax.set_ylabel('Y')
        ax.set_ylim(-35, 35)
        ax.set_zlabel('Z')
        ax.set_zlim(-1, 50)
        ax.plot(xs, ys, zs, label="Lorenz's strange attractor")
        ax.legend()
        plt.pause(0.001)

def neighbors(y, metric='chebyshev', window=0, maxnum=None):
    """Find nearest neighbors of all points in the given array.

    Finds the nearest neighbors of all points in the given array using
    SciPy's KDTree search.

    Parameters
    ----------
    y : ndarray
        N-dimensional array containing time-delayed vectors.
    metric : string, optional (default = 'chebyshev')
        Metric to use for distance computation.  Must be one of
        "cityblock" (aka the Manhattan metric), "chebyshev" (aka the
        maximum norm metric), or "euclidean".
    window : int, optional (default = 0)
        Minimum temporal separation (Theiler window) that should exist
        between near neighbors.  This is crucial while computing
        Lyapunov exponents and the correlation dimension.
    maxnum : int, optional (default = None (optimum))
        Maximum number of near neighbors that should be found for each
        point.  In rare cases, when there are no neighbors that are at a
        nonzero distance, this will have to be increased (i.e., beyond
        2 * window + 3).

    Returns
    -------
    index : array
        Array containing indices of near neighbors.
    dist : array
        Array containing near neighbor distances.
    """
    if metric == 'cityblock':
        p = 1
    elif metric == 'euclidean':
        p = 2
    elif metric == 'chebyshev':
        p = np.inf
    else:
        raise ValueError('Unknown metric.  Should be one of "cityblock", '
                         '"euclidean", or "chebyshev".')

    tree = KDTree(y)
    n = len(y)

    if not maxnum:
        maxnum = (window + 1) + 1 + (window + 1)
    else:
        maxnum = max(1, maxnum)

    if maxnum >= n:
        raise ValueError('maxnum is bigger than array length.')

    dists = np.empty(n)
    indices = np.empty(n, dtype=int)

    for i, x in enumerate(y):
        for k in range(2, maxnum + 2):
            dist, index = tree.query(x, k=k, p=p)
            valid = (np.abs(index - i) > window) & (dist > 0)

            if np.count_nonzero(valid):
                dists[i] = dist[valid][0]
                indices[i] = index[valid][0]
                break

            if k == (maxnum + 1):
                raise Exception('Could not find any near neighbor with a '
                                'nonzero distance.  Try increasing the '
                                'value of maxnum.')

    return np.squeeze(indices), np.squeeze(dists)

def mle(y, maxt=500, window=10, metric='euclidean', maxnum=None):
    """Estimate the maximum Lyapunov exponent.

    Estimates the maximum Lyapunov exponent (MLE) from a
    multi-dimensional series using the algorithm described by
    Rosenstein et al. (1993).

    Parameters
    ----------
    y : ndarray
        Multi-dimensional real input array containing points in the
        phase space.
    maxt : int, optional (default = 500)
        Maximum time (iterations) up to which the average divergence
        should be computed.
    window : int, optional (default = 10)
        Minimum temporal separation (Theiler window) that should exist
        between near neighbors (see Notes).
    maxnum : int, optional (default = None (optimum))
        Maximum number of near neighbors that should be found for each
        point.  In rare cases, when there are no neighbors that are at a
        nonzero distance, this will have to be increased (i.e., beyond
        2 * window + 3).

    Returns
    -------
    d : array
        Average divergence for each time up to maxt.

    Notes
    -----
    This function does not directly estimate the MLE.  The MLE should be
    estimated by linearly fitting the average divergence (i.e., the
    average of the logarithms of near-neighbor distances) with time.
    It is also important to choose an appropriate Theiler window so that
    the near neighbors do not lie on the same trajectory, in which case
    the estimated MLE will always be close to zero.
    """
    index, dist = utils.neighbors(y, metric=metric, window=window,
                                  maxnum=maxnum)
    m = len(y)
    maxt = min(m - window - 1, maxt)

    d = np.empty(maxt)
    d[0] = np.mean(np.log(dist))

    for t in range(1, maxt):
        t1 = np.arange(t, m)
        t2 = index[:-t] + t

        # Sometimes the nearest point would be farther than (m - maxt)
        # in time.  Such trajectories needs to be omitted.
        valid = t2 < m
        t1, t2 = t1[valid], t2[valid]

        d[t] = np.mean(np.log(utils.dist(y[t1], y[t2], metric=metric)))

    return d

def poly_fit(x, y, degree, fit="RANSAC"):
  # check if we can use RANSAC
  global skpre
  if fit == "RANSAC":
    try:
      # ignore ImportWarnings in sklearn
      with warnings.catch_warnings():
        warnings.simplefilter("ignore", ImportWarning)
        import sklearn.linear_model as sklin
        import sklearn.preprocessing as skpre
    except ImportError:
      warnings.warn(
        "fitting mode 'RANSAC' requires the package sklearn, using"
        + " 'poly' instead",
        RuntimeWarning)
      fit = "poly"

  if fit == "poly":
    return np.polyfit(x, y, degree)
  elif fit == "RANSAC":
    model = sklin.RANSACRegressor(sklin.LinearRegression(fit_intercept=False))
    xdat = np.asarray(x)
    if len(xdat.shape) == 1:
      # interpret 1d-array as list of len(x) samples instead of
      # one sample of length len(x)
      xdat = xdat.reshape(-1, 1)
    polydat = skpre.PolynomialFeatures(degree).fit_transform(xdat)
    try:
      model.fit(polydat, y)
      coef = model.estimator_.coef_[::-1]
    except ValueError:
      warnings.warn(
        "RANSAC did not reach consensus, "
        + "using numpy's polyfit",
        RuntimeWarning)
      coef = np.polyfit(x, y, degree)
    return coef
  else:
    raise ValueError("invalid fitting mode ({})".format(fit))
