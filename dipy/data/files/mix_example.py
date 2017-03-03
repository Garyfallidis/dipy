# -*- coding: utf-8 -*-
"""
Created on Thu Mar 02 15:37:46 2017

@author: elef
"""

import numpy as np
from os.path import join
import nibabel as nib
from dipy.core.gradients import gradient_table
from numpy.testing import assert_equal, assert_almost_equal, assert_array_almost_equal


# dname = 'C:\Users\elef\Devel\dipy\dipy\data\files'
fname = 'Aax_synth_data.nii'
fscanner = 'ActiveAx_xdata.txt'

params = np.loadtxt(fscanner)

img = nib.load(fname)
data = img.get_data()
affine = img.affine

bvecs = params[:, 0:3]
G = params[:, 3] / 10 ** 6  # gradient strength
big_delta = params[:, 4]
small_delta = params[:, 5]
te = params[:, 6]
gamma = 2.675987 * 10 ** 8

bvals = gamma ** 2 * G ** 2 * small_delta ** 2 * (big_delta - small_delta / 3.)
bvals = bvals

print(bvals * 10 ** 6)

gtab = gradient_table(bvals, bvecs, big_delta=big_delta,
                      small_delta=small_delta,
                      b0_threshold=0, atol=1e-2)


def activax_exvivo_params(x, bvals, bvecs, G, small_delta, big_delta,
                          gamma=gamma,
                          D_intra=0.6 * 10 ** 3, D_iso=2 * 10 ** 3,
                          debug=False):
    """ Aax_exvivo_nlin
    """

    sinT = np.sin(x[0])
    cosT = np.cos(x[0])
    sinP = np.sin(x[1])
    cosP = np.cos(x[1])
    n = np.array([cosP * sinT, sinP * sinT, cosT])

    # Cylinder
    L = bvals * D_intra
    print(L)
    print(bvecs.shape)
    print(n.shape)
    L1 = L * np.dot(bvecs, n) ** 2
    am = np.array([1.84118307861360, 5.33144196877749,
                   8.53631578218074, 11.7060038949077,
                   14.8635881488839, 18.0155278304879,
                   21.1643671187891, 24.3113254834588,
                   27.4570501848623, 30.6019229722078,
                   33.7461812269726, 36.8899866873805,
                   40.0334439409610, 43.1766274212415,
                   46.3195966792621, 49.4623908440429,
                   52.6050411092602, 55.7475709551533,
                   58.8900018651876, 62.0323477967829,
                   65.1746202084584, 68.3168306640438,
                   71.4589869258787, 74.6010956133729,
                   77.7431620631416, 80.8851921057280,
                   84.0271895462953, 87.1691575709855,
                   90.3110993488875, 93.4530179063458,
                   96.5949155953313, 99.7367932203820,
                   102.878653768715, 106.020498619541,
                   109.162329055405, 112.304145672561,
                   115.445950418834, 118.587744574512,
                   121.729527118091, 124.871300497614,
                   128.013065217171, 131.154821965250,
                   134.296570328107, 137.438311926144,
                   140.580047659913, 143.721775748727,
                   146.863498476739, 150.005215971725,
                   153.146928691331, 156.288635801966,
                   159.430338769213, 162.572038308643,
                   165.713732347338, 168.855423073845,
                   171.997111729391, 175.138794734935,
                   178.280475036977, 181.422152668422,
                   184.563828222242, 187.705499575101])

    am2 = (am / x[2]) ** 2

    summ = np.zeros((len(bvals), len(am)))

    for i in range(len(am)):
        num = (2 * D_intra * am2[i] * small_delta) -2 + \
              (2 * np.exp(-(D_intra * am2[i] * small_delta))) + \
              (2 * np.exp(-(D_intra * am2[i] * big_delta))) - \
              (np.exp(-(D_intra * am2[i] * (big_delta - small_delta)))) - \
              (np.exp(-(D_intra * am2[i] * (big_delta + small_delta))))

        denom = (D_intra ** 2) * (am2[i] ** 3) * ((x[2]) ** 2 * am2[i] - 1)
        summ[:, i] = num / denom

    summ_rows = np.sum(summ, axis=1)
    g_per = np.zeros(bvals.shape)

    for i in range(len(bvecs)):
        g_per[i] = np.dot(bvecs[i, :], bvecs[i, :]) - \
                   np.dot(bvecs[i, :], n) ** 2

    L2 = 2 * (g_per * gamma ** 2) * summ_rows * G ** 2

    yhat_cylinder = L1 + L2

    # zeppelin
    yhat_zeppelin = bvals * ((D_intra - (D_intra * (1 - x[3]))) * (np.dot(bvecs, n) ** 2) + (D_intra * (1 - x[3])))

    # ball
    yhat_ball = (D_iso * bvals)

    # dot
    yhat_dot = np.dot(bvecs, np.array([0, 0, 0]))

    if debug:
        return L1, summ, summ_rows, g_per, L2, yhat_cylinder, yhat_zeppelin, \
            yhat_ball, yhat_dot
    return yhat_cylinder, yhat_zeppelin, yhat_ball, yhat_dot


def test_activax_exvivo_model():

    x = np.array([0.5, 0.5, 10, 0.8])
    sigma = 0.05
    L1, summ, summ_rows, gper, L2, yhat_cylinder, \
        yhat_zeppelin, yhat_ball, yhat_one = activax_exvivo_params(
            data, x, bvals, bvecs, G, small_delta, big_delta,
            gamma=gamma, D_intra=0.6 * 10 ** 3, D_iso=2 * 10 ** 3,
            debug=True)

    assert_almost_equal(L1[3], 3.96735278865)
    assert_almost_equal(L1[5], 5.54561445859)

    assert_equal(summ.shape, (372, 60))
    assert_almost_equal(summ[3, 3], 2.15633236279e-07)
    assert_almost_equal(summ[20, 2], 1.3498305300e-06)

    assert_equal(summ_rows.shape, (372,))

    print(summ[3, 3])

    assert_equal(gper.shape, (372,))
    assert_almost_equal(gper[39], 0.004856614915350943)
    assert_almost_equal(gper[45], 0.60240287122917402)

    print(L2.shape)
    print(L2[5])

    assert_equal(L2.shape, (372,))
    assert_almost_equal(L2[5], 1.33006433687)

    print(yhat_zeppelin.shape)
    assert_equal(yhat_zeppelin.shape, (372,))
    assert_almost_equal(yhat_zeppelin[6], 2.23215830075)

    assert_almost_equal(yhat_ball[25], 26.3995293508)


def activax_exvivo_model(x, bvals, bvecs, G, small_delta, big_delta,
                         gamma=gamma,
                         D_intra=0.6 * 10 ** 3, D_iso=2 * 10 ** 3,
                         debug=False):

    res = activax_exvivo_params(x, bvals, bvecs, G, small_delta, big_delta,
                                gamma=gamma,
                                D_intra=0.6 * 10 ** 3, D_iso=2 * 10 ** 3,
                                debug=False)

    yhat_cylinder, yhat_zeppelin, yhat_ball, yhat_dot = res

    phi = np.vstack([yhat_cylinder, yhat_zeppelin, yhat_ball, yhat_dot]).T
    phi = np.ascontiguousarray(phi)

    return np.exp(-phi)


def activeax_cost_one(data, phi, sigma):

    phi_mp = np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T) # moore-penrose
    f = np.dot(phi_mp, data)
    yhat = np.dot(phi, f) - sigma
    return np.dot((data - yhat).T, data - yhat)


def test_activax_exvivo_model():
    x = np.array([0.5, 0.5, 10, 0.8])
    sigma = 0.05 # offset gaussian constant (not really needed)
    phi = activax_exvivo_model(x, bvals, bvecs, G,
                               small_delta, big_delta)
    print(phi.shape)
    assert_equal(phi.shape, (372, 4))
    error_one =  activeax_cost_one(data[0, 0, 0], phi, sigma)

    assert_almost_equal(error_one, 2.5070277363920468)


test_activax_exvivo_model()





test_activax_exvivo_model()