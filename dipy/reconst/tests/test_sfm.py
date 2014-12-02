import numpy as np
import numpy.testing as npt
import nibabel as nib
import dipy.reconst.sfm as sfm
import dipy.data as dpd
import dipy.core.gradients as grad
import dipy.sims.voxel as sims
import dipy.core.optimize as opt


def test_design_matrix():
    data, gtab = dpd.dsi_voxels()
    sphere = dpd.get_sphere()
    # Make it with NNLS, so that it gets tested regardless of sklearn
    sparse_fascicle_model = sfm.SparseFascicleModel(gtab, sphere, solver='NNLS')
    npt.assert_equal(sparse_fascicle_model.design_matrix.shape,
                     (np.sum(~gtab.b0s_mask), sphere.vertices.shape[0]))


@npt.dec.skipif(not sfm.has_sklearn)
def test_SparseFascicleModel():
    fdata, fbvals, fbvecs = dpd.get_data()
    data = nib.load(fdata).get_data()
    gtab = grad.gradient_table(fbvals, fbvecs)
    sfmodel = sfm.SparseFascicleModel(gtab)
    sffit1 = sfmodel.fit(data[0, 0, 0])
    sphere = dpd.get_sphere('symmetric642')
    odf1 = sffit1.odf(sphere)
    pred1 = sffit1.predict(gtab)
    mask = np.ones(data.shape[:-1])
    sffit2 = sfmodel.fit(data, mask)
    pred2 = sffit2.predict(gtab)
    odf2 = sffit2.odf(sphere)
    sffit3 = sfmodel.fit(data)
    pred3 = sffit3.predict(gtab)
    odf3 = sffit3.odf(sphere)

    npt.assert_almost_equal(pred3, pred2, decimal=2)
    npt.assert_almost_equal(pred3[0,0,0], pred1, decimal=2)
    npt.assert_almost_equal(odf3[0,0,0], odf1, decimal=2)

    # Fit zeros and you will get back zeros
    npt.assert_almost_equal(sfmodel.fit(np.zeros(data[0,0,0].shape)).beta,
                            np.zeros(sfmodel.design_matrix[0].shape[-1]))


@npt.dec.skipif(not sfm.has_sklearn)
def test_predict():
    SNR = 1000
    S0 = 1
    _, fbvals, fbvecs = dpd.get_data('small_64D')
    bvals = np.load(fbvals)
    bvecs = np.load(fbvecs)
    gtab = grad.gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))
    angles = [(0, 0), (60, 0)]
    S, sticks = sims.multi_tensor(gtab, mevals, S0, angles=angles,
                                  fractions=[50, 50], snr=SNR)

    sfmodel = sfm.SparseFascicleModel(gtab, response=[0.0015, 0.0003, 0.0003])
    sffit = sfmodel.fit(S)
    pred = sffit.predict()
    npt.assert_almost_equal(pred, S, decimal=1)


def test_SparseFascicleModel_stick():
    fdata, fbvals, fbvecs = dpd.get_data()
    data = nib.load(fdata).get_data()
    gtab = grad.gradient_table(fbvals, fbvecs)
    sfmodel = sfm.SparseFascicleModel(gtab, solver='NNLS',
                                      response=[0.001, 0, 0])
    sffit1 = sfmodel.fit(data[0, 0, 0])
    sphere = dpd.get_sphere('symmetric642')
    odf1 = sffit1.odf(sphere)
    pred1 = sffit1.predict(gtab)

    SNR = 1000
    S0 = 1
    mevals = np.array(([0.001, 0, 0],
                       [0.001, 0, 0]))
    angles = [(0, 0), (60, 0)]
    S, sticks = sims.multi_tensor(gtab, mevals, S0, angles=angles,
                                  fractions=[50, 50], snr=SNR)

    sfmodel = sfm.SparseFascicleModel(gtab, solver='NNLS',
                                      response=[0.001, 0, 0])
    sffit = sfmodel.fit(S)
    pred = sffit.predict()
    npt.assert_almost_equal(pred, S, decimal=1)


def test_SparseFascicleModel_SKLearnlinearsolver():
    class SillySolver(opt.SKLearnLinearSolver):
        def fit(self, X, y):
            self.coef_ = np.ones(X.shape[-1])

    class EvenSillierSolver(object):
        def fit(self, X, y):
            self.coef_ = np.ones(X.shape[-1])

    fdata, fbvals, fbvecs = dpd.get_data()
    data = nib.load(fdata).get_data()
    gtab = grad.gradient_table(fbvals, fbvecs)
    sfmodel = sfm.SparseFascicleModel(gtab, solver=SillySolver())

    npt.assert_(isinstance(sfmodel.solver, SillySolver))
    npt.assert_raises(ValueError,
                      sfm.SparseFascicleModel, gtab, solver=EvenSillierSolver())