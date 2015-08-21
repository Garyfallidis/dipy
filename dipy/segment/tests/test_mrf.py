import numpy as np
import numpy.testing as npt
from dipy.data import get_data
from dipy.sims.voxel import add_noise
from dipy.segment.mrf import (ConstantObservationModel,
                              IteratedConditionalModes)
from dipy.segment.tissue import (TissueClassifierHMRF)


# Load a coronal slice from a T1-weighted MRI
fname = get_data('t1_coronal_slice')
single_slice = np.load(fname)

# Stack a few copies to form a 3D volume
nslices = 5
image = np.zeros(shape=single_slice.shape + (nslices,))
image[..., :nslices] = single_slice[..., None]

# Set up parameters
nclasses = 4
beta = np.float64(0.0)
max_iter = 10
background_noise = True

# Making squares
square = np.zeros((256, 256, 3))
square[42:213, 42:213, :] = 1
square[71:185, 71:185, :] = 2
square[99:157, 99:157, :] = 3

square_gauss = np.zeros((256, 256, 3)) + 0.001
square_gauss = add_noise(square_gauss, 10000, 1, noise_type='gaussian')
square_gauss[42:213, 42:213, :] = 1
noise_1 = np.random.normal(1.001, 0.0001,
                           size=square_gauss[42:213, 42:213, :].shape)
square_gauss[42:213, 42:213, :] = square_gauss[42:213, 42:213, :] + noise_1
square_gauss[71:185, 71:185, :] = 2
noise_2 = np.random.normal(2.001, 0.0001,
                           size=square_gauss[71:185, 71:185, :].shape)
square_gauss[71:185, 71:185, :] = square_gauss[71:185, 71:185, :] + noise_2
square_gauss[99:157, 99:157, :] = 3
noise_3 = np.random.normal(3.001, 0.0001,
                           size=square_gauss[99:157, 99:157, :].shape)
square_gauss[99:157, 99:157, :] = square_gauss[99:157, 99:157, :] + noise_3

square_1 = np.zeros((256, 256, 3)) + 0.001
square_1 = add_noise(square_1, 10000, 1, noise_type='gaussian')
temp_1 = np.random.random_integers(20, size=(171, 171, 3))
temp_1 = np.where(temp_1 < 20, 1, 3)
square_1[42:213, 42:213, :] = temp_1
temp_2 = np.random.random_integers(20, size=(114, 114, 3))
temp_2 = np.where(temp_2 < 19, 2, 1)
square_1[71:185, 71:185, :] = temp_2
temp_3 = np.random.random_integers(20, size=(58, 58, 3))
temp_3 = np.where(temp_3 < 20, 3, 1)
square_1[99:157, 99:157, :] = temp_3


def test_greyscale_image():

    com = ConstantObservationModel()
    icm = IteratedConditionalModes()

    mu, sigma = com.initialize_param_uniform(image, nclasses)
    sigmasq = sigma ** 2
    npt.assert_equal(mu, np.array([0., 0.25, 0.5, 0.75]))
    npt.assert_equal(sigma, np.array([1.0, 1.0, 1.0, 1.0]))
    npt.assert_equal(sigmasq, np.array([1.0, 1.0, 1.0, 1.0]))

    neglogl = com.negloglikelihood(image, mu, sigmasq, nclasses)
    npt.assert_equal((neglogl[100, 100, 1, 0] != neglogl[100, 100, 1, 1]),
                     True)
    npt.assert_equal((neglogl[100, 100, 1, 1] != neglogl[100, 100, 1, 2]),
                     True)
    npt.assert_equal((neglogl[100, 100, 1, 2] != neglogl[100, 100, 1, 3]),
                     True)
    npt.assert_equal((neglogl[100, 100, 1, 1] != neglogl[100, 100, 1, 3]),
                     True)

    initial_segmentation = icm.initialize_maximum_likelihood(neglogl)
    npt.assert_equal(initial_segmentation.max(), nclasses - 1)
    npt.assert_equal(initial_segmentation.min(), 0)

    PLN = com.prob_neighborhood(image, initial_segmentation, beta, nclasses)
    npt.assert_equal(PLN.all() >= 0.0, True)
    npt.assert_equal(PLN.all() <= 1.0, True)

    if beta == 0.0:
        npt.assert_equal((PLN[50, 50, 1, 0] == 0.25), True)
        npt.assert_equal((PLN[50, 50, 1, 1] == 0.25), True)
        npt.assert_equal((PLN[50, 50, 1, 2] == 0.25), True)
        npt.assert_equal((PLN[50, 50, 1, 3] == 0.25), True)
        npt.assert_equal((PLN[147, 129, 1, 0] == 0.25), True)
        npt.assert_equal((PLN[147, 129, 1, 1] == 0.25), True)
        npt.assert_equal((PLN[147, 129, 1, 2] == 0.25), True)
        npt.assert_equal((PLN[147, 129, 1, 3] == 0.25), True)
        npt.assert_equal((PLN[61, 152, 1, 0] == 0.25), True)
        npt.assert_equal((PLN[61, 152, 1, 1] == 0.25), True)
        npt.assert_equal((PLN[61, 152, 1, 2] == 0.25), True)
        npt.assert_equal((PLN[61, 152, 1, 3] == 0.25), True)
        npt.assert_equal((PLN[100, 100, 1, 0] == 0.25), True)
        npt.assert_equal((PLN[100, 100, 1, 1] == 0.25), True)
        npt.assert_equal((PLN[100, 100, 1, 2] == 0.25), True)
        npt.assert_equal((PLN[100, 100, 1, 3] == 0.25), True)

    PLY = com.prob_image(image, nclasses, mu, sigmasq, PLN)
    npt.assert_equal(PLY.all() >= 0.0, True)
    npt.assert_equal(PLY.all() <= 1.0, True)

    mu_upd, sigmasq_upd = com.update_param(image, PLY, mu, nclasses)
    npt.assert_equal(mu_upd != mu, True)
    npt.assert_equal(sigmasq_upd != sigmasq, True)

    icm_segmentation, energy = icm.icm_ising(neglogl, beta,
                                             initial_segmentation)
    npt.assert_equal(np.abs(np.sum(icm_segmentation)) != 0, True)
    npt.assert_equal(icm_segmentation.max(), nclasses - 1)
    npt.assert_equal(icm_segmentation.min(), 0)


def test_greyscale_iter():

    max_iter = 15
    beta = np.float64(0.1)

    com = ConstantObservationModel()
    icm = IteratedConditionalModes()

    mu, sigma = com.initialize_param_uniform(image, nclasses)
    sigmasq = sigma ** 2
    neglogl = com.negloglikelihood(image, mu, sigmasq, nclasses)
    initial_segmentation = icm.initialize_maximum_likelihood(neglogl)
    npt.assert_equal(initial_segmentation.max(), nclasses - 1)
    npt.assert_equal(initial_segmentation.min(), 0)

    mu, sigma, sigmasq = com.seg_stats(image, initial_segmentation, nclasses)
    npt.assert_equal(mu.all() >= 0, True)
    npt.assert_equal(sigmasq.all() >= 0, True)

    if background_noise:
        zero = np.zeros_like(image) + 0.001
        zero_noise = add_noise(zero, 10000, 1, noise_type='gaussian')
        image_gauss = np.where(image == 0, zero_noise, image)
    else:
        image_gauss = image

    final_segmentation = np.empty_like(image)
    seg_init = initial_segmentation.copy()
    energies = []

    for i in range(max_iter):

        PLN = com.prob_neighborhood(image_gauss, initial_segmentation, beta,
                                    nclasses)
        npt.assert_equal(PLN.all() >= 0.0, True)

        if beta == 0.0:

            npt.assert_equal((PLN[50, 50, 1, 0] == 0.25), True)
            npt.assert_equal((PLN[50, 50, 1, 1] == 0.25), True)
            npt.assert_equal((PLN[50, 50, 1, 2] == 0.25), True)
            npt.assert_equal((PLN[50, 50, 1, 3] == 0.25), True)
            npt.assert_equal((PLN[147, 129, 1, 0] == 0.25), True)
            npt.assert_equal((PLN[147, 129, 1, 1] == 0.25), True)
            npt.assert_equal((PLN[147, 129, 1, 2] == 0.25), True)
            npt.assert_equal((PLN[147, 129, 1, 3] == 0.25), True)
            npt.assert_equal((PLN[61, 152, 1, 0] == 0.25), True)
            npt.assert_equal((PLN[61, 152, 1, 1] == 0.25), True)
            npt.assert_equal((PLN[61, 152, 1, 2] == 0.25), True)
            npt.assert_equal((PLN[61, 152, 1, 3] == 0.25), True)
            npt.assert_equal((PLN[100, 100, 1, 0] == 0.25), True)
            npt.assert_equal((PLN[100, 100, 1, 1] == 0.25), True)
            npt.assert_equal((PLN[100, 100, 1, 2] == 0.25), True)
            npt.assert_equal((PLN[100, 100, 1, 3] == 0.25), True)

        PLY = com.prob_image(image_gauss, nclasses, mu, sigmasq, PLN)
        npt.assert_equal(PLY.all() >= 0.0, True)
        npt.assert_equal(PLY[50, 50, 1, 0] > PLY[50, 50, 1, 1], True)
        npt.assert_equal(PLY[50, 50, 1, 0] > PLY[50, 50, 1, 2], True)
        npt.assert_equal(PLY[50, 50, 1, 0] > PLY[50, 50, 1, 3], True)
        npt.assert_equal(PLY[100, 100, 1, 3] > PLY[100, 100, 1, 0], True)
        npt.assert_equal(PLY[100, 100, 1, 3] > PLY[100, 100, 1, 1], True)
        npt.assert_equal(PLY[100, 100, 1, 3] > PLY[100, 100, 1, 2], True)

        mu_upd, sigmasq_upd = com.update_param(image_gauss, PLY, mu, nclasses)
        npt.assert_equal(mu_upd.all() >= 0.0, True)
        npt.assert_equal(sigmasq_upd.all() >= 0.0, True)

        negll = com.negloglikelihood(image_gauss,
                                     mu_upd, sigmasq_upd, nclasses)
        npt.assert_equal(negll[50, 50, 1, 0] < negll[50, 50, 1, 1], True)
        npt.assert_equal(negll[50, 50, 1, 0] < negll[50, 50, 1, 2], True)
        npt.assert_equal(negll[50, 50, 1, 0] < negll[50, 50, 1, 3], True)
        npt.assert_equal(negll[100, 100, 1, 3] < negll[100, 100, 1, 0], True)
        npt.assert_equal(negll[100, 100, 1, 3] < negll[100, 100, 1, 1], True)
        npt.assert_equal(negll[100, 100, 1, 3] < negll[100, 100, 1, 2], True)

        final_segmentation, energy = icm.icm_ising(negll, beta,
                                                   initial_segmentation)
        print(energy[energy > -np.inf].sum())
        energies.append(energy[energy > -np.inf].sum())

        initial_segmentation = final_segmentation.copy()
        mu = mu_upd.copy()
        sigmasq = sigmasq_upd.copy()

    npt.assert_equal(energies[-1] < energies[0], True)
#    npt.assert_equal(energies[-1] < energies[-50], True)

    difference_map = np.abs(seg_init - final_segmentation)
    npt.assert_equal(np.abs(np.sum(difference_map)) != 0, True)


def test_square_iter():

    com = ConstantObservationModel()
    icm = IteratedConditionalModes()

    initial_segmentation = square

    mu, sigma, sigmasq = com.seg_stats(square_gauss, initial_segmentation,
                                       nclasses)
    npt.assert_equal(mu.all() >= 0, True)
    npt.assert_equal(sigmasq.all() >= 0, True)

    final_segmentation = np.empty_like(square_gauss)
    seg_init = initial_segmentation.copy()
    energies = []

    for i in range(max_iter):

        print('\n')
        print('>> Iteration: ' + str(i))
        print('\n')

        PLN = com.prob_neighborhood(square_gauss, initial_segmentation, beta,
                                    nclasses)
        npt.assert_equal(PLN.all() >= 0.0, True)

        if beta == 0.0:

            npt.assert_equal((PLN[25, 25, 1, 0] == 0.25), True)
            npt.assert_equal((PLN[25, 25, 1, 1] == 0.25), True)
            npt.assert_equal((PLN[25, 25, 1, 2] == 0.25), True)
            npt.assert_equal((PLN[25, 25, 1, 3] == 0.25), True)
            npt.assert_equal((PLN[50, 50, 1, 0] == 0.25), True)
            npt.assert_equal((PLN[50, 50, 1, 1] == 0.25), True)
            npt.assert_equal((PLN[50, 50, 1, 2] == 0.25), True)
            npt.assert_equal((PLN[50, 50, 1, 3] == 0.25), True)
            npt.assert_equal((PLN[90, 90, 1, 0] == 0.25), True)
            npt.assert_equal((PLN[90, 90, 1, 1] == 0.25), True)
            npt.assert_equal((PLN[90, 90, 1, 2] == 0.25), True)
            npt.assert_equal((PLN[90, 90, 1, 3] == 0.25), True)
            npt.assert_equal((PLN[125, 125, 1, 0] == 0.25), True)
            npt.assert_equal((PLN[125, 125, 1, 1] == 0.25), True)
            npt.assert_equal((PLN[125, 125, 1, 2] == 0.25), True)
            npt.assert_equal((PLN[125, 125, 1, 3] == 0.25), True)

        PLY = com.prob_image(square_gauss, nclasses, mu, sigmasq, PLN)
        npt.assert_equal(PLY.all() >= 0.0, True)
        npt.assert_equal(PLY[25, 25, 1, 0] > PLY[25, 25, 1, 1], True)
        npt.assert_equal(PLY[25, 25, 1, 0] > PLY[25, 25, 1, 2], True)
        npt.assert_equal(PLY[25, 25, 1, 0] > PLY[25, 25, 1, 3], True)
        npt.assert_equal(PLY[125, 125, 1, 3] > PLY[125, 125, 1, 0], True)
        npt.assert_equal(PLY[125, 125, 1, 3] > PLY[125, 125, 1, 1], True)
        npt.assert_equal(PLY[125, 125, 1, 3] > PLY[125, 125, 1, 2], True)

        mu_upd, sigmasq_upd = com.update_param(square_gauss, PLY, mu, nclasses)
        npt.assert_equal(mu_upd.all() >= 0.0, True)
        npt.assert_equal(sigmasq_upd.all() >= 0.0, True)

        negll = com.negloglikelihood(square_gauss,
                                     mu_upd, sigmasq_upd, nclasses)
        npt.assert_equal(negll[25, 25, 1, 0] < negll[25, 25, 1, 1], True)
        npt.assert_equal(negll[25, 25, 1, 0] < negll[25, 25, 1, 2], True)
        npt.assert_equal(negll[25, 25, 1, 0] < negll[25, 25, 1, 3], True)
        npt.assert_equal(negll[100, 100, 1, 3] < negll[125, 125, 1, 0], True)
        npt.assert_equal(negll[100, 100, 1, 3] < negll[125, 125, 1, 1], True)
        npt.assert_equal(negll[100, 100, 1, 3] < negll[125, 125, 1, 2], True)

        final_segmentation, energy = icm.icm_ising(negll, beta,
                                                   initial_segmentation)
                
        energies.append(energy[energy > -np.inf].sum())

        initial_segmentation = final_segmentation.copy()
        mu = mu_upd.copy()
        sigmasq = sigmasq_upd.copy()

    np.set_printoptions(3, suppress=True)
    print(np.diff(energies) * 0.0001)

    difference_map = np.abs(seg_init - final_segmentation)
    npt.assert_equal(np.abs(np.sum(difference_map)) == 0, True)


def test_icm_square():

    com = ConstantObservationModel()
    icm = IteratedConditionalModes()

    initial_segmentation = square.copy()

    mu, sigma, sigmasq = com.seg_stats(square_1, initial_segmentation, nclasses)
    npt.assert_equal(mu.all() >= 0, True)
    npt.assert_equal(sigmasq.all() >= 0, True)

    negll = com.negloglikelihood(square_1, mu, sigmasq, nclasses)

    final_segmentation_1 = np.empty_like(square_1)
    final_segmentation_2 = np.empty_like(square_1)

    beta = 0.0

    for i in range(max_iter):

        print('\n')
        print('>> Iteration: ' + str(i))
        print('\n')

        final_segmentation_1, energy_1 = icm.icm_ising(negll, beta, initial_segmentation)
        initial_segmentation = final_segmentation_1.copy()

    beta = 2
    initial_segmentation = square.copy()

    for j in range(max_iter):

        print('\n')
        print('>> Iteration: ' + str(j))
        print('\n')

        final_segmentation_2, energy_2 = icm.icm_ising(negll, beta, 
                                                       initial_segmentation)
        initial_segmentation = final_segmentation_2.copy()

    difference_map = np.abs(final_segmentation_1 - final_segmentation_2)
    npt.assert_equal(np.abs(np.sum(difference_map)) != 0, True)


def test_classify():

    imgseg = TissueClassifierHMRF()

    beta = 0.1
    max_iter = 10
    
    npt.assert_equal(image.max(), 1)
    npt.assert_equal(image.min(), 0)

    seg_init, seg_final, PVE = imgseg.classify(image, nclasses,
                                               beta, max_iter)
    
    npt.assert_equal(seg_final.max(), nclasses)
    npt.assert_equal(seg_final.min(), 0)
        
    imgseg = TissueClassifierHMRF(save_history=True)
    
    seg_init, seg_final, PVE = imgseg.classify(200 * image, nclasses,
                                               beta, max_iter)
    
    npt.assert_equal(seg_final.max(), nclasses)
    npt.assert_equal(seg_final.min(), 0)
   
    npt.assert_equal(len(imgseg.segmentations), max_iter)  


if __name__ == '__main__':
    pass
    npt.run_module_suite()
    
    
    
    
    
    
    
