from __future__ import division, print_function, absolute_import

from warnings import warn

import numpy as np

from dipy.reconst.dti import fractional_anisotropy, color_fa
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)

from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align.metrics import CCMetric, EMMetric
from dipy.segment.fast_patch_averaging import fast_patch_averaging
from scipy.ndimage.filters import median_filter
try:
    from skimage.filters import threshold_otsu as otsu
except:
    from dipy.segment.threshold import otsu

from scipy.ndimage import binary_dilation, generate_binary_structure


def jaccard_index(mask1, mask2):
    """ Computes Jaccard's measure between two binary 3D data

    Parameters
    ----------
    mask1 : boolean ndarray
        Mask of the first volume
    mask2 : boolean ndarray
        Mask of the second volume, of the same size as mask1

    Returns
    -------
    mea : double
        The output of jaccard's measure
    """

    # check if the images are boolean
    if(not(mask1.dtype == bool and mask2.dtype == bool)):
        raise ValueError(
            "Only 3D boolean arrays supported",
            mask1.dtype,
            mask2.dtype)

    if(mask1.shape != mask2.shape):
        raise ValueError("Dimension mismatch", mask1.shape, mask2.shape)

    intsec = np.zeros(mask1.shape)
    union = np.zeros(mask1.shape)

    intsec = mask1.astype(np.int8) * mask2.astype(np.int8)
    union = (mask1.astype(np.int8) + mask2.astype(np.int8)) > 0

    mea = np.double(np.sum(intsec)) / np.double(np.sum(union))
    return mea


def multi_median(input, median_radius, numpass):
    """ Applies median filter multiple times on input data.

    Parameters
    ----------
    input : ndarray
        The input volume to apply filter on.
    median_radius : int
        Radius (in voxels) of the applied median filter
    numpass: int
        Number of pass of the median filter

    Returns
    -------
    input : ndarray
        Filtered input volume.
    """
    # Array representing the size of the median window in each dimension.
    medarr = np.ones_like(input.shape) * ((median_radius * 2) + 1)

    # Multi pass
    for i in range(0, numpass):
        median_filter(input, medarr, output=input)
    return input


def applymask(vol, mask):
    """ Mask vol with mask.

    Parameters
    ----------
    vol : ndarray
        Array with $V$ dimensions
    mask : ndarray
        Binary mask.  Has $M$ dimensions where $M <= V$. When $M < V$, we
        append $V - M$ dimensions with axis length 1 to `mask` so that `mask`
        will broadcast against `vol`.  In the typical case `vol` can be 4D,
        `mask` can be 3D, and we append a 1 to the mask shape which (via numpy
        broadcasting) has the effect of appling the 3D mask to each 3D slice in
        `vol` (``vol[..., 0]`` to ``vol[..., -1``).

    Returns
    -------
    masked_vol : ndarray
        `vol` multiplied by `mask` where `mask` may have been extended to match
        extra dimensions in `vol`
    """
    mask = mask.reshape(mask.shape + (vol.ndim - mask.ndim) * (1,))
    return vol * mask


def bounding_box(vol):
    """Compute the bounding box of nonzero intensity voxels in the volume.

    Parameters
    ----------
    vol : ndarray
        Volume to compute bounding box on.

    Returns
    -------
    npmins : list
        Array containg minimum index of each dimension
    npmaxs : list
        Array containg maximum index of each dimension
    """
    # Find bounds on first dimension
    temp = vol
    for i in range(vol.ndim - 1):
        temp = temp.any(-1)
    mins = [temp.argmax()]
    maxs = [len(temp) - temp[::-1].argmax()]
    # Check that vol is not all 0
    if mins[0] == 0 and temp[0] == 0:
        warn('No data found in volume to bound. Returning empty bounding box.')
        return [0] * vol.ndim, [0] * vol.ndim
    # Find bounds on remaining dimensions
    if vol.ndim > 1:
        a, b = bounding_box(vol.any(0))
        mins.extend(a)
        maxs.extend(b)
    return mins, maxs


def crop(vol, mins, maxs):
    """Crops the input volume.

    Parameters
    ----------
    vol : ndarray
        Volume to crop.
    mins : array
        Array containg minimum index of each dimension.
    maxs : array
        Array containg maximum index of each dimension.

    Returns
    -------
    vol : ndarray
        The cropped volume.
    """
    return vol[tuple(slice(i, j) for i, j in zip(mins, maxs))]


def median_otsu(input_volume, median_radius=4, numpass=4,
                autocrop=False, vol_idx=None, dilate=None):
    """Simple brain extraction tool method for images from DWI data.

    It uses a median filter smoothing of the input_volumes `vol_idx` and an
    automatic histogram Otsu thresholding technique, hence the name
    *median_otsu*.

    This function is inspired from Mrtrix's bet which has default values
    ``median_radius=3``, ``numpass=2``. However, from tests on multiple 1.5T
    and 3T data     from GE, Philips, Siemens, the most robust choice is
    ``median_radius=4``, ``numpass=4``.

    Parameters
    ----------
    input_volume : ndarray
        ndarray of the brain volume
    median_radius : int
        Radius (in voxels) of the applied median filter (default: 4).
    numpass: int
        Number of pass of the median filter (default: 4).
    autocrop: bool, optional
        if True, the masked input_volume will also be cropped using the
        bounding box defined by the masked data. Should be on if DWI is
        upsampled to 1x1x1 resolution. (default: False).
    vol_idx : None or array, optional
        1D array representing indices of ``axis=3`` of a 4D `input_volume` None
        (the default) corresponds to ``(0,)`` (assumes first volume in
        4D array).

    dilate : None or int, optional
        number of iterations for binary dilation

    Returns
    -------
    maskedvolume : ndarray
        Masked input_volume
    mask : 3D ndarray
        The binary brain mask

    Notes
    -----
    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    """
    if len(input_volume.shape) == 4:
        if vol_idx is not None:
            b0vol = np.mean(input_volume[..., tuple(vol_idx)], axis=3)
        else:
            b0vol = input_volume[..., 0].copy()
    else:
        b0vol = input_volume.copy()
    # Make a mask using a multiple pass median filter and histogram
    # thresholding.
    mask = multi_median(b0vol, median_radius, numpass)
    thresh = otsu(mask)
    mask = mask > thresh

    if dilate is not None:
        cross = generate_binary_structure(3, 1)
        mask = binary_dilation(mask, cross, iterations=dilate)

    # Auto crop the volumes using the mask as input_volume for bounding box
    # computing.
    if autocrop:
        mins, maxs = bounding_box(mask)
        mask = crop(mask, mins, maxs)
        croppedvolume = crop(input_volume, mins, maxs)
        maskedvolume = applymask(croppedvolume, mask)
    else:
        maskedvolume = applymask(input_volume, mask)
    return maskedvolume, mask


def segment_from_cfa(tensor_fit, roi, threshold, return_cfa=False):
    """
    Segment the cfa inside roi using the values from threshold as bounds.

    Parameters
    -------------
    tensor_fit : TensorFit object
        TensorFit object

    roi : ndarray
        A binary mask, which contains the bounding box for the segmentation.

    threshold : array-like
        An iterable that defines the min and max values to use for the
        thresholding.
        The values are specified as (R_min, R_max, G_min, G_max, B_min, B_max)

    return_cfa : bool, optional
        If True, the cfa is also returned.

    Returns
    ----------
    mask : ndarray
        Binary mask of the segmentation.

    cfa : ndarray, optional
        Array with shape = (..., 3), where ... is the shape of tensor_fit.
        The color fractional anisotropy, ordered as a nd array with the last
        dimension of size 3 for the R, G and B channels.
    """

    FA = fractional_anisotropy(tensor_fit.evals)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)  # Clamp the FA to remove degenerate tensors

    cfa = color_fa(FA, tensor_fit.evecs)
    roi = np.asarray(roi, dtype=bool)

    include = ((cfa >= threshold[0::2]) &
               (cfa <= threshold[1::2]) &
               roi[..., None])
    mask = np.all(include, axis=-1)

    if return_cfa:
        return mask, cfa

    return mask


def clean_cc_mask(mask):
    """
    Cleans a segmentation of the corpus callosum so no random pixels
    are included.

    Parameters
    ----------
    mask : ndarray
        Binary mask of the coarse segmentation.

    Returns
    -------
    new_cc_mask : ndarray
        Binary mask of the cleaned segmentation.
    """

    from scipy.ndimage.measurements import label

    new_cc_mask = np.zeros(mask.shape)

    # Flood fill algorithm to find contiguous regions.
    labels, numL = label(mask)

    volumes = [len(labels[np.where(labels == l_idx + 1)])
               for l_idx in np.arange(numL)]
    biggest_vol = np.arange(numL)[np.where(volumes == np.max(volumes))] + 1
    new_cc_mask[np.where(labels == biggest_vol)] = 1

    return new_cc_mask


def template_only_averaging(input_data, template_data, template_mask,
                            patch_radius=1, threshold=0.5):
    """
    The averaging approach which only uses the registered template to extract
    the brain.

    Parameters
    ----------
    input_data : 3D ndarray
        The input data from which the brain has to be extracted
    template_data : 3D ndarray
        The template data
    template_mask : 3D ndarray
        The binary mask of the template data which in 1 where the brain is
        and 0 otherwise (essentially the brain extracted from the template data)
    patch_radius : int
        The patch radius of the windows that needs to be compared
    threshold : double
        Between 0 to 1 which decides the percentage of template mask patch which needs
        to be filled so as judge weather the mask at center voxel of input image
        is 1 or 0
    Returns
    -------
    output_data : 3D ndarray
        The extracted brain from the input data
    output_mask : 3D ndarray
        The brain extraction mask of the input data

    """
    n0 = template_data.shape[0]
    n1 = template_data.shape[1]
    n2 = template_data.shape[2]
    patch_size = patch_radius**3
    output_mask = np.zeros(input_data.shape)
    output_data = np.ones(input_data.shape, dtype=np.float64) * input_data
    # print(template_mask.shape)
    for i in range(patch_radius, n0 - patch_radius):
        for j in range(patch_radius, n1 - patch_radius):
            for k in range(patch_radius, n2 - patch_radius):

                mask_patch = template_mask[i - patch_radius: i + patch_radius + 1,
                                           j - patch_radius: j + patch_radius + 1,
                                           k - patch_radius: k + patch_radius + 1]

                percent = np.double(np.sum(mask_patch)) / np.double(patch_size)

                if percent >= threshold:
                    output_mask[i, j, k] = 1

    output_data[output_mask == 0] = 0
    return [output_data, output_mask]


def brain_extraction(input_data, input_affine, template_data,
                     template_affine, template_mask,
                     patch_radius=1, block_radius=1, parameter=1,
                     threshold=0.5, same_modality=True):
    r"""
    A robust brain extraction which uses a template to reduce the skull intensities.
    The affine information is required because we need to register the template to the
    input data

    Parameters
    ----------
    input_data : 3D ndarray
        The input data from which the brain has to be extracted
    input_affine : ndarray
        The input affine matrix
    template_data : 3D ndarray
        The template data
    template_affine : ndarray
        The template affine matrix
    template_mask : 3D ndarray
        The binary mask of the template data which in 1 where the brain is
        and 0 otherwise (essentially the brain extracted from the template data)
    patch_radius : integer
        The patch size which has to be taken around the voxels for weight computation
    block_radius : integer
        Defining the neighbourhood around the voxel for patch wise similarity searching
    parameter : Double
        Adaptive parameter governing the weights for similar patches
    threshold : Double
        The threshold between 0 to 1 which decides the erosion of the mask boundary
    same_modality : boolean
        The variable which governs the relation between types of modalities of the input
        and the template data. Set it to be true if the modalities are the same (like both
        T1's) and false if not

    Returns
    -------
    output_data : 3D ndarray
        The extracted brain from the input data
    output_mask : 3D ndarray
        The brain extraction mask of the input data

    Notes
    -----
    The implementation takes ideas from both the references given below

    References
    ----------
    .. [Eskildsen11]  Simon Fristed Eskildsen et al., BEaST : Brain extraction based on
                      nonlocal segmentation technique, NeuroImage, vol 59, 2011.

    .. [Lutkenhoff14] Evan S. Lutkenhoff et al., Optimized Brain Extraction for
                      Pathological Brains (OptiBET), PLOS, 2014

    """

    static = input_data
    static_grid2world = input_affine
    moving = template_data
    moving_grid2world = template_affine

    c_of_mass = transform_centers_of_mass(static, static_grid2world,
                                          moving, template_affine)

    # register the template data to input using affine registeration

    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)

    level_iters = [10000, 1000, 100]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]
    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = TranslationTransform3D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affreg.optimize(
        input_data,
        template_data,
        transform,
        params0,
        input_affine,
        template_affine,
        starting_affine=starting_affine)

    transform = RigidTransform3D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=starting_affine)

    transform = AffineTransform3D()
    params0 = None
    starting_affine = rigid.affine
    affine = affreg.optimize(static, moving, transform, params0,
                             static_grid2world, moving_grid2world,
                             starting_affine=starting_affine)

    trans_template = affine.transform(moving)
    trans_mask = affine.transform(template_mask)

    return trans_template, trans_mask

    # Now perform the Non-linear Registration

    pre_align = translation.affine
    if same_modality:
        metric = CCMetric(3)

    else:
        metric = EMMetric(3)

    level_iters = [10, 10, 5]
    sdr = SymmetricDiffeomorphicRegistration(
        metric,
        level_iters,
        ss_sigma_factor=1.7)

    mapping = sdr.optimize(input_data, template_data, input_affine,
                           template_affine, pre_align)

    trans_template = mapping.transform(template_data)
    trans_mask = mapping.transform(template_mask)

    if same_modality:
        [output_data, output_mask] = fast_patch_averaging(
                input_data.astype(np.float64),
                trans_template.astype(np.float64),
                trans_mask.astype(np.float64),
                patch_radius,
                block_radius,
                parameter,
                threshold)
    else:
        [output_data, output_mask] = template_only_averaging(
                input_data.astype(np.float64),
                trans_template.astype(np.float64),
                trans_mask.astype(np.float64),
                patch_radius,
                threshold)

    return output_data, output_mask, trans_template, trans_mask
