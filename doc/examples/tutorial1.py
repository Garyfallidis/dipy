import numpy as np
from dipy.viz import actor, window, widget
# from skimage import exposure
# from skimage.morphology import disk
# from skimage.filters import rank


def histogram_normalization(data):

    g,h = np.histogram(data)
    m = np.zeros((10,3))
    for i in np.array(range(10)):
        m[i,0] = g[i]
        m[i,1] = h[i]
        m[i,2] = h[i+1]


    return m[0,1], m[2,2]


def upper_bound_by_vox_count_percentage(data, percent=0.01):
    """ Find the upper bound for visualization of medical images

    Calculate the histogram of the image and go right to left until you find
    the bound that contains more than a percentage of the image.

    Parameters
    ----------
    data : ndarray
    percent : float

    Returns
    -------
    upper_bound : float
    """

    values, bounds = np.histogram(data, 20)
    total_voxels = np.prod(data.shape)
    agg = 0

    for i in range(len(values) - 1, 0, -1):
        agg += values[i]
        if agg/float(total_voxels) > percent:
            return bounds[i]


def interact_volumes(data, affine, world_coords=True, value_range=None):

    shape = data.shape

    ren = window.Renderer()
    ren.background((0, 0, .6))

    """
    g,h = np.histogram(data)
    m = np.zeros((10,3))
    low = data.min()
    high = data.max()
    for i in np.array(range(10)):
        m[i,0] = g[i]
        m[i,1] = h[i]
        m[i,2] = h[i+1]

    g = np.sort(g)

    for i in np.array(range(10)):
        if g[9] == m[i,0]:
            low = m[i,1]
        if g[7] == m[i,0]:
            high = m[i,2]
    """

    # l,h = histogram_normalization(data)
    if not world_coords:
        image_actor = actor.slicer(data, affine=np.eye(4),
                                   value_range=value_range)
    else:
        image_actor = actor.slicer(data, affine, value_range=value_range)

    slicer_opacity = 1.
    image_actor.opacity(slicer_opacity)
    ren.add(image_actor)

    ren.add(actor.axes((100, 100, 100)))

    show_m = window.ShowManager(ren, size=(1200, 900), reset_camera=False)
    show_m.initialize()

    def change_slice(obj, event):
        z = int(np.round(obj.get_value()))
        image_actor.display_extent(0, shape[0] - 1,
                                   0, shape[1] - 1, z, z)

    slider = widget.slider(show_m.iren, show_m.ren,
                           callback=change_slice,
                           min_value=0,
                           max_value=shape[2] - 1,
                           value=shape[2] / 2,
                           label="Move slice",
                           right_normalized_pos=(.98, 0.6),
                           size=(120, 0), label_format="%0.lf",
                           color=(1., 1., 1.),
                           selected_color=(0.86, 0.33, 1.))

    global size
    size = ren.GetSize()

    def win_callback(obj, event):
        global size
        if size != obj.GetSize():

            slider.place(ren)
            size = obj.GetSize()

    show_m.initialize()
    show_m.add_window_callback(win_callback)
    show_m.render()
    show_m.start()
    return
    # ren.zoom(1.5)
    # ren.reset_clipping_range()

    # window.show(ren)
# window.record(ren, out_path='bundles_and_a_slice.png', size=(1200, 900),
#               reset_camera=False)


# dname = "/Users/tiwanyan/Data/Sherbrooke_Subj_1/"
dname = "/home/eleftherios/Data/3_subjects_for_dipy/subj_1/"
fbvals = dname + "dwi.bval"
fbvecs = dname + "dwi.bvec"
fdwi = dname + "dwi.nii.gz"
ft1 = dname + "t1.nii.gz"

import nibabel as nib

from dipy.io.gradients import read_bvals_bvecs

img = nib.load(fdwi)

data = img.get_data()

t1 = nib.load(ft1).get_data()
affine_t1 = nib.load(ft1).affine

print(data.shape)

affine = img.affine

print(affine)

voxel_size = img.header.get_zooms()[:3]

print(voxel_size)

ondisk_order = nib.aff2axcodes(affine)

print(ondisk_order)

bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)

"""
selem = disk(30)
for i in range(176):
    g,h = np.histogram(t1[:,:,i])
    t1[:,:,i] = np.interp(t1[:,:,i], xp=[h[1],h[5]], fp=[0, 255])
    #t1[:,:,i] = exposure.equalize_hist(t1[:,:,i])
    t1[:,:,i] = rank.equalize(t1[:,:,i],selem = selem)
"""

# m = interact_volumes(data[..., 0], affine, False)

upper_bound = upper_bound_by_vox_count_percentage(t1, 0.005)

n = interact_volumes(t1, affine_t1, True, value_range=(0, upper_bound))

from ipdb import set_trace
set_trace()
