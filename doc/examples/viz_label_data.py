

"""
=====================
Simple volume slicing
=====================

Here we present an example for visualizing slices from 3D images.

"""

from __future__ import division

import os
import numpy as np
import nibabel as nib
from dipy.data import get_data
from dipy.viz import window, actor, ui
from dipy.core.gradients import gradient_table

fimg, fbvals, fbvecs = get_data('small_64D')
bvals = np.load(fbvals)
bvecs = np.load(fbvecs)
bvecs[np.isnan(bvecs)] = 0

gtab = gradient_table(bvals, bvecs)


# vol = np.load('vol.npy')
vol = np.load('vol_reg_cross.npy')


affine = np.eye(4)

"""
Create a Renderer object which holds all the actors which we want to visualize.
"""
from dipy.reconst.dti import TensorModel
tensor_model = TensorModel(gtab)
tensor_fit = tensor_model.fit(vol)
FA = tensor_fit.fa
# print vol
FA[np.isnan(FA)] = 0

data = FA

renderer = window.Renderer()
renderer.background((0.5, 0.5, 0.5))

slice_actor = actor.slicer(data)

show_m = window.ShowManager(renderer, size=(1200, 900))
show_m.initialize()

label_position = ui.TextBlock2D(text='Position:')
label_value = ui.TextBlock2D(text='Value:')

result_position = ui.TextBlock2D(text='')
result_value = ui.TextBlock2D(text='')

panel_picking = ui.Panel2D(center=(200, 120),
                           size=(250, 125),
                           color=(0, 0, 0),
                           opacity=0.75,
                           align="left")

panel_picking.add_element(label_position, 'relative', (0.1, 0.55))
panel_picking.add_element(label_value, 'relative', (0.1, 0.25))

panel_picking.add_element(result_position, 'relative', (0.45, 0.55))
panel_picking.add_element(result_value, 'relative', (0.45, 0.25))

show_m.ren.add(panel_picking)

"""
Add a left-click callback to the slicer. Also disable interpolation so you can
see what you are picking.
"""

renderer.clear()
renderer.projection('parallel')

result_position.message = ''
result_value.message = ''

show_m_mosaic = window.ShowManager(renderer, size=(1200, 900))
show_m_mosaic.initialize()


def left_click_callback_mosaic(obj, ev):
    """Get the value of the clicked voxel and show it in the panel."""
    event_pos = show_m_mosaic.iren.GetEventPosition()

    obj.picker.Pick(event_pos[0],
                    event_pos[1],
                    0,
                    show_m_mosaic.ren)

    i, j, k = obj.picker.GetPointIJK()
    result_position.message = '({}, {}, {})'.format(str(i), str(j), str(k))
    result_value.message = '%.8f' % data[i, j, k]


cnt = 0

X, Y, Z = slice_actor.shape[:3]

rows = 10
cols = 15
border = 10

for j in range(rows):
    for i in range(cols):
        slice_mosaic = slice_actor.copy()
        slice_mosaic.display(None, None, cnt)
        slice_mosaic.SetPosition((X + border) * i,
                                 0.5 * cols * (Y + border) - (Y + border) * j,
                                 0)
        slice_mosaic.SetInterpolate(False)
        slice_mosaic.AddObserver('LeftButtonPressEvent',
                                 left_click_callback_mosaic,
                                 1.0)
        renderer.add(slice_mosaic)
        cnt += 1
        if cnt > Z:
            break
    if cnt > Z:
        break

renderer.reset_camera()
renderer.zoom(1.6)

show_m_mosaic.ren.add(panel_picking)
show_m_mosaic.start()

