import numpy as np
from dipy.workflows.workflow import Workflow
from dipy.io.streamline import load_trk, save_trk
from dipy.tracking.streamline import transform_streamlines, length, Streamlines
from dipy.io.image import load_nifti, save_nifti
from dipy.segment.clustering import qbx_and_merge
from dipy.viz import actor, window, ui
from dipy.viz.window import vtk
from dipy.viz.utils import get_polydata_lines


def check_range(streamline, lt, gt):
    length_s = length(streamline)
    if (length_s < lt) & (length_s > gt):
        return True
    else:
        return False


def build_label(text):
    label = ui.TextBlock2D()
    label.message = text
    label.font_size = 18
    label.font_family = 'Arial'
    label.justification = 'left'
    label.bold = False
    label.italic = False
    label.shadow = False
    label.actor.GetTextProperty().SetBackgroundColor(0, 0, 0)
    label.actor.GetTextProperty().SetBackgroundOpacity(0.0)
    label.color = (1, 1, 1)

    return label


def slicer_panel(renderer, data, affine, world_coords):

    shape = data.shape
    if not world_coords:
        image_actor_z = actor.slicer(data, affine=np.eye(4))
    else:
        image_actor_z = actor.slicer(data, affine)

    slicer_opacity = 0.6
    image_actor_z.opacity(slicer_opacity)

    image_actor_x = image_actor_z.copy()
    x_midpoint = int(np.round(shape[0] / 2))
    image_actor_x.display_extent(x_midpoint,
                                 x_midpoint, 0,
                                 shape[1] - 1,
                                 0,
                                 shape[2] - 1)

    image_actor_y = image_actor_z.copy()
    y_midpoint = int(np.round(shape[1] / 2))
    image_actor_y.display_extent(0,
                                 shape[0] - 1,
                                 y_midpoint,
                                 y_midpoint,
                                 0,
                                 shape[2] - 1)

    renderer.add(image_actor_z)
    renderer.add(image_actor_x)
    renderer.add(image_actor_y)

    line_slider_z = ui.LineSlider2D(min_value=0,
                                    max_value=shape[2] - 1,
                                    initial_value=shape[2] / 2,
                                    text_template="{value:.0f}",
                                    length=140)

    line_slider_x = ui.LineSlider2D(min_value=0,
                                    max_value=shape[0] - 1,
                                    initial_value=shape[0] / 2,
                                    text_template="{value:.0f}",
                                    length=140)

    line_slider_y = ui.LineSlider2D(min_value=0,
                                    max_value=shape[1] - 1,
                                    initial_value=shape[1] / 2,
                                    text_template="{value:.0f}",
                                    length=140)

    opacity_slider = ui.LineSlider2D(min_value=0.0,
                                     max_value=1.0,
                                     initial_value=slicer_opacity,
                                     length=140)

    def change_slice_z(i_ren, obj, slider):
        z = int(np.round(slider.value))
        image_actor_z.display_extent(0, shape[0] - 1,
                                     0, shape[1] - 1, z, z)

    def change_slice_x(i_ren, obj, slider):
        x = int(np.round(slider.value))
        image_actor_x.display_extent(x, x, 0, shape[1] - 1, 0,
                                     shape[2] - 1)

    def change_slice_y(i_ren, obj, slider):
        y = int(np.round(slider.value))
        image_actor_y.display_extent(0, shape[0] - 1, y, y,
                                     0, shape[2] - 1)

    def change_opacity(i_ren, obj, slider):
        slicer_opacity = slider.value
        image_actor_z.opacity(slicer_opacity)
        image_actor_x.opacity(slicer_opacity)
        image_actor_y.opacity(slicer_opacity)

    line_slider_z.add_callback(line_slider_z.slider_disk,
                               "MouseMoveEvent",
                               change_slice_z)
    line_slider_z.add_callback(line_slider_z.slider_line,
                               "LeftButtonPressEvent",
                               change_slice_z)

    line_slider_x.add_callback(line_slider_x.slider_disk,
                               "MouseMoveEvent",
                               change_slice_x)
    line_slider_x.add_callback(line_slider_x.slider_line,
                               "LeftButtonPressEvent",
                               change_slice_x)

    line_slider_y.add_callback(line_slider_y.slider_disk,
                               "MouseMoveEvent",
                               change_slice_y)
    line_slider_y.add_callback(line_slider_y.slider_line,
                               "LeftButtonPressEvent",
                               change_slice_y)

    opacity_slider.add_callback(opacity_slider.slider_disk,
                                "MouseMoveEvent",
                                change_opacity)
    opacity_slider.add_callback(opacity_slider.slider_line,
                                "LeftButtonPressEvent",
                                change_opacity)

    line_slider_label_z = build_label(text="Z Slice")
    line_slider_label_x = build_label(text="X Slice")
    line_slider_label_y = build_label(text="Y Slice")
    opacity_slider_label = build_label(text="Opacity")

    panel = ui.Panel2D(center=(1030, 120),
                       size=(300, 200),
                       color=(1, 1, 1),
                       opacity=0.1,
                       align="right")

    panel.add_element(line_slider_label_x, 'relative', (0.1, 0.75))
    panel.add_element(line_slider_x, 'relative', (0.65, 0.8))
    panel.add_element(line_slider_label_y, 'relative', (0.1, 0.55))
    panel.add_element(line_slider_y, 'relative', (0.65, 0.6))
    panel.add_element(line_slider_label_z, 'relative', (0.1, 0.35))
    panel.add_element(line_slider_z, 'relative', (0.65, 0.4))
    panel.add_element(opacity_slider_label, 'relative', (0.1, 0.15))
    panel.add_element(opacity_slider, 'relative', (0.65, 0.2))

    renderer.add(panel)
    return panel


def apply_shader(actor):
    global opacity_level

    gl_mapper = actor.GetMapper()

    gl_mapper.AddShaderReplacement(
        vtk.vtkShader.Vertex,
        "//VTK::ValuePass::Impl",  # replace the normal block
        False,
        "//VTK::ValuePass::Impl\n",  # we still want the default
        False)

    gl_mapper.AddShaderReplacement(
        vtk.vtkShader.Fragment,
        "//VTK::Light::Impl",
        True,
        "//VTK::Light::Impl\n"
        "if (selected == 1){\n"
        " fragOutput0 = fragOutput0 + vec4(0.2, 0.2, 0, opacity_level);\n"
        "}\n",
        False)

    gl_mapper.AddShaderReplacement(
        vtk.vtkShader.Fragment,
        "//VTK::Coincident::Dec",
        True,
        "//VTK::Coincident::Dec\n"
        "uniform float selected;\n"
        "uniform float opacity_level;\n",
        False)

    @window.vtk.calldata_type(window.vtk.VTK_OBJECT)
    def vtk_shader_callback(caller, event, calldata=None):
        global opacity_level, cluster_actors
        program = calldata
        if program is not None:
            try:
                program.SetUniformf("selected",
                                    centroid_actors[actor]['selected'])
            except KeyError:
                pass
            try:
                program.SetUniformf("selected",
                                    cluster_actors[actor]['selected'])
            except KeyError:
                pass
            program.SetUniformf("opacity_level", 1)

    gl_mapper.AddObserver(window.vtk.vtkCommand.UpdateShaderEvent,
                          vtk_shader_callback)


def horizon(tractograms, images, cluster, cluster_thr, random_colors,
            length_lt, length_gt, clusters_lt, clusters_gt):

    world_coords = True
    interactive = True

    prng = np.random.RandomState(27)  # 1838
    global centroid_actors, cluster_actors, visible_centroids, visible_clusters
    global cluster_access
    centroid_actors = {}
    cluster_actors = {}
    global tractogram_clusters, text_block, show_m
    tractogram_clusters = {}

    global opacity_level
    opacity_level = 0

    ren = window.Renderer()
    for (t, streamlines) in enumerate(tractograms):
        if random_colors:
            colors = prng.random_sample(3)
        else:
            colors = None

        if cluster:

            text_block = ui.TextBlock2D()
            text_block.message = \
                ' >> left click: select centroid, i: invert selection, h: hide unselected centroids\n >> e: show selected clusters, a: select all centroids and remove highlight s: save in file'

            ren.add(text_block.get_actor())
            print(' Clustering threshold {} \n'.format(cluster_thr))
            clusters = qbx_and_merge(streamlines,
                                     [40, 30, 25, 20, cluster_thr])
            tractogram_clusters[t] = clusters
            centroids = clusters.centroids
            print(' Number of centroids is {}'.format(len(centroids)))
            sizes = np.array([len(c) for c in clusters])
            linewidths = np.interp(sizes,
                                   [sizes.min(), sizes.max()], [0.1, 2.])
            centroid_lengths = np.array([length(c) for c in centroids])

            print(' Minimum number of streamlines in cluster {}'
                  .format(sizes.min()))

            print(' Maximum number of streamlines in cluster {}'
                  .format(sizes.max()))

            print(' Construct cluster actors')
            for (i, c) in enumerate(centroids):

                centroid_actor = actor.streamtube([c], colors,
                                                  linewidth=linewidths[i],
                                                  lod=False)
                ren.add(centroid_actor)

                cluster_actor = actor.line(clusters[i],
                                           lod=False)
                cluster_actor.GetProperty().SetRenderLinesAsTubes(1)
                cluster_actor.GetProperty().SetLineWidth(6)
                cluster_actor.GetProperty().SetOpacity(1)
                cluster_actor.VisibilityOff()

                ren.add(cluster_actor)

                # Every centroid actor is paired to a cluster actor
                centroid_actors[centroid_actor] = {
                    'cluster_actor': cluster_actor,
                    'cluster': i, 'tractogram': t,
                    'size': sizes[i], 'length': centroid_lengths[i],
                    'selected': 0, 'expanded':0}

                cluster_actors[cluster_actor] = {
                    'centroid_actor': centroid_actor,
                    'cluster': i, 'tractogram': t,
                    'size': sizes[i], 'length': centroid_lengths[i],
                    'selected': 0}
                apply_shader(cluster_actor)
                apply_shader(centroid_actor)

        else:
            streamline_actor = actor.line(streamlines, colors=colors)
            streamline_actor.GetProperty().SetEdgeVisibility(1)
            streamline_actor.GetProperty().SetRenderLinesAsTubes(1)
            streamline_actor.GetProperty().SetLineWidth(6)
            streamline_actor.GetProperty().SetOpacity(1)
            ren.add(streamline_actor)

    show_m = window.ShowManager(ren, size=(1200, 900), order_transparent=True)
    show_m.initialize()

    if cluster:

        lengths = np.array(
            [cluster_actors[c]['length'] for c in cluster_actors])
        sizes = np.array([cluster_actors[c]['size'] for c in cluster_actors])

        global panel2, slider_length, slider_size
        panel2 = ui.Panel2D(center=(1030, 320),
                            size=(300, 200),
                            color=(1, 1, 1),
                            opacity=0.1,
                            align="right")

        slider_label_length = build_label(text="Length")
        slider_length = ui.LineSlider2D(min_value=lengths.min(),
                                        max_value=lengths.max(),
                                        initial_value=lengths.min(),
                                        text_template="{value:.0f}",
                                        length=140)

        slider_label_size = build_label(text="Size")
        slider_size = ui.LineSlider2D(min_value=sizes.min(),
                                      max_value=sizes.max(),
                                      initial_value=sizes.min(),
                                      text_template="{value:.0f}",
                                      length=140)

        global length_min, size_min
        size_min = sizes.min()
        length_min = lengths.min()

        def hide_clusters_length(i_ren, obj, slider):
            global show_m, length_min, size_min, expand_all
            length_min = np.round(slider.value)

            for k in cluster_actors:
                if cluster_actors[k]['length'] < length_min or cluster_actors[k]['size'] < size_min :
                    cluster_actors[k]['centroid_actor'].SetVisibility(0)
                    if k.GetVisibility() == 1:
                        k.SetVisibility(0)
                else:
                    cluster_actors[k]['centroid_actor'].SetVisibility(1)
            show_m.render()

        def hide_clusters_size(i_ren, obj, slider):
            global show_m, length_min, size_min
            size_min = np.round(slider.value)

            for k in cluster_actors:
                if cluster_actors[k]['length'] < length_min or cluster_actors[k]['size'] < size_min :
                    cluster_actors[k]['centroid_actor'].SetVisibility(0)
                    if k.GetVisibility() == 1:
                        k.SetVisibility(0)
                else:
                    cluster_actors[k]['centroid_actor'].SetVisibility(1)
            show_m.render()

        slider_length.add_callback(slider_length.slider_disk,
                                   "MouseMoveEvent",
                                   hide_clusters_length)
        slider_length.add_callback(slider_length.slider_line,
                                   "LeftButtonPressEvent",
                                   hide_clusters_length)

        panel2.add_element(slider_label_length, 'relative', (0.1, 0.333))
        panel2.add_element(slider_length, 'relative', (0.65, 0.333))

        slider_size.add_callback(slider_size.slider_disk,
                                 "MouseMoveEvent",
                                 hide_clusters_size)
        slider_size.add_callback(slider_size.slider_line,
                                 "LeftButtonPressEvent",
                                 hide_clusters_size)

        panel2.add_element(slider_label_size, 'relative', (0.1, 0.6666))
        panel2.add_element(slider_size, 'relative', (0.65, 0.6666))

        ren.add(panel2)

    if len(images) > 0:
        # !!Only first image loading supported')
        data, affine = images[0]
        panel = slicer_panel(ren, data, affine, world_coords)
    else:
        data = None
        affine = None

    global size
    size = ren.GetSize()

    def win_callback(obj, event):
        global size
        if size != obj.GetSize():
            size_old = size
            size = obj.GetSize()
            size_change = [size[0] - size_old[0], 0]
            if data is not None:
                panel.re_align(size_change)
            if cluster:
                panel2.re_align(size_change)

    show_m.initialize()

    global picked_actors
    picked_actors = {}

    def left_click_centroid_callback(obj, event):

        centroid_actors[obj]['selected'] = not centroid_actors[obj]['selected']
        cluster_actors[centroid_actors[obj]['cluster_actor']]['selected'] = \
            centroid_actors[obj]['selected']
        show_m.render()

    def left_click_cluster_callback(obj, event):

        if cluster_actors[obj]['selected']:
            cluster_actors[obj]['centroid_actor'].VisibilityOn()
            centroid_actors[cluster_actors[obj]['centroid_actor']]['selected'] = 0
            obj.VisibilityOff()
            centroid_actors[cluster_actors[obj]['centroid_actor']]['expanded'] = 0

        show_m.render()


    for cl in cluster_actors:
        cl.AddObserver('LeftButtonPressEvent', left_click_cluster_callback,
                       1.0)
        cluster_actors[cl]['centroid_actor'].AddObserver(
            'LeftButtonPressEvent', left_click_centroid_callback, 1.0)


    global hide_centroids
    hide_centroids = True
    global select_all
    select_all = False

    def key_press(obj, event):
        global opacity_level, slider_length, slider_size, length_min, size_min
        global select_all, tractogram_clusters, hide_centroids
        key = obj.GetKeySym()
        if cluster:

            # hide on/off unselected centroids
            if key == 'h' or key == 'H':
                if hide_centroids:
                    for ca in centroid_actors:
                        if (centroid_actors[ca]['length'] >= length_min or
                                centroid_actors[ca]['size'] >= size_min):
                            if centroid_actors[ca]['selected'] == 0:
                                ca.VisibilityOff()
                else:
                    for ca in centroid_actors:
                        if (centroid_actors[ca]['length'] >= length_min and
                                centroid_actors[ca]['size'] >= size_min):
                            if centroid_actors[ca]['selected'] == 0:
                                ca.VisibilityOn()
                hide_centroids = not hide_centroids
                show_m.render()

            # invert selection
            if key == 'i' or key == 'I':

                for ca in centroid_actors:
                    if (centroid_actors[ca]['length'] >= length_min and
                            centroid_actors[ca]['size'] >= size_min):
                        centroid_actors[ca]['selected'] = not centroid_actors[ca]['selected']
                        cluster_actors[centroid_actors[ca]['cluster_actor']]['selected'] = \
                            centroid_actors[ca]['selected']
                show_m.render()

            # save current result
            if key == 's' or key == 'S':
                saving_streamlines = Streamlines()
                for bundle in cluster_actors.keys():
                    if bundle.GetVisibility():
                        t = cluster_actors[bundle]['tractogram']
                        c = cluster_actors[bundle]['cluster']
                        indices = tractogram_clusters[t][c]
                        saving_streamlines.extend(Streamlines(indices))
                print('Saving result in tmp.trk')
                save_trk('tmp.trk', saving_streamlines, np.eye(4))

            if key == 'a' or key == 'A':

                if select_all is False:
                    for ca in centroid_actors:
                        if (centroid_actors[ca]['length'] >= length_min and
                                centroid_actors[ca]['size'] >= size_min):
                            centroid_actors[ca]['selected'] = 1
                            cluster_actors[centroid_actors[ca]['cluster_actor']]['selected'] = \
                                centroid_actors[ca]['selected']
                    show_m.render()
                    select_all = True
                else:
                    for ca in centroid_actors:
                        if (centroid_actors[ca]['length'] >= length_min and
                                    centroid_actors[ca]['size'] >= size_min):
                            centroid_actors[ca]['selected'] = 0
                            cluster_actors[centroid_actors[ca]['cluster_actor']]['selected'] = \
                            centroid_actors[ca]['selected']
                    show_m.render()
                    select_all = False

            if key == 'e' or key == 'E':

                for c in centroid_actors:
                    if centroid_actors[c]['selected']:
                        if not centroid_actors[c]['expanded']:
                            if (centroid_actors[c]['length'] >= length_min and
                                    centroid_actors[c]['size'] >= size_min):
                                centroid_actors[c]['cluster_actor'].VisibilityOn()
                                c.VisibilityOff()
                                centroid_actors[c]['expanded'] = 1
                        else:
                            if (centroid_actors[c]['length'] >= length_min and
                                    centroid_actors[c]['size'] >= size_min):
                                centroid_actors[c]['cluster_actor'].VisibilityOff()
                                c.VisibilityOn()
                                centroid_actors[c]['expanded'] = 0

                show_m.render()


    ren.zoom(1.5)
    ren.reset_clipping_range()

    if interactive:

        show_m.add_window_callback(win_callback)
        show_m.iren.AddObserver('KeyPressEvent', key_press)
        show_m.render()
        show_m.start()

    else:

        window.record(ren, out_path='bundles_and_3_slices.png',
                      size=(1200, 900),
                      reset_camera=False)


class HorizonFlow(Workflow):

    @classmethod
    def get_short_name(cls):
        return 'horizon'

    def run(self, input_files, cluster=False, cluster_thr=15.,
            random_colors=False,
            length_lt=1000, length_gt=0,
            clusters_lt=10**8, clusters_gt=0):
        """ Advanced visualization utility

        Parameters
        ----------
        input_files : variable string
        cluster : bool
        cluster_thr : float
        random_colors : bool
        length_lt : float
        length_gt : float
        clusters_lt : int
        clusters_gt : int
        """
        verbose = True
        tractograms = []
        images = []

        for f in input_files:

            if verbose:
                print('Loading file ...')
                print(f)
                print('\n')

            if f.endswith('.trk'):

                streamlines, hdr = load_trk(f)
                tractograms.append(streamlines)

            if f.endswith('.nii.gz') or f.endswith('.nii'):

                data, affine = load_nifti(f)
                images.append((data, affine))
                if verbose:
                    print(affine)

        horizon(tractograms, images, cluster, cluster_thr,
                random_colors, length_lt, length_gt, clusters_lt,
                clusters_gt)
