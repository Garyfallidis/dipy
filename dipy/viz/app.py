import numpy as np
from dipy.segment.clustering import qbx_and_merge
from dipy.viz import actor, window, ui
from dipy.viz.window import vtk
from dipy.viz.panel import slicer_panel, build_label
from dipy.tracking.streamline import transform_streamlines, length, Streamlines
from dipy.io.streamline import load_trk, save_trk
from dipy.io.image import load_nifti, save_nifti


def check_range(streamline, lt, gt):
    length_s = length(streamline)
    if (length_s < lt) & (length_s > gt):
        return True
    else:
        return False


def apply_shader_new(hz, actor):
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
                                    hz.cea[actor]['selected'])
            except KeyError:
                pass
            try:
                program.SetUniformf("selected",
                                    hz.cla[actor]['selected'])
            except KeyError:
                pass
            program.SetUniformf("opacity_level", 1)

    gl_mapper.AddObserver(window.vtk.vtkCommand.UpdateShaderEvent,
                          vtk_shader_callback)


HELP_MESSAGE = """
>> left click: select centroid
>> e: expand centroids
>> r: collapse open clusters
>> h: hide unselected centroids
>> i: invert selection
>> a: select all centroids
>> s: save in file
"""


class Horizon(object):

    def __init__(self, tractograms, images, cluster, cluster_thr,
                 random_colors, length_lt, length_gt, clusters_lt, clusters_gt,
                 world_coords=True, interactive=True):

        self.cluster = cluster
        self.cluster_thr = cluster_thr
        self.random_colors = random_colors
        self.length_lt = length_lt
        self.length_gt = length_gt
        self.clusters_lt = clusters_lt
        self.clusters_gt = clusters_gt
        self.world_coords = world_coords
        self.interactive = interactive
        self.prng = np.random.RandomState(27)
        self.tractograms = tractograms
        self.images = images
        self.cea = {}  # holds centroid actors
        self.cla = {}  # holds cluster actors
        self.tractogram_clusters = {}

    def build_renderer(self):

        ren = window.Renderer()
        for (t, streamlines) in enumerate(self.tractograms):
            if self.random_colors:
                colors = self.prng.random_sample(3)
            else:
                colors = None

            if self.cluster:

                # _cluster(streamlines, cluster_thr, )
                print(' Clustering threshold {} \n'.format(self.cluster_thr))
                clusters = qbx_and_merge(streamlines,
                                         [40, 30, 25, 20, self.cluster_thr])
                self.tractogram_clusters[t] = clusters
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
                    self.cea[centroid_actor] = {
                        'cluster_actor': cluster_actor,
                        'cluster': i, 'tractogram': t,
                        'size': sizes[i], 'length': centroid_lengths[i],
                        'selected': 0, 'expanded': 0}

                    self.cla[cluster_actor] = {
                        'centroid_actor': centroid_actor,
                        'cluster': i, 'tractogram': t,
                        'size': sizes[i], 'length': centroid_lengths[i],
                        'selected': 0}
                    apply_shader(self, cluster_actor)
                    apply_shader(self, centroid_actor)

            else:
                streamline_actor = actor.line(streamlines, colors=colors)
                streamline_actor.GetProperty().SetEdgeVisibility(1)
                streamline_actor.GetProperty().SetRenderLinesAsTubes(1)
                streamline_actor.GetProperty().SetLineWidth(6)
                streamline_actor.GetProperty().SetOpacity(1)
                ren.add(streamline_actor)
        return ren

    def build_show(self, ren):

        show_m = window.ShowManager(ren, size=(1200, 900),
                                    order_transparent=True,
                                    reset_camera=False)
        show_m.initialize()

        if self.cluster:

            lengths = np.array(
                [self.cla[c]['length'] for c in self.cla])
            szs = [self.cla[c]['size'] for c in self.cla]
            sizes = np.array(szs)

            global panel2, slider_length, slider_size
            panel2 = ui.Panel2D(size=(300, 200),
                                position=(850, 320),
                                color=(1, 1, 1),
                                opacity=0.1,
                                align="right")

            slider_label_length = build_label(text="Length")
            slider_length = ui.LineSlider2D(
                    min_value=lengths.min(),
                    max_value=np.percentile(lengths, 98),
                    initial_value=np.percentile(lengths, 25),
                    text_template="{value:.0f}",
                    length=140)

            slider_label_size = build_label(text="Size")
            slider_size = ui.LineSlider2D(min_value=sizes.min(),
                                          max_value=np.percentile(sizes, 98),
                                          initial_value=np.percentile(sizes, 50),
                                          text_template="{value:.0f}",
                                          length=140)

            global length_min, size_min
            size_min = sizes.min()
            length_min = lengths.min()

            def hide_clusters_length(slider):
                global show_m, length_min, size_min, expand_all
                length_min = np.round(slider.value)

                for k in self.cla:
                    if (self.cla[k]['length'] < length_min or
                            self.cla[k]['size'] < size_min):
                        self.cla[k]['centroid_actor'].SetVisibility(0)
                        if k.GetVisibility() == 1:
                            k.SetVisibility(0)
                    else:
                        self.cla[k]['centroid_actor'].SetVisibility(1)
                show_m.render()

            def hide_clusters_size(slider):
                global show_m, length_min, size_min
                size_min = np.round(slider.value)

                for k in self.cla:
                    if (self.cla[k]['length'] < length_min or
                            self.cla[k]['size'] < size_min):
                        self.cla[k]['centroid_actor'].SetVisibility(0)
                        if k.GetVisibility() == 1:
                            k.SetVisibility(0)
                    else:
                        self.cla[k]['centroid_actor'].SetVisibility(1)
                show_m.render()

            slider_length.on_change = hide_clusters_length

            panel2.add_element(slider_label_length, coords=(0.1, 0.333))
            panel2.add_element(slider_length, coords=(0.4, 0.333))

            slider_size.on_change = hide_clusters_size

            panel2.add_element(slider_label_size, coords=(0.1, 0.6666))
            panel2.add_element(slider_size, coords=(0.4, 0.6666))

            ren.add(panel2)

            text_block = build_label(HELP_MESSAGE, 16)  # ui.TextBlock2D()
            text_block.message = HELP_MESSAGE

            help_panel = ui.Panel2D(size=(300, 200),
                                    color=(1, 1, 1),
                                    opacity=0.1,
                                    align="left")

            help_panel.add_element(text_block, coords=(0.05, 0.1))
            ren.add(help_panel)

        if len(self.images) > 0:
            # !!Only first image loading supported for now')
            data, affine = self.images[0]
            panel = slicer_panel(ren, data, affine, self.world_coords)
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
                if self.cluster:
                    panel2.re_align(size_change)
                    help_panel.re_align(size_change)

        show_m.initialize()

        global picked_actors
        picked_actors = {}

        def left_click_centroid_callback(obj, event):

            self.cea[obj]['selected'] = not self.cea[obj]['selected']
            self.cla[self.cea[obj]['cluster_actor']]['selected'] = \
                self.cea[obj]['selected']
            show_m.render()

        def left_click_cluster_callback(obj, event):

            if self.cla[obj]['selected']:
                self.cla[obj]['centroid_actor'].VisibilityOn()
                ca = self.cla[obj]['centroid_actor']
                self.cea[ca]['selected'] = 0
                obj.VisibilityOff()
                self.cea[ca]['expanded'] = 0

            show_m.render()

        for cl in self.cla:
            cl.AddObserver('LeftButtonPressEvent', left_click_cluster_callback,
                           1.0)
            self.cla[cl]['centroid_actor'].AddObserver(
                'LeftButtonPressEvent', left_click_centroid_callback, 1.0)

        global hide_centroids
        hide_centroids = True
        global select_all
        select_all = False

        def key_press(obj, event):
            global opacity_level, slider_length, slider_size, length_min, size_min
            global select_all, tractogram_clusters, hide_centroids
            key = obj.GetKeySym()
            if self.cluster:

                # hide on/off unselected centroids
                if key == 'h' or key == 'H':
                    if hide_centroids:
                        for ca in self.cea:
                            if (self.cea[ca]['length'] >= length_min or
                                    self.cea[ca]['size'] >= size_min):
                                if self.cea[ca]['selected'] == 0:
                                    ca.VisibilityOff()
                    else:
                        for ca in self.cea:
                            if (self.cea[ca]['length'] >= length_min and
                                    self.cea[ca]['size'] >= size_min):
                                if self.cea[ca]['selected'] == 0:
                                    ca.VisibilityOn()
                    hide_centroids = not hide_centroids
                    show_m.render()

                # invert selection
                if key == 'i' or key == 'I':

                    for ca in self.cea:
                        if (self.cea[ca]['length'] >= length_min and
                                self.cea[ca]['size'] >= size_min):
                            self.cea[ca]['selected'] = \
                                not self.cea[ca]['selected']
                            cas = self.cea[ca]['cluster_actor']
                            self.cla[cas]['selected'] = \
                                self.cea[ca]['selected']
                    show_m.render()

                # save current result
                if key == 's' or key == 'S':
                    saving_streamlines = Streamlines()
                    for bundle in self.cla.keys():
                        if bundle.GetVisibility():
                            t = self.cla[bundle]['tractogram']
                            c = self.cla[bundle]['cluster']
                            indices = self.tractogram_clusters[t][c]
                            saving_streamlines.extend(Streamlines(indices))
                    print('Saving result in tmp.trk')
                    save_trk('tmp.trk', saving_streamlines, np.eye(4))

                if key == 'y' or key == 'Y':
                    active_streamlines = Streamlines()
                    for bundle in self.cla.keys():
                        if bundle.GetVisibility():
                            t = self.cla[bundle]['tractogram']
                            c = self.cla[bundle]['cluster']
                            indices = self.tractogram_clusters[t][c]
                            active_streamlines.extend(Streamlines(indices))

                    self.tractograms = [active_streamlines]
                    ren = self.build_renderer()
                    self.build_show(ren)

                if key == 'a' or key == 'A':

                    if select_all is False:
                        for ca in self.cea:
                            if (self.cea[ca]['length'] >= length_min and
                                    self.cea[ca]['size'] >= size_min):
                                self.cea[ca]['selected'] = 1
                                cas = self.cea[ca]['cluster_actor']
                                self.cla[cas]['selected'] = \
                                    self.cea[ca]['selected']
                        show_m.render()
                        select_all = True
                    else:
                        for ca in self.cea:
                            if (self.cea[ca]['length'] >= length_min and
                                    self.cea[ca]['size'] >= size_min):
                                self.cea[ca]['selected'] = 0
                                cas = self.cea[ca]['cluster_actor']
                                self.cla[cas]['selected'] = \
                                    self.cea[ca]['selected']
                        show_m.render()
                        select_all = False

                if key == 'e' or key == 'E':

                    for c in self.cea:
                        if self.cea[c]['selected']:
                            if not self.cea[c]['expanded']:
                                if (self.cea[c]['length'] >= length_min and
                                        self.cea[c]['size'] >= size_min):
                                    self.cea[c]['cluster_actor']. \
                                        VisibilityOn()
                                    c.VisibilityOff()
                                    self.cea[c]['expanded'] = 1

                    show_m.render()

                if key == 'r' or key == 'R':

                    for c in self.cea:

                        if (self.cea[c]['length'] >= length_min and
                                self.cea[c]['size'] >= size_min):
                            self.cea[c]['cluster_actor'].VisibilityOff()
                            c.VisibilityOn()
                            self.cea[c]['expanded'] = 0

                show_m.render()

        ren.reset_camera()
        ren.zoom(1.5)
        ren.reset_clipping_range()

        if self.interactive:

            show_m.add_window_callback(win_callback)
            show_m.iren.AddObserver('KeyPressEvent', key_press)
            show_m.render()
            show_m.start()

        else:

            window.record(ren, out_path='tmp.png',
                          size=(1200, 900),
                          reset_camera=False)


def horizon(tractograms, images, cluster, cluster_thr, random_colors,
            length_lt, length_gt, clusters_lt, clusters_gt,
            world_coords=True, interactive=True):

        hz = Horizon(tractograms, images, cluster, cluster_thr, random_colors,
                     length_lt, length_gt, clusters_lt, clusters_gt,
                     world_coords, interactive)

        renderer = hz.build_renderer()

        hz.build_show(renderer)




