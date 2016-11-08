from dipy.viz import actor, window

import numpy as np

# from dipy.exp_comparisons_experts import labelize_expert_bundles

from dipy.align.streamlinear import remove_clusters_by_size

from dipy.segment.clustering import QuickBundles

from dipy.align.streamlinear import whole_brain_slr,progressive_slr

from dipy.tracking.streamline import set_number_of_points

from dipy.tracking.streamline import set_number_of_points

from dipy.tracking.distances import bundles_distances_mdf

from dipy.segment.clustering import QuickBundles

from dipy.segment.metric import MinimumAverageDirectFlipMetric, AveragePointwiseEuclideanMetric


def show_streamlines(streamlines,streamlines2, color_array,color_array2,translate=False):

    renderer = window.Renderer()
    renderer.background((1, 1, 1))
    st_actor = actor.line(streamlines, colors=color_array, opacity=1, linewidth=0.5, spline_subdiv=None, lod=True, lod_points=10 ** 4, lod_points_size=3, lookup_colormap=None)
    renderer.add(st_actor)
    st_actor2 = actor.line(streamlines2, colors=color_array2, opacity=1, linewidth=0.5, spline_subdiv=None, lod=True, lod_points=10 ** 4, lod_points_size=3, lookup_colormap=None)
    renderer.add(st_actor2)

    if translate:
        st_actor.SetPosition(200,0,0)
    else:
        st_actor.SetPosition(0,0,0)
    st_actor2.SetPosition(0,0,0)
    window.show(renderer, title='DIPY', size=(300, 300), png_magnify=1, reset_camera=True, order_transparent=False)


data_dir = '/home/elef/Data/fancy_data_5_subj/'

#dname_atlas = data_dir + '2013_02_08_Gabriel_Girard/TRK_Files/'
dname_atlas = '/home/elef/Data/fancy_data_5_subj/2013_02_08_Gabriel_Girard/TRK_files/'


dname_full_atlas_streamlines = data_dir + '2013_02_08_Gabriel_Girard/streamlines_500K.trk'

atlas_dix = {}

atlas_dix['af.left'] = {'filename': dname_atlas + 'bundles_af.left.trk'}
atlas_dix['cc_1'] = {'filename': dname_atlas + 'bundles_cc_1.trk'}
atlas_dix['cc_2'] = {'filename': dname_atlas + 'bundles_cc_2.trk'}
atlas_dix['cc_3'] = {'filename': dname_atlas + 'bundles_cc_3.trk'}
atlas_dix['cc_4'] = {'filename': dname_atlas + 'bundles_cc_4.trk'}
atlas_dix['cc_5'] = {'filename' : dname_atlas + 'bundles_cc_5.trk'}
atlas_dix['cc_6'] = {'filename' : dname_atlas + 'bundles_cc_6.trk'}
atlas_dix['cc_7'] = {'filename' : dname_atlas + 'bundles_cc_7.trk'}
atlas_dix['af.right'] = {'filename' : dname_atlas + 'bundles_af.right.trk'}
atlas_dix['cg.left'] = {'filename' : dname_atlas + 'bundles_cg.left.trk'}
atlas_dix['cg.right'] = {'filename' : dname_atlas + 'bundles_cg.right.trk'}
atlas_dix['cst.left'] = {'filename' : dname_atlas + 'bundles_cst.left.trk'}
atlas_dix['cst.right'] = {'filename' : dname_atlas + 'bundles_cst.right.trk'}
atlas_dix['ifof.left'] = {'filename' : dname_atlas + 'bundles_ifof.left.trk'}
atlas_dix['ifof.right'] = {'filename' : dname_atlas + 'bundles_ifof.right.trk'}
atlas_dix['ilf.left'] = {'filename' : dname_atlas + 'bundles_ilf.left.trk'}
atlas_dix['ilf.right'] = {'filename' : dname_atlas + 'bundles_ilf.right.trk'}
atlas_dix['mdlf.left'] = {'filename' : dname_atlas + 'bundles_mdlf.left.trk'}
atlas_dix['mdlf.right'] = {'filename' : dname_atlas + 'bundles_mdlf.right.trk'}
atlas_dix['slf1.left'] = {'filename' : dname_atlas + 'bundles_slf1.left.trk'}
atlas_dix['slf1.right'] = {'filename' : dname_atlas + 'bundles_slf1.right.trk'}
atlas_dix['slf2.left'] = {'filename' : dname_atlas + 'bundles_slf2.left.trk'}
atlas_dix['slf2.right'] = {'filename' : dname_atlas + 'bundles_slf2.right.trk'}
atlas_dix['slf_3.left'] = {'filename' : dname_atlas + 'bundles_slf_3.left.trk'}
atlas_dix['slf_3.right'] = {'filename' : dname_atlas + 'bundles_slf_3.right.trk'}
atlas_dix['uf.left'] = {'filename' : dname_atlas + 'bundles_uf.left.trk'}
atlas_dix['uf.right'] = {'filename' : dname_atlas + 'bundles_uf.right.trk'}

streamlines_file = data_dir + '2013_02_08_Gabriel_Girard/streamlines_500K.trk'

from dipy.io.trackvis import load_trk

from nibabel.streamlines import ArraySequence

full_atlas = ArraySequence()
centroids_atlas = []
centroids_atlas2 = []
color_array = []

#metric = MinimumAverageDirectFlipMetric()
metric = AveragePointwiseEuclideanMetric()

full_atlas2 = []
keys = []
centroids_num = []
total_centroid_num = 0


for key in atlas_dix:
    filename = atlas_dix[key]['filename']

    streamlines, header = load_trk(filename)
    atlas_dix[key]['streamlines'] = streamlines
    discrete_streamlines = set_number_of_points(streamlines, 20)
    clusters = QuickBundles(threshold=15.,
                            metric=metric).cluster(discrete_streamlines)
    atlas_dix[key]['centroids'] = clusters[0].centroid
    lengh_cluster = len(clusters)
    num_centroid = int(round(lengh_cluster * 1))
    if total_centroid_num == 0:
        atlas_dix[key]['low_high'] = [total_centroid_num, num_centroid]
    else:
        atlas_dix[key]['low_high'] = [total_centroid_num - 1, total_centroid_num + num_centroid]
    print(num_centroid)
    if num_centroid == 0:
        num_centroid = 1
    for i in range(num_centroid):
        centroids_atlas += [clusters[i].centroid]

    total_centroid_num = total_centroid_num + num_centroid
    # show_streamlines(streamlines)
    full_atlas.extend(streamlines)
    full_atlas2.append(discrete_streamlines)
    centroids_num.append(num_centroid)
    keys.append(key)
    # show_streamlines(full_atlas)
    # del streamlines
color_array = np.random.rand(len(centroids_atlas), 3) * 10

streamlines_target, header_target = load_trk(streamlines_file)

print('Large streamlines loaded')

def slr(streamlines_target,centroids_atlas2):
    greater_than = 50
    less_than = 250
    qb_thr = 15
    nb_pts = 20
    rm_small_clusters=50
    progressive=True
    #x0='affine'

    discrete_target = set_number_of_points(streamlines_target, nb_points=20)

    def check_range(streamline, gt=greater_than, lt=less_than):

        if (len(streamline) > gt) & (len(streamline) < lt):
            return True
        else:
            return False

    streamlines1 = [s for s in streamlines_target if check_range(s)]

    rstreamlines1 = streamlines1

    rstreamlines1 = set_number_of_points(rstreamlines1, nb_pts)

    #print('Resample finished', sep=' ', end='n', file=sys.stdout, flush=False)

    qb1 = QuickBundles(threshold=qb_thr, metric=metric)
    rstreamlines1 = [s.astype('f4') for s in rstreamlines1]
    cluster_map1 = qb1.cluster(rstreamlines1)
    # clusters1 = remove_clusters_by_size(cluster_map1, rm_small_clusters)
    qb_centroids = [cluster.centroid for cluster in cluster_map1]

    #1/0

    bounds = [(-45, 45), (-45, 45), (-45, 45),
              (-30, 30), (-30, 30), (-30, 30),
              (0.6, 1.4), (0.6, 1.4), (0.6, 1.4),
              (-10, 10), (-10, 10), (-10, 10)]
    slm = progressive_slr(centroids_atlas2, qb_centroids,
                          x0='rigid', metric=None,
                          bounds=bounds)

#show_streamlines(qb_centroids1, centroids_atlas,translate = True)
#streamline_final = ArraySequence()
#color_array_target = []
    return slm.transform(discrete_target)


def getting_final_streamline(streamlines_target, centroids_atlas2, color_array2, threshold):
#    discrete_target = slr(streamlines_target,centroids_atlas2)
    discrete_target = set_number_of_points(streamlines_target, nb_points=20)

    distance_matrix = bundles_distances_mdf(centroids_atlas2, discrete_target)

    index_whole = np.argmin(distance_matrix, axis=0)

    stream_line_min = np.amin(distance_matrix, axis=0)

    index_threshold = np.where(stream_line_min < threshold)

    streamline_final = streamlines_target[index_threshold[0]]

    color_array_target = color_array2[index_whole[index_threshold[0]], :]

#    final_index_whole = index_whole[index_threshold[0]]

    return streamline_final, color_array_target, index_whole, index_threshold

def computing_accuracy(streamline_atlas, streamline_compute):
    distance_matrix = bundles_distances_mdf(streamline_atlas, streamline_compute)

    from ipdb import set_trace
    set_trace()

    stream_line_min = np.amin(distance_matrix, axis=0)

    index_threshold_accuracy = np.where(stream_line_min < 0.01)

    set_trace()
    print(len(index_threshold_accuracy[0]))

    return (len(index_threshold_accuracy[0])/np.float(len(streamline_atlas) + len(streamline_compute) - len(index_threshold_accuracy[0])),
            len(index_threshold_accuracy[0])/np.float(len(streamline_atlas)))


def finding_corresponding_index(key, index_whole, index_threshold, streamline_target):

    low = atlas_dix[key]['low_high'][0]
    high = atlas_dix[key]['low_high'][1]

    if low == 0:
        s = np.where(index_whole < high)
    else:
        s = np.where(np.logical_and(index_whole > low, index_whole < high))

    indices = np.where(np.in1d(s, index_threshold))[0]

    ss = s[0][indices]

    streamline_part = streamline_target[ss]

    return ss, indices, streamline_part


def visualization(key, index_whole, index_threshold, streamline_target, atlas_part, translate=False):
    low = atlas_dix[key]['low_high'][0]

    high = atlas_dix[key]['low_high'][1]
    if low == 0:
        s = np.where(index_whole < high)
    else:
        s = np.where(np.logical_and(index_whole > low, index_whole < high))

    indices = np.where(np.in1d(s, index_threshold))[0]

    ss = s[0][indices]

    streamline_part = streamline_target[ss]

    show_streamlines(atlas_part, streamline_part,
                     [0, 0.5, 0], [0, 0, 0.4], translate)


def computing_range_accuracy(key, index_whole, index_threshold, streamline_target, atlas_part):

    ss, indices, streamline_part = finding_corresponding_index(key, index_whole, index_threshold, streamline_target)

    streamline_np = np.array(streamline_part, dtype=np.object)

    np.save(key, streamline_np)
    discrete_streamline_part = set_number_of_points(streamline_part, 20)

    jaccard, accuracy = computing_accuracy(atlas_part, discrete_streamline_part)

    print(key)
    print(jaccard)
    print(accuracy)

if __name__ == '__main__':



    res = getting_final_streamline(streamlines_target, centroids_atlas,
                                   color_array, 15)
    streamline_final, color_array_target, index_whole, index_threshold = res

    visualization(keys[1], index_whole, index_threshold, streamlines_target, full_atlas2[1], False)

#    discrete_streamlines_target = set_number_of_points(streamlines_target, 20)
    computing_range_accuracy(keys[1], index_whole, index_threshold,
                             streamlines_target,
                             atlas_part=full_atlas2[1])
