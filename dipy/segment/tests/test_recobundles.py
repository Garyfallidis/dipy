import numpy as np
import numpy.testing as npt
import nibabel.trackvis as tv
from dipy.tracking.streamline import transform_streamlines
from copy import deepcopy
from itertools import chain
from dipy.segment.bundles import RecoBundles


def show_bundles(static, moving, linewidth=1., tubes=False,
                 opacity=1., fname=None):

    from dipy.viz import fvtk
    ren = fvtk.ren()
    ren.clear()
    ren.SetBackground(1, 1, 1.)

    if tubes:
        static_actor = fvtk.streamtube(static, fvtk.colors.red,
                                       linewidth=linewidth, opacity=opacity)
        moving_actor = fvtk.streamtube(moving, fvtk.colors.green,
                                       linewidth=linewidth, opacity=opacity)

    else:
        static_actor = fvtk.line(static, fvtk.colors.red,
                                 linewidth=linewidth, opacity=opacity)
        moving_actor = fvtk.line(moving, fvtk.colors.green,
                                 linewidth=linewidth, opacity=opacity)

    fvtk.add(ren, static_actor)
    fvtk.add(ren, moving_actor)

    fvtk.add(ren, fvtk.axes(scale=(20, 20, 20)))

    fvtk.show(ren, size=(900, 900))
    if fname is not None:
        fvtk.record(ren, size=(900, 900), out_path=fname)


def test_recognition():

    disp = True
    dname = '/home/eleftherios/Data/ISMRM_2015_challenge_bundles_RAS/'

    bundle_trk = ['CA', 'CC', 'Cingulum_left',
                  'Cingulum_right', 'CP',
                  'CST_left', 'CST_right',
                  'Fornix', 'FPT_left', 'FPT_right',
                  'ICP_left', 'ICP_right',
                  'IOFF_left', 'IOFF_right', 'MCP',
                  'OR_left', 'OR_right',
                  'POPT_left', 'POPT_right',
                  'SCP_left', 'SCP_right',
                  'SLF_left', 'SLF_right',
                  'UF_left', 'UF_right']

    fnames = [dname + bundle_name + '.trk' for bundle_name in bundle_trk]

    model_bundles_dix = {}
    model_indices_dix = {}

    cnt = 0

    for (i, fname) in enumerate(fnames):
        streams, hdr = tv.read(fname, points_space='rasmm')
        bundle = [s[0] for s in streams]
        key = bundle_trk[i].split('.trk')[0]
        model_bundles_dix[key] = bundle
        model_indices_dix[key] = cnt + np.arange(len(bundle))
        cnt = cnt + len(bundle)

    play_bundles_dix = deepcopy(model_bundles_dix)

    mat = np.eye(4)
    mat[:3, 3] = np.array([-5., 5, 0])

    # tag = 'MCP'
    # tag = 'Fornix'
    # tag = 'Cingulum_right'
    # tag = 'CST_right'
    # tag = 'CST_left'
    tag = 'POPT_left'

    play_bundles_dix[tag] = transform_streamlines(play_bundles_dix[tag], mat)

    model_bundle = model_bundles_dix[tag]

    # make sure that you put the bundles the correct order for the
    # classification tests
    streamlines = []

    for (i, f) in enumerate(fnames):
        streamlines += play_bundles_dix[bundle_trk[i]]

    # show_bundles(model_bundle, streamlines)

    rb = RecoBundles(streamlines, mdf_thr=15)
    recognized_bundle = rb.recognize(model_bundle, mdf_thr=5,
                                     reduction_thr=20,
                                     slr=True,
                                     slr_select=(400, 400),
                                     pruning_thr=5)
    # TODO check why pruning threshold segfaults when very low

    if disp:

        print('Show model centroids and all centroids of new space')
        show_bundles(rb.model_centroids, rb.centroids)

        print('Show model bundle and neighborhood')
        show_bundles(model_bundle, rb.neighb_streamlines)

        print('Show model bundle and transformed neighborhood')
        show_bundles(model_bundle, rb.transf_streamlines)

        print('Show model bundles and pruned streamlines')
        show_bundles(model_bundle, recognized_bundle)

        mat2 = np.eye(4)
        mat2[:3, 3] = np.array([60, 0, 0])

        print('Same with a shift')
        show_bundles(transform_streamlines(model_bundle, mat2),
                     recognized_bundle)

        print('Show initial labels vs model bundle')
        show_bundles(transform_streamlines(rb.labeled_streamlines, mat2),
                     model_bundle)

    print('\a')
    print('Recognized bundle has %d streamlines' % (len(recognized_bundle),))
    print('Model bundle has %d streamlines' % (len(model_bundle),))
    print('\a')

    # intersection = np.intersect1d(model_indices_dix['MCP'], rb.labels)
    difference = np.setdiff1d(rb.labels, model_indices_dix[tag])
    print('Difference %d' % (len(difference),))

    print('\a')
    print('Build the KDTree for this bundle')

    print('\a')

    rb.build_kdtree(mam_metric=None)

    dists, actual_indices, expansion_streamlines = rb.expand(300, True)

    expansion_intersection = np.intersect1d(actual_indices, rb.labels)
    print(len(expansion_intersection))
    npt.assert_equal(len(expansion_intersection), 0)

    show_bundles(recognized_bundle, expansion_streamlines, tubes=False)

    1/0

#    dists, indices = rb.kdtree.query(np.zeros(rb.kd_vectors.shape[1]),
#                                     20, p=2)
#
#    extra_streamlines = [rb.search_rstreamlines[i] for i in indices]
#    show_bundles(recognized_bundle, extra_streamlines, tubes=True)
#
#    print('New streamlines')
#    print(len(extra_streamlines))
#
#
#    dists, indices = rb.kdtree.query(np.zeros(rb.kd_vectors.shape[1]),
#                                     300, p=2)
#
#    extra_streamlines = [rb.search_rstreamlines[i] for i in indices]
#    show_bundles(recognized_bundle, extra_streamlines, tubes=True)
#
#    print('New streamlines')
#    print(len(extra_streamlines))

    # return rb

    # 1/0


if __name__ == '__main__':

    rb = test_recognition()