""" Validating QuickBundles
"""
import os.path as osp
import numpy as np
import dipy as dp
# track reading
from dipy.io.dpy import Dpy
from dipy.io.pickles import load_pickle, save_pickle
# segmenation
from dipy.segment.quickbundles import QuickBundles
# visualization
from fos import Window, Region
from fos.actor import Axes, Text3D, Line
from fos.actor.line import one_colour_per_line
from bundle_picker import TrackLabeler, track2rgb
from dipy.viz import fvtk
# metrics 
from dipy.tracking.metrics import downsample
from dipy.tracking.distances import (bundles_distances_mam,
					bundles_distances_mdf,
					most_similar_track_mam)
from dipy.tracking.distances import approx_polygon_track
from nibabel import trackvis as tv
import colorsys
from matplotlib.mlab import find

def load_data(id):
	ids=['02','03','04','05','06','08','09','10','11','12']
	filename =  'data/subj_'+ids[id]+'_lsc_QA_ref.dpy'
	dp=Dpy(filename,'r')
        print 'Loading', filename
	tracks=dp.read_tracks()
	dp.close()
	return tracks

def load_pbc_data(id=None):
    if id is None:
        path = '/home/eg309/Data/PBC/pbc2009icdm/brain1/'
        streams, hdr = tv.read(path+'brain1_scan1_fiber_track_mni.trk')
        streamlines = [s[0] for s in streams]
        return streamlines
    if not osp.exists('/tmp/'+str(id)+'.pkl'):
        path = '/home/eg309/Data/PBC/pbc2009icdm/brain1/'
        streams, hdr = tv.read(path+'brain1_scan1_fiber_track_mni.trk')
        streamlines = [s[0] for s in streams]
        labels = np.loadtxt(path+'brain1_scan1_fiber_labels.txt')
        labels = labels[:,1]
        mask_cst = labels == id
        cst_streamlines = [s for (i,s) in enumerate(streamlines) if mask_cst[i]]
        save_pickle('/tmp/'+str(id)+'.pkl', cst_streamlines)
        return cst_streamlines
        #return [approx_polygon_track(s, 0.7853) for s in cst_streamlines]
    else:
        return load_pickle('/tmp/'+str(id)+'.pkl')    


def get_tractography_sizes():
        sizes = []
        for d in range(10):
                sizes.append(len(load_data(d)))
        return sizes

def show_qb_streamlines(tracks,qb):
	# Create gui and message passing (events)
	w = Window(caption='QB validation', 
		width=1200, 
		height=800, 
		bgcolor=(0.,0.,0.2) )
	# Create a region of the world of actors
	region = Region(regionname='Main', activate_aabb=False)
	# Create actors
	tl = TrackLabeler('Bundle Picker',
			qb,qb.downsampled_tracks(),
			vol_shape=(182,218,182),tracks_alpha=1)   
	ax = Axes(name = "3 axes", scale= 10, linewidth=2.0)
	vert = np.array( [[2.0,3.0,0.0]], dtype = np.float32 )
	ptr = np.array( [[.2,.2,.2]], dtype = np.float32 )
	tex = Text3D( "Text3D", vert, "(0,0,0)", 10*2.5, 10*.5, ptr)
	#Add actor to their region
	region.add_actor(ax)
	#region.add_actor(tex)
	region.add_actor(tl)
	#Add the region to the window
	w.add_region(region)
	w.refocus_camera()
	print 'Actors loaded'
	return w,region,ax,tex

def show_tracks_colormaps(tracks, qb, alpha=1):
    w = Window(caption='QuickBundles Representation', 
            width=1200, 
            height=800, 
            bgcolor=(0.,0.,0.2))
    region = Region(regionname='Main', activate_aabb=False)

    colormap = np.ones((len(tracks), 3))
    counter = 0
    for curve in tracks:
        colormap[counter:counter+len(curve),:3] = track2rgb(curve).astype('f4')
        counter += len(curve)
    colors = one_colour_per_line(tracks, colormap)
    colors[:,3]=alpha
    la = Line('Streamlines', tracks, colors, line_width=2)
    region.add_actor(la)
    w.add_region(region)
    w.refocus_camera()
    return w, region, la

def show_tracks_fvtk(tracks, qb=None, option='only_reps'):    
    r=fvtk.ren()
    if qb is None:
        colormap = np.ones((len(tracks), 3))
        for i, curve in enumerate(tracks):
            colormap[i] = track2rgb(curve)
        fvtk.add(r, fvtk.line(tracks,colormap,linewidth=3))
    else:
        centroids=qb.virtuals()
        if option == 'only_reps':
            colormap = np.ones((len(centroids), 3))
            H=np.linspace(0,1,len(centroids)+1)
            for i, centroid in enumerate(centroids):
                col=np.array(colorsys.hsv_to_rgb(H[i],1.,1.))
                colormap[i] = col
            fvtk.add(r, fvtk.line(centroids, colormap, linewidth=3))
        if option == 'reps_and_tracks':
            colormap = np.ones((len(tracks), 3))
            H=np.linspace(0, 1, len(centroids)+1)
            for i, centroid in enumerate(centroids):
                col=np.array(colorsys.hsv_to_rgb(H[i], 1., 1.))
                inds=qb.label2tracksids(i)
                colormap[inds]=col
            fvtk.add(r, fvtk.line(tracks, colormap, linewidth=3))
        if option == 'thick_reps':
            H=np.linspace(0,1,len(centroids)+1)
            S=np.array(qb.clusters_sizes())
            for i, centroid in enumerate(centroids):
                col=np.array(colorsys.hsv_to_rgb(H[i],1.,1.))
                fvtk.add(r, fvtk.line([centroid], col, linewidth=np.interp(S[i],(S.min(),S.max()),(1,5))))
    fvtk.show(r)
    return r


def get_random_streamlines(tracks,N):	
	#qb = QuickBundles(tracks,dist,18)
	#N=qb.total_clusters()
	random_labels = np.random.permutation(np.arange(len(tracks)))[:N]
	random_streamlines = [tracks[i] for i in random_labels]
	return random_streamlines
		
def count_close_tracks(sla, slb, dist_thr=20):
        cnt_a_close = np.zeros(len(slb))
        for ta in sla:
            dta = bundles_distances_mdf([ta],slb)[0]
            #dta = bundles_distances_mam([ta],slb)[0]
            cnt_a_close += binarise(dta, dist_thr)
        return cnt_a_close

tractography_sizes = [175544, 161218, 155763, 141877, 149272, 226456, 168833, 186543, 191087, 153432]

def split_halves(id):
        tracks = load_data(id)
        N = tractography_sizes[id]
        M = N/2
	first_half = np.random.permutation(np.arange(len(tracks)))[:M]
        second_half= np.random.permutation(np.arange(len(tracks)))[M:N]
        return [tracks[n] for n in first_half], [tracks[n] for n in second_half]

'''
coverage = # neighb tracks / #tracks 
         = cntT.sum()/len(T)

overlap = (cntT>1).sum()/len(T)

missed == (cntT==0).sum()/len(T)
'''

#virtuals/#tracks
        
'''
compare_streamline_sets(sla,slb,dist=20):
	d = bundles_distances_mdf(sla,slb)
	d[d<dist]=1
	d[d>=dist]=0
	return d 
'''

def binarise(D, thr):
#Replaces elements of D which are <thr with 1 and the rest with 0
        return 1*(np.array(D)<thr)

def half_split_comparisons():

    tractography_sizes = [175544, 161218, 155763, 141877, 149272, 226456, 168833, 186543, 191087, 153432]

    # size 02 175544

    id=0

    first, second = split_halves(id)

    print len(first), len(second)

    '''
    track_subset_size = 50000

    tracks=tracks[:track_subset_size]
    print 'Streamlines loaded'
    #qb=QuickBundles(tracks,20,18)
    #print 'QuickBundles finished'
    #print 'visualize/interact with streamlines'
    #window,region,axes,labeler = show_qb_streamlines(tracks,qb)
    '''

    downsampling = 12

    first_qb = QuickBundles(first,20,downsampling)
    n_clus = first_qb.total_clusters()
    print 'QB for first half has', n_clus, 'clusters'
    second_down = [downsample(s, downsampling) for s in second]

    '''
    random_streamlines={}
    for rep in [0]:
            random_streamlines[rep] = get_random_streamlines(qb.downsampled_tracks(), N)
    '''

    # Thresholded distance matrices (subset x tracks) where subset Q = QB centroids
    # and subset R = matched random subset. Matrices have 1 if the compared
    # tracks have MDF distance < threshold a,d 0 otherwise.
    #DQ=compare_streamline_sets(qb.virtuals(),qb.downsampled_tracks(), 20)
    #DR=compare_streamline_sets(random_streamlines[0],qb.downsampled_tracks(), 20)

    # The number of subset tracks 'close' to each track
    #neighbours_Q = np.sum(DQ, axis=0)
    #neighbours_R = np.sum(DR, axis=0)

    #neighbours_Q = count_close_tracks(qb.virtuals(), qb.downsampled_tracks(), 20)
    #neighbours_R = count_close_tracks(random_streamlines[0], qb.downsampled_tracks(), 20)

    neighbours_first = count_close_tracks(first_qb.virtuals(), first_qb.downsampled_tracks(), 20)
    neighbours_second = count_close_tracks(first_qb.virtuals(), second_down, 20)

    maxclose = np.int(np.max(np.hstack((neighbours_first,neighbours_second))))

    # The numbers of tracks 0, 1, 2, ... 'close' subset tracks
    counts = [(np.int(n), len(find(neighbours_first==n)), len(find(neighbours_second==n)))
              for n in range(maxclose+1)]

    print np.array(counts)


if __name__ == '__main__' :
    #"""
    id=0
    tracks=load_data(id)
    track_subset_size = 1000
    tracks=tracks[:track_subset_size]
    #"""
    #tracks=load_pbc_data(1)
    print 'Streamlines loaded'
    qb=QuickBundles(tracks, 30, 18)
    #print 'QuickBundles finished'
    #print 'visualize/interact with streamlines'
    #window, region, axes, labeler = show_qb_streamlines(tracks, qb)
    #w, region, la = show_tracks_colormaps(tracks,qb)
    r = show_tracks_fvtk(tracks, qb, option='thick_reps')

    """
    N=qb.total_clusters()
    print 'QB finished with', N, 'clusters'

    random_streamlines={}
    for rep in [0]:
        random_streamlines[rep] = get_random_streamlines(qb.downsampled_tracks(), N)
            
    # Thresholded distance matrices (subset x tracks) where subset Q = QB centroids
    # and subset R = matched random subset. Matrices have 1 if the compared
    # tracks have MDF distance < threshold a,d 0 otherwise.
    #DQ=compare_streamline_sets(qb.virtuals(),qb.downsampled_tracks(), 20)
    #DR=compare_streamline_sets(random_streamlines[0],qb.downsampled_tracks(), 20)

    # The number of subset tracks 'close' to each track
    #neighbours_Q = np.sum(DQ, axis=0)
    #neighbours_R = np.sum(DR, axis=0)
    neighbours_Q = count_close_tracks(qb.virtuals(), qb.downsampled_tracks(), 20)
    neighbours_R = count_close_tracks(random_streamlines[0], qb.downsampled_tracks(), 20)

    maxclose = np.int(np.max(np.hstack((neighbours_Q,neighbours_R))))

    # The numbers of tracks 0, 1, 2, ... 'close' subset tracks
    counts = [(np.int(n), len(find(neighbours_Q==n)), len(find(neighbours_R==n)))
              for n in range(maxclose+1)]

    print np.array(counts)

    # Typically counts_Q shows (a) very few tracks with 0 close QB
    # centroids, (b) many tracks with a small number (between 1 and 3?) close QB
    # tracks, and (c) few tracks with many (>3?) close QB tracks

    # By contrast counts_R shows (a) a large number of tracks with 0 close
    # R (random) neighbours, (b) fewer tracks with a small number of close R
    # tracks, and (c) a long tail showing how the R sample has over-sampled
    # in dense parts of the tractography, coming up with several rather
    # similar tracks. By contast the QB tracks are dissimilar by design - or
    # can be thought of as more evenly distributed in track space.

    # The output below was generated with subject 02, 5k tracks, and threshold 20.
    # Column 0 is the neighbour count, and Columns 1 and 2 are the
    # number of tracks with that neighbour count.

    # I suppose you could say this revealed some kind of sparseness for the
    # QB subset by comparison with the Random one
    """


