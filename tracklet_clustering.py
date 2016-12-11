__author__ = 'aclapes'

from os import path, makedirs, remove, walk
import cPickle
import random
import time
from math import isnan
import sys
import threading

import numpy as np
from sklearn.metrics import pairwise
from sklearn.neighbors import KDTree

from Queue import PriorityQueue

from spectral_division import spectral_embedding_nystrom, spectral_clustering_division

import cv2
from joblib import delayed, Parallel

from pymediainfo import MediaInfo

INTERNAL_PARAMETERS = dict(
    initial_ridge_value = 1e-10,
    L = 15,
    feats_dict = dict(
        obj = 10,
        trj = 2,  # eventually multiplied by length of trajectory L (further see the code)
        hog = 96,
        hof = 108,
        mbh = 192),
    nx=4,
    ny=3,
    p=0.01
)


def cluster_multithread(tracklets_path, videofiles, clusters_path, nt=4, verbose=False, visualize=False):
    try:
        makedirs(clusters_path)
    except (OSError, IOError):
        pass

    # Choose between: random order or not
    # inds = np.random.permutation(len(videofiles)).astype('int')
    inds = np.linspace(0,len(videofiles)-1,len(videofiles)).astype('int')
    lock = threading.Lock()  # avoid _cluster threads concurrently read IDT files from disk
    Parallel(n_jobs=nt, backend='threading')(delayed(cluster)(tracklets_path, videofiles, [i],
                                                              clusters_path, lock=lock, verbose=verbose, visualize=visualize)
                                                     for i in inds)


def cluster(tracklets_path, videofiles, indices, clusters_path, lock=None, verbose=False, visualize=False):
    """
    This function implements the method described in Section 2 ("Clustering dense tracklets")
    of the paper 'Activity representation with motion hierarchies' (IJCV, 2014).
    :param tracklets_path:
    :param videofiles:
    :param indices:
    :param clusters_path:
    :param visualize:
    :return:
    """


    obj_range = (0, INTERNAL_PARAMETERS['feats_dict']['obj'])
    trj_range = (INTERNAL_PARAMETERS['feats_dict']['obj'], \
                 INTERNAL_PARAMETERS['feats_dict']['obj']+(INTERNAL_PARAMETERS['L']+1)*INTERNAL_PARAMETERS['feats_dict']['trj'])

    # process the videos
    total = len(videofiles)
    for i in indices:
        media_info = MediaInfo.parse(videofiles[i])
        tracks = {track.track_type : track for track in media_info.tracks}

        videoname_stem = path.splitext(path.basename(videofiles[i]))[0]  # path.splitext(videofiles[i])[0]
        print videoname_stem
        clusters_file_path = path.join(clusters_path, videoname_stem + '.pkl')
        if path.isfile(clusters_file_path):
            # DEBUG
            # -----
            if visualize:
                data_obj = data_trj = None
                try:
                    # if lock is not None:
                    #     lock.acquire()
                    with open(path.join(tracklets_path, videoname_stem + '.idt'), 'r', 1) as f:
                        lines = [line.split() for line in f.readlines()]
                        data_obj  = np.array([np.array(line[obj_range[0]:obj_range[1]], dtype=np.float32) for line in lines])
                        data_trj  = np.array([np.array(line[trj_range[0]:trj_range[1]], dtype=np.float32) for line in lines])
                    # if lock is not None:
                    #     lock.release()
                except IOError:
                    sys.stderr.write("[Error] Tracklet files not found for %s." % videofiles[i])
                    continue

                with open(path.join(clusters_path, videoname_stem + '.pkl'), 'r', 1) as f:
                    clusters = cPickle.load(f)
                inliers = clusters['tracklet_inliers']

                visualize_tree_leafs(tracks['Video'], videofiles[i], data_obj[inliers,:], data_trj[inliers,:], clusters['int_paths'],isolated_leafs=False)

            if verbose:
                print('[_cluster] %s.pkl -> OK' % videoname_stem)
            continue

        data_obj = data_trj = None
        try:
            # if lock is not None:
            #     lock.acquire()
            with open(path.join(tracklets_path, videoname_stem + '.idt'), 'r', 1) as f:
                lines = [line.split() for line in f.readlines()]
                data_obj  = np.array([np.array(line[obj_range[0]:obj_range[1]], dtype=np.float32) for line in lines])
                data_trj  = np.array([np.array(line[trj_range[0]:trj_range[1]], dtype=np.float32) for line in lines])
            # if lock is not None:
            #     lock.release()
        except IOError:
            sys.stderr.write("[Error] Tracklet files not found for %s." % videofiles[i])
            continue

        try:
            with open(path.join(clusters_path, videoname_stem + '.filtering.pkl'), 'rb') as f:
                inliers = cPickle.load(f)
        except IOError:
            inliers = filter_low_density(data_obj)
            with open(path.join(clusters_path, videoname_stem + '.filtering.pkl'), 'wb') as f:
                cPickle.dump(inliers,f)
            if verbose:
                print( '[_cluster] %d/%d kept after low density filtering' % (len(inliers),data_obj.shape[0]))

        if verbose:
            print len(inliers), data_obj.shape[0], float(len(inliers))/data_obj.shape[0]

        data_obj = data_obj[inliers,:]
        data_trj = data_trj[inliers,:]

        start_time = time.time()
        # (Sec. 2.2) get a dictionary of separate channels
        D = dict()
        for k in xrange(data_obj.shape[0]): # range(0,100):  #
            T = np.reshape(data_trj[k], (data_trj.shape[1]/2,2))  # trajectory features into matrix (time length x 2)
            D.setdefault('x',[]).append( T[:-1,0]/tracks['Video'].width )  # x's offset + x's relative displacement
            D.setdefault('y',[]).append( T[:-1,1]/tracks['Video'].height )  #  y's offset + y's relative displacement
            Tt = (data_obj[k,0] - np.linspace(T[:-1].shape[0]-1, 0, T[:-1].shape[0]))
            D.setdefault('t',[]).append( Tt/float(tracks['Video'].frame_count) )
            D.setdefault('v_x',[]).append( (T[1:,0] - T[:-1,0]) / tracks['Video'].width )
            D.setdefault('v_y',[]).append( (T[1:,1] - T[:-1,1]) /tracks['Video'].height )

        # (Sec. 2.3.1)
        # A, B = get_tracklet_similarities(D, data_obj[:,7:9])
        # create a subsample (n << N) stratified by a grid
        if verbose:
            print 'stratified_subsample_of_tracklets_in_grid'
        # try:
        #     with open(path.join(clusters_path, videoname_stem + '.gridsample.pkl'), 'rb') as f:
        #         insample, outsample, nx, ny, p = cPickle.load(f)
        #         if INTERNAL_PARAMETERS['nx'] != nx or INTERNAL_PARAMETERS['ny'] != ny or INTERNAL_PARAMETERS['p'] != p:
        #             raise IOError('nx,ny,p values differ from the ones used in temporary gridsample file.')
        # except IOError:
        nx, ny, p = INTERNAL_PARAMETERS['nx'], INTERNAL_PARAMETERS['ny'], INTERNAL_PARAMETERS['p']
        insample, outsample = stratified_subsample_of_tracklets_in_grid(data_obj[:,7:9], nx=nx, ny=ny, p=p)
        with open(path.join(clusters_path, videoname_stem + '.gridsample.pkl'), 'wb') as f:
            cPickle.dump((insample,outsample,nx,ny,p), f)

        if verbose:
            print len(insample), len(outsample)

        if verbose:
            print 'multimodal_product_kernel'
        # get the similarities of
        try:
            with open(path.join(clusters_path, videoname_stem + '.mpkern.pkl'), 'rb') as f:
                AB = cPickle.load(f)
        except:
            A, medians = multimodal_product_kernel(D, insample, insample)  # (n), n << N tracklets
            B, _ = multimodal_product_kernel(D, insample, outsample, medians=medians)  # (N - n) tracklets
            del D  # clear memory
            AB = np.hstack((A,B)).astype('float64')
            with open(path.join(clusters_path, videoname_stem + '.mpkern.pkl'), 'wb') as f:
                cPickle.dump(AB,f)

        if verbose:
            print 'spectral_embedding_nystrom'
        try:
            with open(path.join(clusters_path, videoname_stem + '.embedding.pkl'), 'rb') as f:
                E,ridge = cPickle.load(f)
                if ridge != INTERNAL_PARAMETERS['initial_ridge_value']:
                    raise IOError('New ridge value established, different form the one on temporary embedding file.')
        except IOError:
            # (Sec. 2.3.2 and 2.3.3)
            E = spectral_embedding_nystrom(AB, ridge=INTERNAL_PARAMETERS['initial_ridge_value'])
            with open(path.join(clusters_path, videoname_stem + '.embedding.pkl'), 'wb') as f:
                cPickle.dump((E,INTERNAL_PARAMETERS['initial_ridge_value']),f)

        if verbose:
            print 'spectral_clustering_division]'
        # (Sec. 2.4)
        # pre-requisite: re-organizing obj rows according to in- and out-sample indices
        orgdata_obj = np.vstack( (data_obj[:len(insample),:], data_obj[len(insample):,:]) )
        division = spectral_clustering_division(E, orgdata_obj[:,7:10])

        # re-organize division according to Sec 2.3.1's sampling
        best_labels = np.zeros(division[0].shape, dtype=division[0].dtype)
        int_paths   = np.zeros(division[1].shape, dtype=division[1].dtype)
        best_labels[insample], best_labels[outsample] = division[0][:len(insample)], division[0][len(insample):]
        int_paths[insample], int_paths[outsample] = division[1][:len(insample)], division[1][len(insample):]

        tree = reconstruct_tree_from_leafs(np.unique(int_paths))

        elapsed_time = time.time() - start_time
        if verbose:
            print('[_cluster] %s (in %.2f secs)' % (videofiles[i], elapsed_time))

        with open(path.join(clusters_path, videoname_stem + '.pkl'), 'wb') as f:
            cPickle.dump({'tracklet_inliers' : inliers, 'best_labels' : best_labels, 'int_paths' : int_paths,
                          'tree_keys' : tree.keys(), 'tree_values' : tree.values(),  # avoid storing dicts (in case scipy.io.savemat is used)
                          'ridge' : INTERNAL_PARAMETERS['initial_ridge_value']}, f)
            # remove temp files
            remove(path.join(clusters_path, videoname_stem + '.filtering.pkl'))
            remove(path.join(clusters_path, videoname_stem + '.gridsample.pkl'))
            remove(path.join(clusters_path, videoname_stem + '.mpkern.pkl'))
            remove(path.join(clusters_path, videoname_stem + '.embedding.pkl'))

        if visualize:
            visualize_tree_leafs(tracks['Video'], data_obj, data_trj, int_paths, isolated_leafs=True)



# ==============================================================================
# Helper functions
# ==============================================================================

# def visualize_tree_leafs(video_track, video_frames, data_obj, data_trj, leaf_labels, isolated_leafs=False):
#     A = np.zeros((video_track.height/2.,1280,3), dtype=np.uint8)
#     t_step = (A.shape[1]*2.0) / float(video_track.frame_count)
#
#     s = (A.shape[1]-(video_track.width/2)) / (float(video_track.frame_count)-1)
#
#     for i,frame in enumerate(video_frames):
#         rframe = cv2.resize(frame, (0,0), fx=1/2., fy=1/2.)
#         fid = min(i * float(video_track.frame_count)/(len(video_frames)-1), float(video_track.frame_count)-1)
#         offset = fid * s
#         A[:,0+offset:rframe.shape[1]+offset,:] = rframe
#
#     unique_labels = np.unique(leaf_labels)
#     n_unique_labels = len(unique_labels)
#
#     for i,label in enumerate(unique_labels):
#         cluster_inds = np.where(leaf_labels == label)[0]
#         print label
#         hue = ((float(i)/n_unique_labels))  # + random.random() + random.random()) % 1
#         for k in xrange(0, len(cluster_inds)):
#             T = np.reshape(data_trj[cluster_inds[k],:], (data_trj.shape[1]/2,2))
#             t = int(data_obj[cluster_inds[k],0])
#             L = data_obj[0,0]
#             for j in range(1,T.shape[0]):
#                 pt1 = ( int((T[j-1,0]+(t-L)*t_step)/2.0), int(T[j-1,1]/2.0) )
#                 pt2 = ( int((T[j,0]+(t-L)*t_step)/2.0), int(T[j,1]/2.0) )
#                 if j < T.shape[0]-1:
#                     cv2.line(A, pt1, pt2, hsv_to_rgb((hue,0.5,0.8)),thickness=1)
#                 else:
#                     cv2.arrowedLine(A, pt1, pt2, hsv_to_rgb((hue,0.5,0.8)),thickness=1,tipLength=0.5)
#
#         if isolated_leafs:
#             cv2.imshow("#DEBUG Clustering visualization", A)
#             cv2.waitKey(0)
#             A = np.zeros((video_track.height/2.0,1280,3), dtype=np.uint8)
#     cv2.imshow("#DEBUG Clustering visualization", A)
#     cv2.waitKey(0)

# def visualize_tree_leafs(video_track, videofile, data_obj, data_trj, leaf_labels, isolated_leafs=False):
#     cap = cv2.VideoCapture(videofile)
#     frame_count = float(video_track.frame_count)
#     n_viz_frames = 4
#     viz_frames = [None] * n_viz_frames
#     for i in range(0,n_viz_frames):
#         fid = int(i * (frame_count/(n_viz_frames-1)))
#         fid = min(fid,frame_count-1)  # don't go index out of bounds
#         cap.set(1,fid);
#         _, viz_frames[i] = cap.read()
#
#     data_trj = data_trj[data_obj[:,6]==1]
#     leaf_labels = leaf_labels[data_obj[:,6]==1]
#     data_obj = data_obj[data_obj[:,6]==1]
#
#     width = video_track.width*n_viz_frames
#     # t_step = (width - video_track.width)/(float(video_track.frame_count)-1)
#     t_step = (width)/(float(video_track.frame_count)-15)
#     # A = 255*np.ones((video_track.height,video_track.width+t_step*(float(video_track.frame_count)-1),3), dtype=np.uint8)
#     A = 255*np.ones((video_track.height,width,3), dtype=np.uint8)
#     # step = (3./4)*video_track.width
#
#     for i,frame in enumerate(viz_frames):
#         # fid = min(i * float(video_track.frame_count)/(len(video_frames)-1), float(video_track.frame_count)-1)
#         fid = (i * (frame_count))/n_viz_frames
#         roi = A[:,t_step*fid:video_track.width+t_step*fid,:]
#         frame_roi = cv2.resize(frame, (roi.shape[1],roi.shape[0]))
#         A[:,t_step*fid:video_track.width+t_step*fid,:] = frame_roi
#
#     unique_labels = np.unique(leaf_labels)
#     n_unique_labels = len(unique_labels)
#
#     for i,label in enumerate(unique_labels):
#         cluster_inds = np.where(leaf_labels == label)[0]
#         print label
#         hue = ((float(i)/n_unique_labels))  # + random.random() + random.random()) % 1
#         for k in xrange(0, len(cluster_inds)):
#             T = np.reshape(data_trj[cluster_inds[k],:], (data_trj.shape[1]/2,2))
#             L = data_obj[0,0]
#             t = np.linspace(data_obj[cluster_inds[k],0]-L+1, data_obj[cluster_inds[k],0], L)-1
#             for j in range(1,T.shape[0]):
#                 x1, y1 = T[j-1,0] + (t[j] * t_step), T[j-1,1]
#                 x2, y2 = T[  j,0] + (t[j] * t_step), T[  j,1]
#                 pt1 = (int(x1),int(y1))
#                 pt2 = (int(x2),int(y2))
#                 if j < T.shape[0]-1:
#                     cv2.line(A, pt1, pt2, hsv_to_rgb((hue,0.5,0.8)),thickness=2)
#                 else:
#                     cv2.arrowedLine(A, pt1, pt2, hsv_to_rgb((hue,0.5,0.8)),thickness=2,tipLength=0.5)
#
#     A = cv2.resize(A, (0,0), fx=1/2., fy=1/2.)
#     cv2.imshow("#DEBUG Clustering visualization", A)
#     cv2.waitKey(0)


# def visualize_tree_leafs(video_track, videofile, data_obj, data_trj, leaf_labels, isolated_leafs=False):
#
#     cap = cv2.VideoCapture(videofile)
#
#     end_frame = float(video_track.frame_count)-1
#     frame_count = float(video_track.frame_count)-15.
#
#     n_viz_frames = 5
#     fids = [None] * n_viz_frames
#     viz_frames = [None] * n_viz_frames
#
#     for i in range(0,n_viz_frames):
#         fids[i] = i * (frame_count/(n_viz_frames-1)) + 15
#         fids[i] = int(min(fids[i],end_frame))
#         cap.set(1,fids[i]);
#         ret, viz_frames[i] = cap.read()
#         while not ret:
#             fids[i] -= 1
#             cap.set(1,fids[i]);
#             ret, viz_frames[i] = cap.read()
#         print i, fids[i], ret
#
#
#     # data_trj = data_trj[data_obj[:,6]==1]
#     # leaf_labels = leaf_labels[data_obj[:,6]==1]
#     # data_obj = data_obj[data_obj[:,6]==1]
#
#     width = video_track.width*n_viz_frames
#     t_step = width/frame_count
#     A = 255*np.ones((video_track.height,width,3), dtype=np.uint8)
#
#     for i,frame in enumerate(viz_frames):
#         # fid = min(i * float(video_track.frame_count)/(len(video_frames)-1), float(video_track.frame_count)-1)
#         fid = (i * (frame_count))/n_viz_frames
#         try:
#             print (i, t_step*fid, video_track.width+t_step*fid),
#             print (video_track.width+t_step*fid) -  t_step*fid,
#             x1 = int(np.round(t_step*fid))
#             x2 = int(np.round(video_track.width+t_step*fid))
#             A[:,x1:x2,:] = cv2.resize(frame, (x2-x1, frame.shape[0]))
#             print ' -> ok'
#         except cv2.error as e:
#             print e
#         except TypeError as e:
#             print e
#
#     unique_labels = np.unique(leaf_labels)
#     n_u_labels = len(unique_labels)
#     colors = {lbl:(float(i)/n_u_labels) for i,lbl in enumerate(unique_labels)}
#
#     for f,fid in enumerate(fids):
#         frame_inds = np.where(data_obj[:,0] == fid)[0]
#         print len(frame_inds)
#         u_frame_labels = np.unique(leaf_labels[frame_inds])
#         for i,label in enumerate(u_frame_labels):
#             cluster_inds = np.where(leaf_labels[frame_inds] == label)[0]
#             for k in xrange(len(cluster_inds)):
#                 T = np.reshape(data_trj[frame_inds][k], (data_trj.shape[1]/2,2))
#                 # print k
#                 for j in range(1,T.shape[0]):
#                     x1, y1 = T[j-1,0] + (f * video_track.width), T[j-1,1]
#                     x2, y2 = T[  j,0] + (f * video_track.width), T[  j,1]
#                     pt1 = (int(x1),int(y1))
#                     pt2 = (int(x2),int(y2))
#                     cval = 0.5+0.5*(float(j+1)/(T.shape[0]+1))
#                     hsv = hsv_to_rgb((colors[label],1,cval))
#                     cv2.line(A, pt1, pt2, hsv,thickness=1,lineType=cv2.CV_AA)
#                 cv2.circle(A, pt2, 2, hsv_to_rgb((colors[label],1,1)), -1)
#
#
#     #
#     #     unique_labels = np.unique(frame_labels)
#     #     n_unique_labels = len(unique_labels)
#     #
#     #     for i,label in enumerate(unique_labels):
#     #         cluster_inds = np.where(leaf_labels == label)[0]
#     #         hue = ((float(i)/n_unique_labels))  # + random.random() + random.random()) % 1
#     #         for k in xrange(0, len(cluster_inds)):
#     #             T = np.reshape(data_trj[cluster_inds[k],:], (data_trj.shape[1]/2,2))
#     #             L = data_obj[0,0]
#     #             t = np.linspace(data_obj[cluster_inds[k],0]-L+1, data_obj[cluster_inds[k],0], L)-1
#     #             for j in range(1,T.shape[0]):
#     #                 x1, y1 = T[j-1,0] + (t[j] * t_step), T[j-1,1]
#     #                 x2, y2 = T[  j,0] + (t[j] * t_step), T[  j,1]
#     #                 pt1 = (int(x1),int(y1))
#     #                 pt2 = (int(x2),int(y2))
#     #                 if j < T.shape[0]-1:
#     #                     cv2.line(A, pt1, pt2, hsv_to_rgb((hue,0.5,0.8)),thickness=2)
#
#     A = cv2.resize(A, (0,0), fx=.66, fy=.66)
#     cv2.imwrite(path.splitext(path.basename(videofile))[0] + '.png', A)
#     # cv2.imshow("#DEBUG Clustering visualization", A)
#     # cv2.waitKey(0)

def visualize_tree_leafs(video_track, videofile, data_obj, data_trj, leaf_labels, isolated_leafs=False):

    cap = cv2.VideoCapture(videofile)

    end_frame = float(video_track.frame_count)-1
    frame_count = float(video_track.frame_count)-15.

    n_viz_frames = 5
    fids = [None] * n_viz_frames
    viz_frames = [None] * n_viz_frames

    for i in range(0,n_viz_frames):
        fids[i] = i * (frame_count/(n_viz_frames-1)) + 15
        fids[i] = int(min(fids[i],end_frame))
        while cap.get(1) < fids[i]:
            cap.grab()
        cap.grab()
        ret, viz_frames[i] = cap.retrieve()
        while not ret:
            fids[i] -= 1
            cap.set(1,fids[i]);
            ret, viz_frames[i] = cap.read()

    unique_labels = np.unique(leaf_labels)
    n_u_labels = len(unique_labels)
    colors = {lbl:(float(i)/n_u_labels) for i,lbl in enumerate(unique_labels)}

    for f,fid in enumerate(fids):
        frame_inds = np.where(data_obj[:,0] == fid)[0]
        print len(frame_inds)
        u_frame_labels = np.unique(leaf_labels[frame_inds])
        for i,label in enumerate(u_frame_labels):
            cluster_inds = np.where(leaf_labels[frame_inds] == label)[0]
            for k,idx in enumerate(cluster_inds):
                T = np.reshape(data_trj[frame_inds][idx], (data_trj.shape[1]/2,2))
                for j in range(1,T.shape[0]):
                    x1, y1 = T[j-1,0], T[j-1,1]
                    x2, y2 = T[  j,0], T[  j,1]
                    pt1 = (int(x1),int(y1))
                    pt2 = (int(x2),int(y2))
                    cval = 0.5+0.5*(float(j+1)/(T.shape[0]+1))
                    hsv = hsv_to_rgb((colors[label],1,cval))
                    cv2.line(viz_frames[f], pt1, pt2, hsv,thickness=1,lineType=cv2.CV_AA)
                cv2.circle(viz_frames[f], pt2, 2, hsv_to_rgb((colors[label],1,1)), -1)

    width = video_track.width*n_viz_frames
    t_step = width/frame_count
    A = 255*np.ones((video_track.height,width,3), dtype=np.uint8)

    for i,frame in enumerate(viz_frames):
        # fid = min(i * float(video_track.frame_count)/(len(video_frames)-1), float(video_track.frame_count)-1)
        fid = (i * (frame_count))/n_viz_frames
        try:
            print (i, t_step*fid, video_track.width+t_step*fid),
            print (video_track.width+t_step*fid) -  t_step*fid,
            x1 = int(np.round(t_step*fid))
            x2 = int(np.round(video_track.width+t_step*fid))
            A[:,x1:x2,:] = cv2.resize(frame, (x2-x1, frame.shape[0]))
            print ' -> ok'
        except cv2.error as e:
            print e
        except TypeError as e:
            print e

    A = cv2.resize(A, (0,0), fx=.66, fy=.66)
    cv2.imwrite(path.splitext(path.basename(videofile))[0] + '.png', A)
    # cv2.imshow("#DEBUG Clustering visualization", A)
    # cv2.waitKey(0)
#
def filter_low_density(data, k=30, r=5):
    """
    Filter out low density tracklets from the sequence.
    :param data: the tracklets, a T x num_features matrix.
    :return:
    """

    # each tracklet's mean x and y position
    P = data[:,1:3]  # (these are index 1 and 2 of data)

    all_sparsities = np.zeros((P.shape[0],k), dtype=np.float32)
    subset_indices = []  # optimization. see (*) below
    for i in range(0, P.shape[0]):
        new_subset_indices = np.where((data[:,0] >= data[i,0] - r) & (data[:,0] <= data[i,0] + r))[0]
        if len(new_subset_indices) == 1:
            all_sparsities[i,:] = np.nan
        else:
            # (*) a new KDTree is constructed only if the subset of data changes
            if not np.array_equal(new_subset_indices, subset_indices):
                subset_indices = new_subset_indices
                tree = KDTree(P[subset_indices,:], leaf_size=1e2)

            p = P[i,np.newaxis] # query instance
            if k+1 <= len(subset_indices):
                dists, inds = tree.query(p, k=k+1)
                dists = dists[0,1:]  # asked the neighbors of only 1 instance, returned in dists as 0-th element
            else:  #len(subset_indices) > 1:
                dists, inds = tree.query(p, k=len(subset_indices))
                dists = np.concatenate([dists[0,1:], [np.nan]*(k-len(dists[0,1:]))])
            all_sparsities[i,:] = dists

    local_sparsities = np.nanmean(all_sparsities, axis=1)
    mean_sparsity = np.nanmean(all_sparsities)
    stddev_sparsity = np.nanstd(all_sparsities)

    f = 1.0
    while f <= 3.0:
        inliers = np.where(local_sparsities <= (mean_sparsity + f * stddev_sparsity))[0]
        if len(inliers) > 0:  # all ok
            return inliers
        f += 1.0

    return np.where(~np.isnan(local_sparsities))[0]


def stratified_subsample_of_tracklets_in_grid(P, nx=4, ny=3, p=0.01):
    """
    Subsample a factor p of the total tracklets stratifying the sampling in a
    grid of nx-by-ny cells.
    :param P: N-by-2 matrix of tracklet (ending) positions
    :param p: the sampling probability
    :param nx: number of horizontal divisions of the grid
    :param ny: number of vertical divisions of the grid
    :return insample, outsample:
    """
    MIN_SAMPLE_SIZE = 10*nx*ny
    MAX_SAMPLE_SIZE = 100*nx*ny
    p_cell = max(min(P.shape[0]*p, MAX_SAMPLE_SIZE), MIN_SAMPLE_SIZE) / float(nx*ny)
    insample = []
    outsample = []
    for i in range(0,ny):
        y_ran = (i*(1.0/ny), (i+1)*(1.0/ny))
        for j in range(0,nx):
            x_ran = (j*(1.0/nx), (j+1)*(1.0/nx))
            cell_inds = np.where((P[:,0] >= x_ran[0]) & (P[:,0] < x_ran[1]) & (P[:,1] >= y_ran[0]) & (P[:,1] < y_ran[1]))[0]
            n = int(min(p_cell, len(cell_inds)))
            random.shuffle(cell_inds)
            insample.append( cell_inds[:n].astype(dtype=np.int32) )
            outsample.append( cell_inds[n:].astype(dtype=np.int32) )

    return np.concatenate(insample), np.concatenate(outsample)


def multimodal_product_kernel(D, primary_inds=None, secondary_inds=None, medians=None):
    """
    Merges the different modalities (or channels) using the product of rbf kernels.
    The similarity matrix computed is the one from the samples in the primary indices to the secondary indices.
    If some indices are not specified (None) all samples are used.
    :param D: a python dict containing the data in the different modalitites (or channels).
    keys are the names of the modalities
    :param primary_inds:
    :param secondary_inds:
    :return K:
    """
    n = len(primary_inds) if primary_inds is not None else len(D['x'])
    m = len(secondary_inds) if secondary_inds is not None else len(D['x'])

    channels = ['x','y','t','v_x','v_y']
    if medians is None:
        medians = []

    K = np.ones((n, m), dtype=np.float32)  # prepare kernel product
    for i, channel_t in enumerate(channels):
        D[channel_t] = np.array(D[channel_t], dtype=np.float64)
        X_primary = D[channel_t][primary_inds] if primary_inds is not None else D[channel_t]
        # print channel_t, X_primary.shape[1]
        X_secondary = D[channel_t][secondary_inds] if secondary_inds is not None else D[channel_t]
        S = pairwise.pairwise_distances(X=X_primary, Y=X_secondary, metric='euclidean')
        median = np.nanmedian(S[S!=0])
        if len(medians) == len(channels):
            median = medians[i]
        else:
            medians.append(median)
        gamma = 1.0/(2*median) if (not isnan(median) and median != 0.0) else 0.0
        K_tmp = np.exp(-gamma * np.power(S,2)) # rbf kernel and element-wise multiplication
        K = np.multiply(K, K_tmp)

    return K, medians


def reconstruct_tree_from_leafs(leafs):
    """
    Given a list of leaf, recover all the nodes.

    Parameters
    ----------
    leafs:  Leafs are integers, each representing a path in the binary tree.
            For instance, a leaf value of 5 indicates the leaf is the one
            reached going through the following path: root-left-right.

    Returns
    -------
    A dictionary indicating for each node a list of all its descendents.
    Exemple:
        { 1 : [2,3,4,5,6,7,12,13,26,27],
          2 : [4,5],
          3 : [6,7,12,13,26,27],
          ...
        }
    """
    h = dict()
    q = PriorityQueue()

    # recover first intermediate nodes (direct parents from leafs)
    for path in leafs:
        parent_path = int(path/2)
        if not parent_path in h and parent_path > 1:
            q.put(-parent_path)  # deeper nodes go first (queue reversed by "-")
        h.setdefault(parent_path, []).append(path)

    # recover other intermediates notes recursevily
    while not q.empty():
        path = -q.get()
        parent_path = int(path/2)
        if not parent_path in h and parent_path > 1:  # list parent also for further processing
            q.put(-parent_path)

        h.setdefault(parent_path, [])
        h[parent_path] += ([path] + h[path])  # append children from current node to their parent

    # update with leafs
    h.update(dict((i,[i]) for i in leafs))

    return h


def hsv_to_rgb(hsv):
    '''
    HSV values in [0..1]
    :param h:
    :param s:
    :param v:
    :return (r, g, b) tuple, with values from 0 to 255:
    '''
    h, s, v = hsv[0], hsv[1], hsv[2]

    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f*s)
    t = v * (1 - (1 - f) * s)

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q

    return (int(r*256), int(g*256), int(b*256))


if __name__ == "__main__":
    #
    # Example of parametrization:
    # /Volumes/MacintoshHD/Users/aclapes/Datasets/ucf_sports_actions/
    # /Volumes/MacintoshHD/Users/aclapes/Derived/improved_dense_trajectories/ucf_sports_actions/
    # /Volumes/MacintoshHD/Users/aclapes/Derived/spectral_clustering/ucf_sports_actions/
    # avi
    #

    #
    # Example of parametrization:
    # /Volumes/MacintoshHD/Users/aclapes/Datasets/highfive/tv_human_interactions_videos/
    # /Volumes/MacintoshHD/Users/aclapes/Derived/improved_dense_trajectories/highfive/
    # /Volumes/MacintoshHD/Users/aclapes/Derived/spectral_clustering/highfive/
    # avi
    #

    videos_path = sys.argv[1]
    tracklets_path = sys.argv[2]
    clusters_path = sys.argv[3]
    video_ext = sys.argv[4]  # extension name, e.g. "mp4"

    print('INPUT (videos,tracklets): %s; %s; \nOUTPUT (clusters): %s.' % (videos_path, tracklets_path, clusters_path))

    ext = '.' + video_ext
    videofiles = [path.join(dp, f) for dp, dn, fn in walk(path.expanduser(videos_path)) for f in fn if f.endswith(ext)]

    cluster(tracklets_path, ['/Volumes/MacintoshHD/Users/aclapes/Datasets/highfive/tv_human_interactions_videos/kiss_0038.avi'], clusters_path, nt=1, verbose=True, visualize=True)
    cluster(tracklets_path, ['/Volumes/MacintoshHD/Users/aclapes/Datasets/highfive/tv_human_interactions_videos/hug_0017.avi'], clusters_path, nt=1, verbose=True, visualize=True)
    cluster(tracklets_path, ['/Volumes/MacintoshHD/Users/aclapes/Datasets/highfive/tv_human_interactions_videos/hug_0029.avi'], clusters_path, nt=1, verbose=True, visualize=True)
    cluster(tracklets_path, ['/Volumes/MacintoshHD/Users/aclapes/Datasets/highfive/tv_human_interactions_videos/hug_0002.avi'], clusters_path, nt=1, verbose=True, visualize=True)
    cluster(tracklets_path, ['/Volumes/MacintoshHD/Users/aclapes/Datasets/highfive/tv_human_interactions_videos/kiss_0040.avi'], clusters_path, nt=1, verbose=True, visualize=True)

    cluster(tracklets_path, videofiles, clusters_path, nt=1, verbose=True, visualize=True)

    # cluster(tracklets_path, ['/Volumes/MacintoshHD/Users/aclapes/Datasets/ucf_sports_actions/SkateBoarding-Front_004.avi'], clusters_path, nt=1, verbose=True, visualize=True)
    # cluster(tracklets_path, ['/Volumes/MacintoshHD/Users/aclapes/Datasets/ucf_sports_actions/Run-Side_001.avi'], clusters_path, nt=1, verbose=True, visualize=True)
    # cluster(tracklets_path, ['/Volumes/MacintoshHD/Users/aclapes/Datasets/ucf_sports_actions/Kicking-Front_002.avi'], clusters_path, nt=1, verbose=True, visualize=True)
    # cluster(tracklets_path, ['/Volumes/MacintoshHD/Users/aclapes/Datasets/ucf_sports_actions/Golf-Swing-Back_005.avi'], clusters_path, nt=1, verbose=True, visualize=True)
    # cluster(tracklets_path, ['/Volumes/MacintoshHD/Users/aclapes/Datasets/ucf_sports_actions/Golf-Swing-Back_002.avi'], clusters_path, nt=1, verbose=True, visualize=True)

