import numpy as np
from scipy.io import loadmat
from os import listdir,makedirs,walk,remove
from os.path import isfile,join,splitext,exists,basename,expanduser
import h5py
import cPickle
from collections import OrderedDict
import cv2
from sklearn import multiclass, preprocessing, svm, metrics

def get_id_from_darwinfile(filepath):
    stem = splitext(basename(filepath))[0]  # '/home/aclapes/1-Cval1.000000-Gap1-Max-10000.mat' -> '1-Cval1.000000-Gap1-Max-10000'
    return (str.split(stem, '-')[0]).zfill(5)  # '1-Cval1.000000-Gap1-Max-10000' -> '00001'


def load_trajectories(filepath):
    L = 15
    feats_dict = dict(obj = 10, trj = 2, hog = 96, hof = 108, mbh = 192)

    try:
        with open(filepath, 'rb') as f:
            objs = []
            trjs = []
            for lid,line in enumerate(f):
                x = np.array(line.split(), dtype=np.float32)

                obj_range = (0, feats_dict['obj'])
                trj_range = (feats_dict['obj'], feats_dict['obj'] + L*feats_dict['trj'])

                objs.append(x[obj_range[0]:obj_range[1]])
                trjs.append(x[trj_range[0]:trj_range[1]])
            return np.array(objs), np.array(trjs)
        return None, None
    except IOError:
        sys.stderr.write("[Error] Tracklet files not found for %s." % filepath)
        return None, None


def load_clusters(filepath):
    with open(filepath ,'rb') as f:
        return cPickle.load(f)
    return None

def load_branches(branch_filepath, max_depth=4):
    nodeid = None
    try:
        mat = loadmat(branch_filepath)  # raise not implemented ee
        nodeid = np.squeeze(mat['nodeid'])
        Btree = mat['Btree'][np.logical_and(nodeid > 1, nodeid < 2**max_depth),:]
    except (NotImplementedError, ValueError) as e:
        mat = h5py.File(branch_filepath)
        if 'nodeid' in mat.keys():
            nodeid = np.squeeze(mat['nodeid'].value)
            Btree = mat['Btree'].value[:, nodeid < 2**max_depth].T
        else:
            Btree = mat['Btree'].value.T

    if nodeid is None:
        return Btree
    else:
        return {nid:Btree[i,:] for i,nid in enumerate(nodeid[nodeid < 2**max_depth])}

def load_darwins(representations, max_depth=4):
    aux = []
    nodeid = None
    for j in xrange(len(representations)):
        try:
            mat = loadmat(representations[j])  # raise not implemented ee
            if 'W' in mat:
                aux.append(np.squeeze(mat['W']))
            elif 'Wtree' in mat:
                nodeid = np.squeeze(mat['nodeid'])
                Wtree = mat['Wtree'][np.logical_and(nodeid > 1, nodeid < 2**max_depth),:]
                aux.append(Wtree)
            elif 'Btree' in mat:
                nodeid = np.squeeze(mat['nodeid'])
                Btree = mat['Btree'][np.logical_and(nodeid > 1, nodeid < 2**max_depth),:]
                aux.append(Btree)
        except (NotImplementedError, ValueError) as e:
            print 'h5py'
            mat = h5py.File(representations[j][idx])
            if 'W' in mat.keys():
                aux.append(mat['W'].value)
                # raise AssertionError('TODO: assume W is not stored in -v7.3')
            elif 'Wtree' in mat.keys():
                if 'nodeid' in mat.keys():
                    nodeid = np.squeeze(mat['nodeid'].value)
                    Wtree = mat['Wtree'].value[:,nodeid < 2**max_depth].T
                else:
                    Wtree = mat['Wtree'].value.T
                aux.append(Wtree)
            elif 'Btree' in mat.keys():
                if 'nodeid' in mat.keys():
                    nodeid = np.squeeze(mat['nodeid'].value)
                    Btree = mat['Btree'].value[:, nodeid < 2**max_depth].T
                else:
                    Btree = mat['Btree'].value.T
                aux.append(Btree)

    if len(aux) == 1:
        aux = np.ascontiguousarray(aux[0],dtype=np.float32)
    else:
        aux = np.ascontiguousarray(np.concatenate(aux,axis=1),dtype=np.float32)

    if nodeid is None:
        return aux
    else:
        nodeid = nodeid[nodeid < 2**max_depth]
        return np.hstack((nodeid[:,np.newaxis].astype(np.float32), aux))


def sample_training_nodes(training):
    return np.vstack(training)

def supertubeOf(tube, othertube):
    if tube is None:
        return othertube
    if othertube is None:
        return tube

    f_union_min = min(tube.keys()[0], othertube.keys()[0])
    f_union_max = max(tube.keys()[-1], othertube.keys()[-1])

    # f_intersec_min = max(tube[0][0], othertube[0][0])
    # f_intersec_max = min(tube[0][-1], othertube[0][-1])
    #
    # supertube = [None] * (f_union_max-f_union_min+1)
    # supertube[:f_intersec_min-f_union_min] = tube[1][:f_intersec_min-f_union_min] if tube[0][0] < othertube[0][0] else othertube[1][:f_intersec_min - f_union_min]
    # supertube[-(f_union_max-f_intersec_max):] = tube[1][-(f_union_max-f_intersec_max):] if tube[0][-1] > othertube[0][-1] else othertube[1][-(f_union_max - f_intersec_max):]
    #
    # tube_intersec = tube[1][(tube[0] <= f_intersec_max) & (tube[0] >= f_intersec_min)]
    # othertube_intersec = othertube[1][(othertube[0] <= f_intersec_max) & (othertube[0] >= f_intersec_min)]
    # intersec = np.zeros_like(tube_intersec)
    # print tube_intersec.shape, othertube_intersec.shape, intersec.shape
    # for i in xrange(tube_intersec.shape[0]):
    #     print i,
    #     intersec[i] = (min(tube_intersec[i,0],othertube_intersec[i,0]), min(tube_intersec[i,1],othertube_intersec[i,1]),
    #                    max(tube_intersec[i,2],othertube_intersec[i,2]), max(tube_intersec[i,3],othertube_intersec[i,3]))
    # print
    # supertube[f_intersec_min-f_union_min:f_intersec_min-f_union_min+intersec.shape[0]] = intersec
    #
    # return (np.linspace(f_union_min,f_union_max,f_union_max-f_union_min+1).astype('int'), np.array(supertube))

    supertube = OrderedDict()
    for f in range(f_union_min,f_union_max+1):
        if f not in tube:
            try:
                supertube[f] = othertube[f]
            except KeyError:
                pass
        elif f not in othertube:
            try:
                supertube[f] = tube[f]
            except KeyError:
                pass
        elif f in tube and f in othertube:
            tube_f = tube[f]
            othertube_f = othertube[f]
            supertube[f] = (min(tube_f[0],othertube_f[0]), min(tube_f[1],othertube_f[1]),
                            max(tube_f[2],othertube_f[2]), max(tube_f[3],othertube_f[3]))

    if len(supertube) == 0:
        print 'problem'

    return supertube


def find_tubes(T,nids,C):
    tubes_all = dict()
    for nid in np.unique(nids):
        node_msk = (nids == nid)
        tf = np.squeeze(T[0][node_msk])
        xf = T[1][node_msk,-2]
        yf = T[1][node_msk,-1]
        f = np.unique(tf)
        tubes_all[nid] = OrderedDict((f_i, (xf[tf==f_i].min(), yf[tf==f_i].min(), xf[tf==f_i].max(), yf[tf==f_i].max())) for i,f_i in enumerate(f))

    for i,nid in enumerate(C['tree_keys']):
        if nid in tubes_all:
            continue
        childs = C['tree_values'][i]
        supertube = None
        for nid_c in childs:
            if nid_c in tubes_all:
                tube_c = tubes_all[nid_c]
                supertube = supertubeOf(supertube,tube_c)
        tubes_all[nid] = supertube

    return tubes_all


def visualize(input_videofile, annots_gt, tubes_pred, output_videofile):

    cap = cv2.VideoCapture(input_videofile)

    ret = True, frame = cap.read()
    writer = cv2.VideoWriter(output_videofile, cv2.cv.CV_FOURCC(*'XVID'),10.0,(frame.shape[1],frame.shape[0]))
    while ret:
        fid = int(cap.get(1))
        ret, frame = cap.read()
        if not ret:
            break

        annot = annots_gt[fid]

        for nid,tube in tubes_pred.iteritems():
            if fid in tube:
                box = tube[fid]
                color = (0,255,0) if overlap(box,annot) >= 0.6 else (0,0,255)
                cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),color,thickness=1)

        cv2.rectangle(frame,(annot[0],annot[1]),(annot[2],annot[3]),(255,0,0),thickness=1)

        writer.write(frame)

    writer.release()


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

def visualize2(input_videofile, trajs, nids, annots_gt, tubes_train, output_videofile, tubes_discarded=None):

    cap = cv2.VideoCapture(input_videofile)

    unique_labels = np.unique(nids)
    colors = {lbl:(float(i)/len(unique_labels)) for i,lbl in enumerate(unique_labels)}

    ret = True, frame = cap.read()
    writer = cv2.VideoWriter(output_videofile, cv2.cv.CV_FOURCC(*'XVID'),5.0,(frame.shape[1],frame.shape[0]))
    while ret:
        fid = int(cap.get(1))
        ret, frame = cap.read()
        if not ret:
            break

        frame_inds = np.where(trajs[0] == fid)[0]

        u_frame_labels = np.unique(nids[frame_inds])
        for i,label in enumerate(u_frame_labels):
            cluster_inds = np.where(nids[frame_inds] == label)[0]
            for k,idx in enumerate(cluster_inds):
                x,y = trajs[1][frame_inds][idx]
                # T = np.reshape(trajs[1][frame_inds][idx], (trajs[1].shape[1]/2,2))
                # for j in range(1,T.shape[0]):
                #     x1, y1 = T[j-1,0], T[j-1,1]
                #     x2, y2 = T[  j,0], T[  j,1]
                #     pt1 = (int(x1),int(y1))
                #     pt2 = (int(x2),int(y2))
                #     cval = 0.5+0.5*(float(j+1)/(T.shape[0]+1))
                #     hsv = hsv_to_rgb((colors[label],1,cval))
                #     cv2.line(frame, pt1, pt2, hsv,thickness=1,lineType=cv2.CV_AA)
                # cv2.circle(frame, pt2, 2, hsv_to_rgb((colors[label],1,1)), -1)
                cv2.circle(frame, (int(x),int(y)), 2, hsv_to_rgb((colors[label],1,1)), -1)

        for nid,tube in tubes_train.iteritems():
            if fid in tube:
                box = tube[fid]
                cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),(0,255,255),thickness=2)

        if tubes_discarded:
            for nid,tube in tubes_discarded.iteritems():
                if fid in tube:
                    box = tube[fid]
                    cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),(0,0,255),thickness=1)

        annot = annots_gt[fid]
        cv2.rectangle(frame,(annot[0],annot[1]),(annot[2],annot[3]),(255,0,0),thickness=2)


        writer.write(frame)

    writer.release()


def sample_training(annots_gt, tubes_pred):
    tubes_train = OrderedDict()
    tubes_discarded = OrderedDict()
    for nid,tube in tubes_pred.iteritems():
        if nid == 1:
            continue

        acc_ovl = 0
        for fid,box in tube.iteritems():
            acc_ovl += overlap(box,annots_gt[fid])
        if acc_ovl / len(tube) >= 0.5:
            tubes_train[nid] = tube
        else:
            tubes_discarded[nid] = tube

    return tubes_train, tubes_discarded


def union(a,b):
  x = min(a[0], b[0])
  y = min(a[1], b[1])
  w = max(a[0]+a[2], b[0]+b[2]) - x
  h = max(a[1]+a[3], b[1]+b[3]) - y
  return w*h  #(x, y, w, h)

def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: return 0 # or (0,0,0,0) ?
  return w*h #(x, y, w, h)

def overlap(a, b):
    return intersection(a,b) / union(a,b)


def foo(branch_files, cfg, video_names, trajectory_files, cluster_files, tubes_gt, labels, inds_train):
    B_train_pos = []
    B_train_neg = []
    labels_pos = []
    for i, idx in enumerate(inds_train):
        print i, idx, len(inds_train)
        T_obj,T_trj = load_trajectories(trajectory_files[idx])
        C = load_clusters(cluster_files[idx])
        T_obj, T_trj = T_obj[C['tracklet_inliers'],:], T_trj[C['tracklet_inliers'],:]

        T_pos = np.reshape(T_trj,(T_trj.shape[0]*cfg['traj_L'],2))
        f = np.reshape([np.linspace(t-cfg['traj_L']+1, t, cfg['traj_L']).astype('int32') for t in T_obj[:,0]], (T_obj.shape[0]*cfg['traj_L'],1))
        assert T_pos.shape[0] == f.shape[0]
        nids = np.repeat(C['int_paths'],cfg['traj_L'])

        tubes = find_tubes((f,T_pos),nids,C)

        tubes_train, tubes_discarded = sample_training(tubes_gt[video_names[idx]], tubes)
        # visualize2(video_files[idx], (f,T_pos),nids, tubes_gt[video_names[idx]], tubes_train, 'dout-' + video_names[idx] + '.avi', tubes_discarded=tubes_discarded)
        # visualize(video_files[idx], tubes_gt[video_names[idx]], tubes, 'dout-' + video_names[idx] + '.avi')
        print 'done'
        B = load_branches(branch_files[idx], max_depth=cfg['max_depth'])
        B_train_pos += [B[nid] for nid,_ in tubes_train.iteritems()]
        B_train_neg += [B[nid] for nid,_ in tubes_discarded.iteritems()]
        labels_pos += ([labels[idx]]*len(B_train_pos))

    B_train = np.array(B_train_pos + B_train_neg, dtype=np.float32)
    labels_train = np.array(labels_pos + [-1]*len(B_train_neg), dtype=np.int32)

    return B_train, labels_train

def apply_kernel_map(X, map='posneg', axis=1, copy=True):
    if map == 'posneg':
        return posneg(X,axis=axis,copy=copy)
    elif map == 'rootsift':
        return np.sign(X) * np.sqrt(np.abs(X))


def posneg(X, axis=1, copy=True):
    # get the negative part
    X_neg = np.zeros_like(X)
    inds = X < 0
    X_neg[inds] = -X[inds]
    del inds
    # get the positive part
    X_pos = None
    if not copy:
        X[X < 0] = 0
        X_pos = X
    else:
        X_pos = X.copy()
        X_pos[X < 0] = 0

    if len(X_pos.shape) > 1:
        return np.sqrt(np.concatenate([X_pos,X_neg], axis=axis))
    else:
        return np.sqrt(np.concatenate([X_pos,X_neg]))  # simply ignore axis specification

def detection(cfg):
    video_names = [splitext(f)[0] for dp, dn, fn in walk(expanduser(cfg['dataset_path'])) for f in fn if f.endswith(cfg['video_ext'])]
    video_names.sort()

    video_files = [join(cfg['dataset_path'], vn + cfg['video_ext']) for vn in video_names]
    trajectory_files = [join(cfg['trajectories_path'], vn + '.idt') for vn in video_names]
    cluster_files = [join(cfg['clusters_path'], vn + '.pkl') for vn in video_names]

    train_test_split = loadmat(join(cfg['dataset_path'], 'train_test_split.mat'))
    with open(join(cfg['dataset_path'], 'detection_gt.pkl')) as f:
        annotations = cPickle.load(f)  # detection gt

    for pt in cfg['partitions']:
        for f,feat_t in enumerate(cfg['feat_types']):
            try:
                makedirs(cfg['output_kernels_path'])
            except:
                pass

            # Get training and test split from our standard file
            inds_train = np.squeeze(train_test_split['cur_train_indx'][0][pt-1])-1
            inds_test = np.squeeze(train_test_split['cur_test_indx'][0][pt-1])-1

            # Filter out negative examples from test
            if cfg['negative_class'] is not None:
                inds_test = inds_test[np.squeeze(train_test_split['labels2'][inds_test]) != cfg['negative_class']]

             # Encode binary vector labels to integers: [0,0,0,0,1,0] -> 5
            labels = train_test_split['labels2']
            if labels.shape[1] > 1:
                labels = np.argmax(labels,axis=1)
            labels = np.squeeze(labels)

            # auxiliary variable
            labels_train = labels[inds_train]
            labels_test = labels[inds_test]

            branch_path = join(cfg['darws_path'], 'branch_representation-' + str(pt) + '-vd_' + feat_t + '_non-lin_' + cfg['branch_pooling'] + '-pool')

            # List darwin representations/tree-representations paths from disk
            darwin_mat_filenames = [f for f in listdir(branch_path) if isfile(join(branch_path, f)) and splitext(f)[-1] == '.mat']
            darwin_mat_filenames.sort(key=get_id_from_darwinfile)

            # DEBUG (ucf_sports_actions)
            # ---
            # inds_train = np.array([4,5,6, 20,21,22, 38,39,40, 54,55, 62,63,64])
            inds_train = np.array([4,5,6, 20,21,22, 38,39,40, 54,55, 62,63,64])
            inds_test = np.array([0,1, 14,15, 32,33, 52,53, 58,59])
            # ---

            branch_files = [join(branch_path,f) for f in darwin_mat_filenames]

            D_train, labels_train = foo(branch_files, cfg, video_names, trajectory_files, cluster_files, annotations, labels, inds_train)
            D_test, labels_test = foo(branch_files, cfg, video_names, trajectory_files, cluster_files, annotations, labels, inds_test)

            X_train = preprocessing.normalize(apply_kernel_map(D_train, map=cfg['kernel_map'], copy=False), norm=cfg['norm'], axis=1, copy=False)
            X_test = preprocessing.normalize(apply_kernel_map(D_test, map=cfg['kernel_map'], copy=False), norm=cfg['norm'], axis=1, copy=False)

            ovr = multiclass.OneVsRestClassifier(svm.SVC(kernel='linear', class_weight='balanced', C=100, random_state=42))
            ovr.fit(X_train, labels_train)
            acc = metrics.accuracy_score(labels_test, ovr.predict(X_test))
            print 'acc:', acc



            print 'done_all'