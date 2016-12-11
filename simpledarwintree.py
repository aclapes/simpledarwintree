from scipy.io import loadmat
from os import listdir,makedirs,remove
from os.path import isfile,join,splitext,exists,basename
import numpy as np
from sklearn import preprocessing, svm, multiclass, metrics, cross_validation
import itertools
from joblib import delayed, Parallel
import cPickle
import sys
import random
import threading
from copy import deepcopy
import svmutil
import h5py

import pycuda.gpuarray as gpuarray
# import pycuda.autoinit
from pycuda.elementwise import ElementwiseKernel

from sklearn.metrics import confusion_matrix

import skcuda.misc as misc
import pycuda.driver as driver
import skcuda.linalg as linalg
import pycuda.cumath as cumath

import time

# ------------------------------------------------------------------------
# Auxiliary
# ------------------------------------------------------------------------
identity_kernel = ElementwiseKernel(
    "float *y, float *x",
    "y[i] = x[i]",
    "identity_kernel"
)

rootsift_kernel = ElementwiseKernel(
    "float *y, float *x",
    "y[i] = (x[i] >= 0 ? 1:-1)*sqrt(fabs(x[i]))",
    "rootsift_kernel"
)

posneg_complex_kernel = ElementwiseKernel(
    "pycuda::complex<float> *z, float *x",
    "(x[i] > 0) ? z[i] = pycuda::complex<float>(sqrt(x[i]),0) : z[i] = pycuda::complex<float>(0,sqrt(-x[i]))",
    "posneg_complex_kernel",
    preamble="#include <pycuda-complex.hpp>",
)

class ProgressLock(object):
    def __init__(self, n):
        self._lock = threading.Lock()
        self.i = 0
        self.n = n
    def acquire(self):
        # print >>sys.stderr, "acquired", self
        self._lock.acquire()
    def release(self):
        # print >>sys.stderr, "released", self
        self._lock.release()
    def __enter__(self):
        self.acquire()
    def __exit__(self, type, value, traceback):
        self.release()
    def update(self):
        self.i += 1
    def progress(self):
        return float(self.i)/self.n


def get_id_from_darwinfile(filepath):
    stem = splitext(basename(filepath))[0]  # '/home/aclapes/1-Cval1.000000-Gap1-Max-10000.mat' -> '1-Cval1.000000-Gap1-Max-10000'
    return (str.split(stem, '-')[0]).zfill(5)  # '1-Cval1.000000-Gap1-Max-10000' -> '00001'


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


def atep(trees_i, trees_j, is_train=False, kernel_map=None, norm=None, nt=1, use_gpu=None, verbose=False):
    # List all the kernel points to distribute over "nt" threads
    points = []
    if is_train:
        points += [(i,i) for i in xrange(len(trees_i))]  # diagonal
        points += [p for p in itertools.combinations(np.arange(len(trees_i)),2)]  # upper-triangle combinations
    else:
        points += [ p for p in itertools.product(*[np.arange(len(trees_i)),np.arange(len(trees_j))]) ]

    # Complete the kernel from multiple threads' results
    K = np.zeros((len(trees_i),len(trees_j)), dtype=np.float32)
    K[:] = np.nan

    if use_gpu:
        points, values = _atep_gpu(trees_i, trees_j, points, kernel_map=kernel_map, norm=norm, verbose=verbose)
        for k,(i,j) in enumerate(points):
            K[i,j] = values[k]
    # if use_gpu is not None:
    #     random.shuffle(points)  # balances computation among threads
    #     nt = len(use_gpu)
    #     n = int(len(points)/float(nt))+1  # sets the num of points per thread
    #     ret = Parallel(n_jobs=nt, backend='threading')(delayed(_atep_gpu)(trees_i, trees_j, \
    #                                                               points[t*n:((t+1)*n if (t+1)*n < len(points) else len(points))],\
    #                                                               kernel_map=kernel_map, norm=norm, gpu_id=use_gpu[t],
    #                                                               verbose=(verbose if t == 0 else False))
    #                                                for t in xrange(nt))
    #     for points, values in ret:
    #         for k,(i,j) in enumerate(points):
    #             K[i,j] = values[k]
    else:
        random.shuffle(points)  # balances computation among threads
        n = int(len(points)/float(nt))+1  # sets the num of points per thread
        lock = ProgressLock(len(points))
        ret = Parallel(n_jobs=nt, backend='threading')(delayed(_atep)(trees_i, trees_j, \
                                                                  points[t*n:((t+1)*n if (t+1)*n < len(points) else len(points))],\
                                                                  kernel_map=kernel_map, norm=norm, lock=lock,
                                                                  verbose=(verbose if t == 0 else False))
                                                   for t in xrange(nt))
        for points, values in ret:
            for k,(i,j) in enumerate(points):
                K[i,j] = values[k]

    if is_train:
        K = np.triu(K,1).T + np.triu(K)

    return K


def _atep(trees_i, trees_j, points, kernel_map=None, norm=None, lock=None, verbose=False):
    if verbose:
        print 'ATEP computation initiated.'

    res = np.zeros((len(points),), dtype=np.float32)

    points = sorted(points)

    last_i = -1
    for k,(i,j) in enumerate(points):
        if last_i < 0 or last_i != i:
            if kernel_map is None and norm is None:
                ti = trees_i[i]
            else:
                ti = preprocessing.normalize(apply_kernel_map(trees_i[i], map=kernel_map, copy=False), norm=norm, axis=1, copy=False)
        if kernel_map is None and norm is None:
            tj = trees_j[j]
        else:
            tj = preprocessing.normalize(apply_kernel_map(trees_j[j], map=kernel_map, copy=False), norm=norm, axis=1, copy=False)

        pair_leafs_dists = np.dot(ti,tj.T)
        res[k] = np.sum(pair_leafs_dists) / np.prod(pair_leafs_dists.shape)
        last_i = i

        lock.acquire()
        lock.update()  # update progress tracking variable
        if verbose:
            print('Progress: %.2f%%' % (lock.progress()*100.0))
        lock.release()

    return points, res

def _atep_gpu(trees_i, trees_j, points, kernel_map=None, norm=None, verbose=False):
    if verbose:
        print 'ATEP computation initiated.'

    if kernel_map == 'rootsift':
        res = np.zeros((len(points),), dtype=np.float32)
    elif kernel_map == 'posneg':
        res = np.zeros((len(points),), dtype=np.complex64)

    last_i = -1
    for k,(i,j) in enumerate(points):
        if last_i < 0 or last_i != i:
            ti_gpu = gpuarray.to_gpu(trees_i[i][:,1:])
            wi = 2**np.floor(np.log2(trees_i[i][:,0]))

            if kernel_map == 'rootsift':
                ti_m_gpu = gpuarray.empty_like(ti_gpu)
                rootsift_kernel(ti_m_gpu,ti_gpu)
                norm = np.sqrt(misc.sum(linalg.multiply(ti_m_gpu,ti_m_gpu), axis=1, keepdims=True).get())
                norm[norm == 0] = 1
            elif kernel_map == 'posneg':
                ti_m_gpu = gpuarray.empty_like(ti_gpu).astype(np.complex64)
                posneg_complex_kernel(ti_m_gpu,ti_gpu)
                norm = np.sqrt(misc.sum(linalg.multiply(ti_m_gpu,linalg.conj(ti_m_gpu)), axis=1, keepdims=True).get())
                norm[norm == 0+0j] = 1+0j

            ti_mn_gpu = misc.div_matvec(ti_m_gpu, gpuarray.to_gpu(norm), axis=0)

        tj_gpu = gpuarray.to_gpu(trees_j[j][:,1:])
        wj = 2**np.floor(np.log2(trees_j[j][:,0]))

        if kernel_map == 'rootsift':
            tj_m_gpu = gpuarray.empty_like(tj_gpu)
            rootsift_kernel(tj_m_gpu,tj_gpu)
            norm = np.sqrt(misc.sum(linalg.multiply(tj_m_gpu,tj_m_gpu), axis=1, keepdims=True).get())
            norm[norm == 0] = 1
        elif kernel_map == 'posneg':
            tj_m_gpu = gpuarray.empty_like(tj_gpu).astype(np.complex64)
            posneg_complex_kernel(tj_m_gpu,tj_gpu)
            norm = np.sqrt(misc.sum(linalg.multiply(tj_m_gpu,linalg.conj(tj_m_gpu)), axis=1, keepdims=True).get())
            norm[norm == 0+0j] = 1+0j

        tj_mn_gpu = misc.div_matvec(tj_m_gpu, gpuarray.to_gpu(norm), axis=0)

        # W_gpu =  1/cumath.sqrt(gpuarray.to_gpu(np.outer(wi,wj)))
        # res_aux = linalg.multiply(linalg.dot(ti_mn_gpu,tj_mn_gpu,transb='T'),W_gpu)
        res_aux = linalg.dot(ti_mn_gpu,tj_mn_gpu,transb='T')
        res[k] = misc.sum(res_aux).get() / np.prod([ti_mn_gpu.shape[0],tj_mn_gpu.shape[0]])
        print res[k], ' ',
        last_i = i

        if verbose:
            print('Progress: %.2f%%' % ((float(k)/len(points))*100.0))

    return points, res


def train_and_classify(kernels, labels, metric='acc', C=[100]):
    if metric == 'acc':
        train_and_classify_acc(kernels, labels, C=C)
    elif metric == 'map':
        train_and_classify_map(kernels, labels, C=C)


def random_point_distribution(n,m,random_seed=None):
    """
    Generate n m-dimensional points
    :param n: the number of points
    :param m: the number of dimensions for each point
    :return:
    """
    random.seed(random_seed)

    X = np.zeros((n,m), dtype=np.float32)
    for i in xrange(n):
        x = [random.random() for j in xrange(m)]
        X[i,:] = x / np.sum(x)

    return X

def generate_weights(n, m=9):
    weights = []
    for w in itertools.product(*([np.linspace(0.1,0.9,m)]*n)):
        if np.sum(w) == 1:
            weights.append(w)

    return weights


def train_and_classify_acc(kernels, labels, C=[100]):
    # Fuse different modalities by averaging
    for i,representations in enumerate(kernels):
        if not isinstance(representations, tuple):
            a = [1 for (K,_) in representations.values()]  #[np.max(np.diag(K)) for (K,_) in representations.values()]
            kernels[i] = (np.sum([(1.0/len(representations))*(K/a[k]) for k,(K,_) in enumerate(representations.values())],axis=0),
                          np.sum([(1.0/len(representations))*(K/a[k]) for k,(_,K) in enumerate(representations.values())],axis=0))

    # Validate the weight of the different mid-level representations
    if len(kernels) == 1:
        W = np.array([[1.0]])
    elif len(kernels) == 2:
        lin = np.linspace(0.1,0.9,17)
        W = np.vstack((lin, 1-lin)).T
        # W = np.array([[0.3,.7]])
    else:
        W = np.array(generate_weights(len(kernels),17))
        # W = np.array([[1.0/len(kernels)]*len(kernels)])  # equal weight

    labels_train, labels_test = labels  # assume tree_kernel labels coincide! assertion needed

    C_perfs = np.zeros((W.shape[0],len(C)), dtype=np.float32)
    for i in xrange(W.shape[0]):
        w = W[i,:]
        for j,c in enumerate(C):
            cv = cross_validation.StratifiedKFold(labels_train, n_folds=4, random_state=0)
            cv_val_perfs = []
            for v,(val_tr,val_te) in enumerate(cv):
                a = [np.max(np.diag(K[val_tr,:][:,val_tr])) for _,(K,_) in enumerate(kernels)]
                # a = [1 for _,(K,_) in enumerate(kernels)]
                K_train_val = np.sum([w[k]*(K[val_tr,:][:,val_tr]/a[k]) for k,(K,_) in enumerate(kernels)], axis=0)
                K_test_val = np.sum([w[k]*(K[val_te,:][:,val_tr]/a[k]) for k,(K,_) in enumerate(kernels)], axis=0)

                ovr = multiclass.OneVsRestClassifier(svm.SVC(kernel='precomputed', class_weight='balanced', C=c, random_state=42))
                ovr.fit(K_train_val, labels_train[val_tr])
                acc = metrics.accuracy_score(labels_train[val_te], ovr.predict(K_test_val))
                cv_val_perfs.append(acc)
            C_perfs[i,j] = np.mean(cv_val_perfs)

    best_idx = np.unravel_index(np.argmax(C_perfs), C_perfs.shape)
    w = W[best_idx[0],:]
    c = C[best_idx[1]]
    print 'Validation weight: {0}, c: {1}, acc: {2:2.2f}'.format(w,c,np.max(C_perfs)*100.0)

    ovr = multiclass.OneVsRestClassifier(svm.SVC(kernel='precomputed', class_weight='balanced', C=c, random_state=42))

    a = [np.max(np.diag(K)) for _,(K,_) in enumerate(kernels)]
    # a = [1 for _,(K,_) in enumerate(kernels)]
    K_train = np.sum([w[k]*(K/a[k]) for k,(K,_) in enumerate(kernels)], axis=0)
    ovr.fit(K_train, labels_train)

    K_test  = np.sum([w[k]*(K/a[k]) for k,(_,K) in enumerate(kernels)], axis=0)

    preds = ovr.predict(K_test)

    binary_accs = []
    unique_preds = np.unique(preds)
    for u in unique_preds:
        acc = metrics.accuracy_score(labels_test == u, preds == u)
        print '{0:2.4f} '.format(acc)
    print


    print 'y_test: ', ",".join(np.char.mod('%d', labels_test))
    print 'y_pred: ', ",".join(np.char.mod('%d', preds))

    print 'Out-of-sample acc: {0:2.2f}'.format(metrics.accuracy_score(labels_test, preds)*100.0)

# def train_and_classify_acc(kernels, labels, C=[100]):
#     # Fuse different modalities by averaging
#     for i,representations in enumerate(kernels):
#         if not isinstance(representations, tuple):
#             a = [np.max(np.diag(K)) for (K,_) in representations.values()]
#             kernels[i] = (np.sum([(1.0/len(representations))*(K/a[k]) for k,(K,_) in enumerate(representations.values())],axis=0),
#                           np.sum([(1.0/len(representations))*(K/a[k]) for k,(_,K) in enumerate(representations.values())],axis=0))
#
#     labels_train, labels_test = labels  # assume tree_kernel labels coincide! assertion needed
#
#     C_perfs = np.zeros((len(kernels),len(C)), dtype=np.float32)
#     for i,(K_train,K_test) in enumerate(kernels):
#         for j,c in enumerate(C):
#             cv = cross_validation.StratifiedKFold(labels_train, n_folds=4, random_state=42)
#             cv_val_perfs = []
#             for v,(val_tr,val_te) in enumerate(cv):
#                 ovr = multiclass.OneVsRestClassifier(svm.SVC(kernel='precomputed', class_weight='balanced', C=c, random_state=42))
#                 ovr.fit(K_train[val_tr,:][:,val_tr], labels_train[val_tr])
#                 acc = metrics.accuracy_score(labels_train[val_te], ovr.predict(K_train[val_te,:][:,val_tr]))
#                 cv_val_perfs.append(acc)
#             C_perfs[i,j] = np.mean(cv_val_perfs)
#
#     best_idx = np.unravel_index(np.argmax(C_perfs), C_perfs.shape)
#     w = C_perfs[best_idx[0],:]
#     c = C[best_idx[1]]
#     print 'Validation acc:', w,c
#
#     P = [None] * len(kernels)
#     for i,(K_train,K_test) in enumerate(kernels):
#         ovr = multiclass.OneVsRestClassifier(svm.SVC(kernel='precomputed', class_weight='balanced', C=c, random_state=42,probability=True))
#         ovr.fit(K_train, labels_train)
#         P[i] = ovr.predict_proba(K_test)
#
#     # w = np.array(C_perfs)/np.sum(C_perfs)
#
#     preds = np.argmax(np.sum(P,axis=0),axis=1)
#
#     print 'Out-of-sample acc:', metrics.accuracy_score(labels_test, preds)


def train_and_classify_map(kernels, labels, C=[100]):
    _kernels = deepcopy(kernels)
    # Fuse different modalities by averaging
    for i,representations in enumerate(_kernels):
        if not isinstance(representations, tuple):
            a = [np.max(np.diag(K)) for (K,_) in representations.values()]
            _kernels[i] = (np.sum([(1.0/len(representations))*(K/a[k]) for k,(K,_) in enumerate(representations.values())],axis=0),
                          np.sum([(1.0/len(representations))*(K/a[k]) for k,(_,K) in enumerate(representations.values())],axis=0))

    # Validate the weight of the different mid-level representations
    if len(_kernels) == 1:
        W = np.array([[1.0]])
    elif len(_kernels) == 2:
        lin = np.linspace(0.1,0.9,7)
        W = np.vstack((lin, 1-lin)).T
    else:
        W = np.array(generate_weights(len(_kernels),17))
        # W = np.array([[1.0/len(kernels)]*len(kernels)])  # equal weight

    labels_train, labels_test = labels  # assume tree_kernel labels coincide! assertion needed

    unique_classes = np.unique(labels_test)
    ap_classes = [None] * len(unique_classes)
    ap_val_classes = [None] * len(unique_classes)

    for l,cl in enumerate(unique_classes):
        C_perfs = np.zeros((W.shape[0],len(C)), dtype=np.float32)
        for i in xrange(W.shape[0]):
            w = W[i,:]
            # K_train = np.sum([w[k]*(K/np.max(np.diag(K))) for k,(K,_) in enumerate(_kernels)], axis=0)
            # K_test  = np.sum([w[k]*K for k,(_,K) in enumerate(_kernels)], axis=0)
            for j,c in enumerate(C):
                cv = cross_validation.StratifiedKFold(labels_train == cl, n_folds=10, random_state=42)
                cv_perfs = []
                for v,(val_tr,val_te) in enumerate(cv):
                    # a = [np.max(np.diag(K[val_tr,:][:,val_tr])) for _,(K,_) in enumerate(_kernels)]
                    a = [1 for _,(K,_) in enumerate(kernels)]
                    K_train_val = np.sum([w[k]*(K[val_tr,:][:,val_tr]/a[k]) for k,(K,_) in enumerate(_kernels)], axis=0)
                    K_test_val = np.sum([w[k]*(K[val_te,:][:,val_tr]/a[k]) for k,(K,_) in enumerate(_kernels)], axis=0)

                    model = svm.SVC(kernel='precomputed', class_weight='balanced', C=c, random_state=42)
                    model.fit(K_train_val, labels_train[val_tr] == cl)
                    map = metrics.average_precision_score(labels_train[val_te] == cl,
                                                          model.decision_function(K_test_val))
                    cv_perfs.append(map)
                C_perfs[i,j] = np.mean(cv_perfs)

        best_idx = np.unravel_index(np.argmax(C_perfs), C_perfs.shape)
        w = W[best_idx[0],:]
        c = C[best_idx[1]]
        # print w,c,np.max(C_perfs)
        ap_val_classes[l] = np.nanmax(C_perfs)

        model = svm.SVC(kernel='precomputed', class_weight='balanced', C=c, random_state=42)

        # a = [np.max(np.diag(K)) for _,(K,_) in enumerate(_kernels)]
        a = [1 for _,(K,_) in enumerate(_kernels)]
        K_train = np.sum([w[k]*(K/a[k]) for k,(K,_) in enumerate(_kernels)], axis=0)
        model.fit(K_train, labels_train == cl)

        K_test  = np.sum([w[k]*(K/a[k]) for k,(_,K) in enumerate(_kernels)], axis=0)

        print cl
        print labels_test
        y_test = (labels_test == cl)
        y_pred = model.decision_function(K_test)
        print y_test.astype('int')
        print (y_pred >= 0).astype('int')

        ap_classes[l] = metrics.average_precision_score(y_test, y_pred)

    print ap_val_classes
    print 'Validation mAP: {0:2.2f}'.format(np.mean(ap_val_classes)*100.)
    print ap_classes
    print 'Out-of-sample mAP: {0:2.2f}'.format(np.mean(ap_classes)*100.)

def load_darwins(indices, representations, max_depth=4):
    # TODO: this is a nasty fix. find a better solution for polimorphism
    # ---
    if isinstance(representations[0],basestring):
        representations = [representations]
    # ---

    all = [None] * len(indices)

    for i,idx in enumerate(indices):
        print('%d/%d' % (i+1, len(indices)))
        aux = []
        nodeid = None
        for j in xrange(len(representations)):
            try:
                mat = loadmat(representations[j][idx])  # raise not implemented ee
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

        print aux.shape

        if nodeid is None:
            all[i] = aux
        else:
            nodeid = nodeid[nodeid < 2**max_depth]
            all[i] = np.hstack((nodeid[:,np.newaxis].astype(np.float32), aux))

    return np.array(all)


# def load_representations(representations, indices):
#         all = [[] for i in xrange(len(indices))]
#         for i in indices:
#             # print ('[%d/%d] Loading trees..' % (i+1,len(darws)))
#             # print darws[i]
#             tree_mat = loadmat(join(tree_path, darws[i]))
#             # Wtree = np.hstack( (tree_mat['Wtree'], np.zeros((tree_mat['Wtree'].shape[0],1))) )
#             # all_trees[i] = preprocessing.normalize(apply_kernel_map(tree_mat['Wtree'],map=cfg['kernel_map']), norm=cfg['norm'], axis=1)
#             nodeid = tree_mat['nodeid'][tree_mat['nodeid'] > 1]
#             if cfg['maximum_depth'] is None:
#                 tree = tree_mat['Wtree']
#             else:
#                 tree = tree_mat['Wtree'][nodeid < 2**cfg['maximum_depth'],:]
#             # tree = tree_mat['Wtree']
#             if not cfg['pre_processing']:
#                 all_trees[i] = tree
#             else:
#                 all_trees[i] = preprocessing.normalize(apply_kernel_map(tree,map=cfg['kernel_map'],copy=False), norm=cfg['norm'], axis=1, copy=False)
#         all_trees = np.array(all_trees)

def distributed_atep(dw_paths, inds, chunks='all', inds_train=None, max_depth=64,
                     kernel_map='rootsift', norm='l2', use_gpu=None, verbose=False):
    if inds_train is None:
        return distibuted_atep_train(dw_paths, inds, chunks,max_depth=max_depth,
                                     kernel_map=kernel_map, norm=norm, use_gpu=use_gpu, verbose=verbose)
    else:
        return distibuted_atep_test(dw_paths, inds, inds_train, max_depth=max_depth,
                                    kernel_map=kernel_map, norm=norm, use_gpu=use_gpu, verbose=verbose)


def distibuted_atep_train(dw_paths, inds, chunks='all', max_depth=64, kernel_map='rootsift', norm='l2', use_gpu=None, intermediates_path=None, verbose=False):
    n = len(inds)

    x_inds = range(n/4)
    y_inds = range(n/4,n/2)
    z_inds = range(n/2,3*n/4)
    w_inds = range(3*n/4,n)

    xy_inds = np.concatenate([x_inds,y_inds])
    zw_inds = np.concatenate([z_inds,w_inds])

    st_time = time.time()
    xy = load_darwins( inds[xy_inds], dw_paths, max_depth=max_depth )
    A = atep(xy, xy, is_train=True, kernel_map=kernel_map, norm=norm, use_gpu=use_gpu, verbose=verbose)
    if intermediates_path:
        with open(join(intermediates_path, 'A.pkl'), 'wb') as f:
            cPickle.dump(A,f)
    print('[Distributed kernel (train)] Big time %.2f secs.' % (time.time() - st_time))
    x = xy[:xy.shape[0]/2]

    st_time = time.time()
    z = load_darwins( inds[z_inds], dw_paths, max_depth=max_depth )
    C = atep(x, z, is_train=False, kernel_map=kernel_map, norm=norm, use_gpu=use_gpu, verbose=verbose)
    del z
    if intermediates_path:
        with open(join(intermediates_path, 'C.pkl'), 'wb') as f:
            cPickle.dump(C,f)
    print('[Distributed kernel (train)] Little time %.2f secs.' % (time.time() - st_time))

    w = load_darwins( inds[w_inds], dw_paths, max_depth=max_depth )
    D = atep(x, w, is_train=False, kernel_map=kernel_map, norm=norm, use_gpu=use_gpu, verbose=verbose)
    if intermediates_path:
        with open(join(intermediates_path, 'D.pkl'), 'wb') as f:
            cPickle.dump(D,f)
    del x

    y = load_darwins( inds[y_inds], dw_paths, max_depth=max_depth )
    E = atep(y, w, is_train=False, kernel_map=kernel_map, norm=norm, use_gpu=use_gpu, verbose=verbose)
    if intermediates_path:
        with open(join(intermediates_path, 'E.pkl'), 'wb') as f:
            cPickle.dump(E,f)

    z = load_darwins( inds[z_inds], dw_paths, max_depth=max_depth )
    F = atep(y, z, is_train=False, kernel_map=kernel_map, norm=norm, use_gpu=use_gpu, verbose=verbose)
    if intermediates_path:
        with open(join(intermediates_path, 'F.pkl'), 'wb') as f:
            cPickle.dump(F,f)
    del y

    zw = np.concatenate((z,w))
    B = atep(zw, zw, is_train=True, kernel_map=kernel_map, norm=norm, use_gpu=use_gpu, verbose=verbose)
    if intermediates_path:
        with open(join(intermediates_path, 'B.pkl'), 'wb') as f:
            cPickle.dump(B,f)
    del zw

    K = np.zeros((n,n), dtype=np.float32)
    K[np.ix_(xy_inds,xy_inds)] = A  # copy A to the A's corresponding subset of K
    K[np.ix_(x_inds,z_inds)] = C
    K[np.ix_(x_inds,w_inds)] = D
    K[np.ix_(y_inds,w_inds)] = E
    K[np.ix_(y_inds,z_inds)] = F
    K[np.ix_(zw_inds,zw_inds)] = B

    K = np.triu(K,1).T + np.triu(K)

    return K


# def distibuted_atep_test(dw_paths, inds_te, inds_tr, max_depth=64, kernel_map='rootsift', norm='l2', use_gpu=True, intermediates_path=None, verbose=False):
#     n = len(inds_te)
#     m = len(inds_tr)
#
#     xte_inds, xtr_inds = range(n/2), range(m/2)
#     yte_inds, ytr_inds = range(n/2,n), range(m/2,m)
#
#     st_time = time.time()
#     xte = load_darwins(inds_te[xte_inds], dw_paths, max_depth=max_depth)
#     xtr = load_darwins(inds_tr[xtr_inds], dw_paths, max_depth=max_depth)
#     A = atep(xte, xtr, is_train=False, kernel_map=kernel_map, norm=norm, use_gpu=use_gpu, verbose=verbose)
#     del xte
#     if intermediates_path:
#         with open(join(intermediates_path, 'A.pkl'), 'wb') as f:
#             cPickle.dump(A,f)
#     print('[Distributed kernel (test)] Time %.2f secs.' % (time.time() - st_time))
#
#     yte = load_darwins(inds_te[yte_inds], dw_paths, max_depth=max_depth)
#     B = atep(yte, xtr, is_train=False, kernel_map=kernel_map, norm=norm, use_gpu=use_gpu, verbose=verbose)
#     del xtr
#     if intermediates_path:
#         with open(join(intermediates_path, 'B.pkl'), 'wb') as f:
#             cPickle.dump(B,f)
#
#     ytr = load_darwins(inds_tr[ytr_inds], dw_paths, max_depth=max_depth)
#     C = atep(yte, ytr, is_train=False, kernel_map=kernel_map, norm=norm, use_gpu=use_gpu, verbose=verbose)
#     del yte
#     if intermediates_path:
#         with open(join(intermediates_path, 'C.pkl'), 'wb') as f:
#             cPickle.dump(C,f)
#
#     xte = load_darwins(inds_te[xte_inds], dw_paths, max_depth=max_depth)
#     D = atep(xte, ytr, is_train=False, kernel_map=kernel_map, norm=norm, use_gpu=use_gpu, verbose=verbose)
#     if intermediates_path:
#         with open(join(intermediates_path, 'D.pkl'), 'wb') as f:
#             cPickle.dump(D,f)
#
#     K = np.zeros((n,m), dtype=np.float32)
#     K[np.ix_(xte_inds,xtr_inds)] = A
#     K[np.ix_(yte_inds,xtr_inds)] = B
#     K[np.ix_(yte_inds,ytr_inds)] = C
#     K[np.ix_(xte_inds,ytr_inds)] = D
#
#     return K


def distibuted_atep_test(dw_paths, inds_te, inds_tr, max_depth=64, kernel_map='rootsift', norm='l2', use_gpu=True, intermediates_path=None, verbose=False):
    if intermediates_path is None:
        intermediates_path = '.'

    n = len(inds_te)
    m = len(inds_tr)

    xte_inds, xtr_inds = range(n/2), range(m/2)
    yte_inds, ytr_inds = range(n/2,n), range(m/2,m)

    # st_time = time.time()
    # xte = load_darwins(inds_te[xte_inds], dw_paths, max_depth=max_depth)
    # xtr = load_darwins(inds_tr[xtr_inds], dw_paths, max_depth=max_depth)
    # A = atep(xte, xtr, is_train=False, kernel_map=kernel_map, norm=norm, use_gpu=use_gpu, verbose=verbose)
    # if intermediates_path:
    #     with open(join(intermediates_path, 'A.pkl'), 'wb') as f:
    #         cPickle.dump(A,f)
    # print('[Distributed kernel (test)] Time %.2f secs.' % (time.time() - st_time))

    # yte = load_darwins(inds_te[yte_inds], dw_paths, max_depth=max_depth)
    # xtr = load_darwins(inds_tr[xtr_inds], dw_paths, max_depth=max_depth)
    # B = atep(yte, xtr, is_train=False, kernel_map=kernel_map, norm=norm, use_gpu=use_gpu, verbose=verbose)
    # if intermediates_path:
    #     with open(join(intermediates_path, 'B.pkl'), 'wb') as f:
    #         cPickle.dump(B,f)

    # yte = load_darwins(inds_te[yte_inds], dw_paths, max_depth=max_depth)
    # ytr = load_darwins(inds_tr[ytr_inds], dw_paths, max_depth=max_depth)
    # C = atep(yte, ytr, is_train=False, kernel_map=kernel_map, norm=norm, use_gpu=use_gpu, verbose=verbose)
    # if intermediates_path:
    #     with open(join(intermediates_path, 'C.pkl'), 'wb') as f:
    #         cPickle.dump(C,f)

    # xte = load_darwins(inds_te[xte_inds], dw_paths, max_depth=max_depth)
    # ytr = load_darwins(inds_tr[ytr_inds], dw_paths, max_depth=max_depth)
    # D = atep(xte, ytr, is_train=False, kernel_map=kernel_map, norm=norm, use_gpu=use_gpu, verbose=verbose)
    # if intermediates_path:
    #     with open(join(intermediates_path, 'D.pkl'), 'wb') as f:
    #         cPickle.dump(D,f)

    with open(join(intermediates_path, 'A.pkl'), 'rb') as f:
        A = cPickle.load(f)
    with open(join(intermediates_path, 'B.pkl'), 'rb') as f:
        B= cPickle.load(f)
    with open(join(intermediates_path, 'C.pkl'), 'rb') as f:
        C = cPickle.load(f)
    with open(join(intermediates_path, 'D.pkl'), 'rb') as f:
        D = cPickle.load(f)

    K = np.zeros((n,m), dtype=np.float32)
    K[np.ix_(xte_inds,xtr_inds)] = A
    K[np.ix_(yte_inds,xtr_inds)] = B
    K[np.ix_(yte_inds,ytr_inds)] = C
    K[np.ix_(xte_inds,ytr_inds)] = D

    return K


# ------------------------------------------------------------------------
# main function
# ------------------------------------------------------------------------

def simpledarwintree(cfg):
    dev = driver.Device(cfg['gpu_id'])
    ctx = dev.make_context()
    misc.init()
    linalg.init()

    train_test_split = loadmat(join(cfg['dataset_path'], 'train_test_split.mat'))

    for pt in cfg['partitions']:

        root_kernels = {}
        tree_kernels = {}
        branch_kernels = {}
        fusion_kernels = {}

        P = [None] * len(cfg['feat_types'])
        for f,feat_t in enumerate(cfg['feat_types']):
            try:
                makedirs(cfg['output_kernels_path'])
            except:
                pass

            vd_path = join(cfg['darws_path'], 'representation-' + str(pt) + '-vd_' + feat_t + '_non-lin')
            tree_path = join(cfg['darws_path'], 'tree_representation-' + str(pt) + '-vd_' + feat_t + '_non-lin')
            branch_path = join(cfg['darws_path'], 'branch_representation-' + str(pt) + '-vd_' + feat_t + '_non-lin_' + cfg['branch_pooling'] + '-pool')

            root_kernel_name = 'kernel-' + str(pt) + '-vd_' + feat_t + '_non-lin_' + cfg['kernel_map']
            tree_kernel_name = 'tree_kernel-' + str(pt) + '-vd_' + feat_t + '_non-lin_' + cfg['kernel_map']
            branch_kernel_name = 'branch_kernel-' + str(pt) + '-vd_' + feat_t + '_non-lin_' + cfg['branch_pooling'] + '-pool_' + cfg['kernel_map']
            fusion_kernel_name = 'fusion_kernel-' + str(pt) + '-vd_' + feat_t + '_non-lin_' + cfg['branch_pooling'] + '-pool_' + cfg['kernel_map']

            root_kernel_filepath = join(cfg['output_kernels_path'], root_kernel_name + '.pkl')
            tree_kernel_filepath = join(cfg['output_kernels_path'], tree_kernel_name + '.pkl')
            branch_kernel_filepath = join(cfg['output_kernels_path'], branch_kernel_name + '.pkl')  # TODO: remove "test"
            fusion_kernel_filepath = join(cfg['output_kernels_path'], fusion_kernel_name + '.pkl')  # TODO: remove "test"

            # List darwin representations/tree-representations paths from disk
            darwin_mat_filenames = [f for f in listdir(branch_path) if isfile(join(branch_path, f)) and splitext(f)[-1] == '.mat']
            darwin_mat_filenames.sort(key=get_id_from_darwinfile)

            root_dw_paths = [join(vd_path,f) for f in darwin_mat_filenames]
            tree_dw_paths = [join(tree_path,f) for f in darwin_mat_filenames]
            branch_dw_paths = [join(branch_path,f) for f in darwin_mat_filenames]


            # if not exists(branch_kernel_filepath):
            #     all_branches = [[]
            #     for i in xrange(len(darws))]
            #         # st_time = time.time()
            #         branch_mat = loadmat(join(branch_path, darws[i]))
            #         # print (time.time() - st_time),
            #         # with open('tmp.pkl', 'wb') as f:
            #         #     cPickle.dump(root_mat,f)
            #         # st_time = time.time()
            #         # with open('tmp.pkl', 'rb') as f:
            #         #     branch_pkl = cPickle.load(f)
            #         # print (time.time() - st_time)
            #
            #         if not cfg['pre_processing']:
            #             all_branches[i] = branch_mat['Btree']
            #         else:
            #             all_branches[i] = preprocessing.normalize(apply_kernel_map(branch_mat['Btree'],map=cfg['kernel_map'],copy=False), norm=cfg['norm'], axis=1, copy=False)
            #     all_branches = np.array(all_branches)

            labels = train_test_split['labels2']

            # Get training and test split from our standard file
            inds_train = np.squeeze(train_test_split['cur_train_indx'][0][pt-1])-1
            inds_test = np.squeeze(train_test_split['cur_test_indx'][0][pt-1])-1
            # # DEBUG
            # inds_train = np.array([4,5,6, 20,21,22, 38,39,40, 54,55, 62,63,64])
            # inds_train = np.array([4,5,6,7,8, 20,21,22,23,24, 38,39,40,41,42, 54,55,56,57, 62,63,64,65,66])
            # inds_test = np.array([0,1, 14,15, 32,33, 52,53, 58,59])


            # Filter out negative examples from test
            if cfg['negative_class'] is not None:
                inds_test = inds_test[np.squeeze(train_test_split['labels2'][inds_test]) != cfg['negative_class']]


            # if all_roots.shape[0] < len(labels):
            #     sys.stderr.write('Representations missing. Quitting.')
            #     quit()
            #
            # if len(all_trees) < len(labels):
            #     sys.stderr.write('Tree representations missing. Quitting.')
            #     quit()
            #
            # if len(all_branches) < len(labels):
            #     sys.stderr.write('Branch representations missing. Quitting.')
            #     quit()

            # Encode binary vector labels to integers ([0,0,0,0,1,0] -> 5)
            if labels.shape[1] > 1:
                labels = np.argmax(labels,axis=1)
            labels = np.squeeze(labels)

            # Compute kernels
            labels_train = labels[inds_train]  # auxiliary variable
            labels_test = labels[inds_test]    #     "        "

            # TESTS
            # -------------------------------------------------------------------------------------
            # -------------------------------------------------------------------------------------
            # K = np.dot(all_roots[inds_train, :], all_roots[inds_train, :].T)
            #
            # clf = svm.SVC(kernel='precomputed', class_weight='balanced', C=100, random_state=42)
            # ovr = multiclass.OneVsRestClassifier(clf)
            # ovr.fit(K, labels_train)
            #
            # Ktr = np.hstack( (np.arange(K.shape[0])[:,np.newaxis]+1, K) )
            # Ktr = [{j:Ktr[i,j] for j in xrange(Ktr.shape[1])}  for i in xrange(Ktr.shape[0])]
            # prob = svmutil.svm_problem(labels_train.tolist(), Ktr, isKernel=True)
            #
            # C = [0.01, 0.1, 1,10,100,1000, 10000]
            # model = [None] * len(C)
            # for i,c in enumerate(C):
            #     model[i] = svmutil.svm_train(prob, '-t 4 -v 4 -q -c ' + str(c))
            # print model
            # best_model_idx = np.argmax(model)
            #
            # prob = svmutil.svm_problem(labels_train.tolist(), Ktr, isKernel=True)
            # model = svmutil.svm_train(prob, '-t 4 -q -c ' + str(C[best_model_idx]))
            #
            # K = np.dot(all_roots[inds_inds, :], all_roots[inds_train, :].T)
            #
            # print metrics.accuracy_score(labels_test, ovr.predict(K))
            #
            # nancol = np.full((K.shape[0],1), np.nan)
            # Kte = np.hstack( (nancol, K) )  # np.hstack( (np.arange(K.shape[0])[:,np.newaxis]+1, K) )
            # Kte = [{j:Kte[i,j] for j in xrange(Kte.shape[1])}  for i in xrange(Kte.shape[0])]
            #
            # [out1,out2,out3] = svmutil.svm_predict(labels_test.tolist(), Kte, model)
            # print out2
            # -------------------------------------------------------------------------------------
            # -------------------------------------------------------------------------------------

            kernel_map = None if cfg['pre_processing'] else cfg['kernel_map']
            norm = None if cfg['pre_processing'] else cfg['norm']

            st_time = time.time()

            # ------
            # ROOT
            # ------
            if 'r' in cfg['midlevels']:
                try:
                    with open(root_kernel_filepath, 'rb') as f:
                        root_pkl = cPickle.load(f)
                except:
                    all_roots_tr = np.vstack(load_darwins(inds_train, root_dw_paths))
                    all_roots_te = np.vstack(load_darwins(inds_test, root_dw_paths))

                    all_roots_tr = apply_kernel_map(all_roots_tr, map=cfg['kernel_map'])
                    all_roots_te = apply_kernel_map(all_roots_te, map=cfg['kernel_map'])
                    all_roots_tr = preprocessing.normalize(all_roots_tr, norm=cfg['norm'], axis=1)
                    all_roots_te = preprocessing.normalize(all_roots_te, norm=cfg['norm'], axis=1)

                    print('Kernel (train)...')
                    K_train = np.dot(all_roots_tr, all_roots_tr.T)
                    print('Kernel (test)...')
                    K_test  = np.dot(all_roots_te, all_roots_tr.T)

                    # print('Saving (kernel train/test)... '),
                    with open(root_kernel_filepath, 'wb') as f:
                        cPickle.dump(dict(kernel_map=kernel_map, norm=norm, \
                                          kernels=(K_train, K_test), labels=(labels_train, labels_test)), f)

                    root_pkl = dict(kernels=(K_train, K_test),labels=(labels_train,labels_test))
                    del all_roots_tr
                    del all_roots_te
                root_kernels[feat_t] = root_pkl['kernels']

            # ------
            # NODES
            # ------
            if 'n' in cfg['midlevels']:
                try:
                    with open(tree_kernel_filepath, 'rb') as f:
                        tree_pkl = cPickle.load(f)
                except IOError:
                    Kn_train = None
                    train = None
                    # TODO: uncomment
                    # ---
                    try:
                        with open(join(cfg['output_kernels_path'], tree_kernel_name + '.train.pkl'), 'rb') as f:
                            Kn_train,_ = cPickle.load(f)['kernels']
                    except IOError:
                        print('Tree kernel (train)...')
                        if cfg['distributed']:
                            Kn_train = distributed_atep(tree_dw_paths, inds_train, max_depth=cfg['max_depth'], use_gpu=True, verbose=True)
                        else:
                            train = load_darwins(inds_train, tree_dw_paths, max_depth=cfg['max_depth'])
                            Kn_train = atep(train, train, is_train=True, kernel_map=cfg['kernel_map'], norm=cfg['norm'], use_gpu=True, verbose=True)
                        print('Saving (kernel train)... '),
                        with open(join(cfg['output_kernels_path'], tree_kernel_name + '.train.pkl'), 'wb') as f:
                            cPickle.dump(dict(kernel_map=kernel_map, norm=norm, \
                                              kernels=(Kn_train,None), labels=(labels_train,labels_test)), f)
                    # ---
                    print('Tree kernel (test)...')
                    if cfg['distributed']:
                        Kn_test = distributed_atep(tree_dw_paths, inds_test, inds_train=inds_train, max_depth=cfg['max_depth'], use_gpu=True, verbose=True)
                    else:
                        if train is None:
                            train = load_darwins(inds_train, tree_dw_paths, max_depth=cfg['max_depth'])
                        test = load_darwins(inds_test, tree_dw_paths, max_depth=cfg['max_depth'])
                        Kn_test = atep(test, train, is_train=False, kernel_map=cfg['kernel_map'], norm=cfg['norm'], use_gpu=True, verbose=True)
                    print('Saving kernels (train,test)... '),
                    with open(tree_kernel_filepath, 'wb') as f:
                        cPickle.dump(dict(kernel_map=kernel_map, norm=norm, \
                                          kernels=(Kn_train,Kn_test), labels=(labels_train,labels_test)), f)
                        # remove(join(cfg['output_kernels_path'], branch_kernel_name + '.train.pkl'))   # TODO: uncomment
                    tree_pkl = dict(kernels=(Kn_train, Kn_test),labels=(labels_train,labels_test))
                tree_kernels[feat_t] = tree_pkl['kernels']


            # ------
            # BRANCH
            # ------
            if 'b' in cfg['midlevels']:
                try:
                    with open(branch_kernel_filepath, 'rb') as f:
                        branch_pkl = cPickle.load(f)
                except IOError:
                    Kb_train = None
                    train = None
                    # TODO: uncomment
                    # ---
                    try:
                        with open(join(cfg['output_kernels_path'], branch_kernel_name + '.train.pkl'), 'rb') as f:
                            Kb_train,_ = cPickle.load(f)['kernels']
                    except IOError:
                        print('Branch kernel (train)...')
                        if cfg['distributed']:
                            Kb_train = distributed_atep(branch_dw_paths, inds_train, max_depth=cfg['max_depth'], use_gpu=True, verbose=True)
                        else:
                            train = load_darwins(inds_train, branch_dw_paths, max_depth=cfg['max_depth'])
                            Kb_train = atep(train, train, is_train=True, kernel_map=cfg['kernel_map'], norm=cfg['norm'], use_gpu=True, verbose=True)
                        print('Saving (kernel train)... '),
                        with open(join(cfg['output_kernels_path'], branch_kernel_name + '.train.pkl'), 'wb') as f:
                            cPickle.dump(dict(kernel_map=kernel_map, norm=norm, \
                                              kernels=(Kb_train,None), labels=(labels_train,labels_test)), f)
                    # ---
                    print('Branch kernel (test)...')
                    if cfg['distributed']:
                        Kb_test = distributed_atep(branch_dw_paths, inds_test, inds_train=inds_train, max_depth=cfg['max_depth'], use_gpu=True, verbose=True)
                    else:
                        if train is None: # if not train already loaded
                            train = load_darwins(inds_train, branch_dw_paths, max_depth=cfg['max_depth'])
                        test = load_darwins(inds_test, branch_dw_paths, max_depth=cfg['max_depth'])
                        Kb_test = atep(test, train, is_train=False, kernel_map=cfg['kernel_map'], norm=cfg['norm'], use_gpu=True, verbose=True)
                    print('Saving kernels (train,test)... '),
                    with open(branch_kernel_filepath, 'wb') as f:
                        cPickle.dump(dict(kernel_map=kernel_map, norm=norm, \
                                          kernels=(Kb_train,Kb_test), labels=(labels_train,labels_test)), f)
                        # remove(join(cfg['output_kernels_path'], branch_kernel_name + '.train.pkl'))   # TODO: uncomment
                    branch_pkl = dict(kernels=(Kb_train, Kb_test),labels=(labels_train,labels_test))
                branch_kernels[feat_t] = branch_pkl['kernels']

            # ------
            # FUSION
            # ------
            if 'nb' in cfg['midlevels']:
                try:
                    with open(fusion_kernel_filepath, 'rb') as f:
                        fusion_pkl = cPickle.load(f)
                except IOError:
                    Kf_train = None
                    train = None
                    # TODO: uncomment
                    # ---
                    try:
                        with open(join(cfg['output_kernels_path'], fusion_kernel_name + '.train.pkl'), 'rb') as f:
                            Kf_train,_ = cPickle.load(f)['kernels']
                    except IOError:
                        print('Fusion kernel (train)...')
                        if cfg['distributed']:
                            Kf_train = distributed_atep([tree_dw_paths, branch_dw_paths], inds_train, max_depth=cfg['max_depth'], use_gpu=True, verbose=True)
                        else:
                            train = load_darwins(inds_train, [tree_dw_paths, branch_dw_paths], max_depth=cfg['max_depth'])
                            Kf_train = atep(train, train, is_train=True, kernel_map=cfg['kernel_map'], norm=cfg['norm'], use_gpu=True, verbose=True)
                        print('Saving (kernel train)... '),
                        with open(join(cfg['output_kernels_path'], fusion_kernel_name + '.train.pkl'), 'wb') as f:
                            cPickle.dump(dict(kernel_map=kernel_map, norm=norm, \
                                              kernels=(Kf_train,None), labels=(labels_train,labels_test)), f)
                    # ---
                    print('Branch kernel (test)...')
                    if cfg['distributed']:
                        Kf_test = distributed_atep([tree_dw_paths, branch_dw_paths], inds_test, inds_train=inds_train, max_depth=cfg['max_depth'], use_gpu=True, verbose=True)
                    else:
                        if train is None: # if not train already loaded
                            train = load_darwins(inds_train, [tree_dw_paths, branch_dw_paths], max_depth=cfg['max_depth'])
                        test = load_darwins(inds_test, [tree_dw_paths, branch_dw_paths], max_depth=cfg['max_depth'])
                        Kf_test = atep(test, train, is_train=False, kernel_map=cfg['kernel_map'], norm=cfg['norm'], use_gpu=True, verbose=True)
                    print('Saving kernels (train,test)... '),
                    with open(fusion_kernel_filepath, 'wb') as f:
                        cPickle.dump(dict(kernel_map=kernel_map, norm=norm, \
                                          kernels=(Kf_train,Kf_test), labels=(labels_train,labels_test)), f)
                    fusion_pkl = dict(kernels=(Kf_train, Kf_test),labels=(labels_train,labels_test))
                fusion_kernels[feat_t] = fusion_pkl['kernels']
            # ---

            # Evaluate metric
            # print feat_t
            if 'r' in cfg['midlevels']:
                print 'r'
                train_and_classify([root_pkl['kernels']], root_pkl['labels'], C=cfg['svm_C'], metric=cfg['metric'])
            if 'n' in cfg['midlevels']:
                print 'n'
                train_and_classify([tree_pkl['kernels']], tree_pkl['labels'], C=cfg['svm_C'], metric=cfg['metric'])
            if 'b' in cfg['midlevels']:
                print 'b'
                train_and_classify([branch_pkl['kernels']], branch_pkl['labels'], C=cfg['svm_C'], metric=cfg['metric'])
            if 'n' in cfg['midlevels'] and 'b' in cfg['midlevels']:
                print 'n + b'
                train_and_classify([tree_pkl['kernels'], branch_pkl['kernels']], tree_pkl['labels'], C=cfg['svm_C'], metric=cfg['metric'])
            if 'nb' in cfg['midlevels']:
                print 'nb'
                train_and_classify([fusion_pkl['kernels']], fusion_pkl['labels'], C=cfg['svm_C'], metric=cfg['metric'])
            if 'r' in cfg['midlevels'] and 'n' in cfg['midlevels']:
                print 'r + n'
                train_and_classify([root_pkl['kernels'], tree_pkl['kernels']], root_pkl['labels'], C=cfg['svm_C'],  metric=cfg['metric'])
            if 'r' in cfg['midlevels'] and 'b' in cfg['midlevels']:
                print 'r + b'
                train_and_classify([root_pkl['kernels'], branch_pkl['kernels']], root_pkl['labels'], C=cfg['svm_C'],  metric=cfg['metric'])
            if 'r' in cfg['midlevels'] and 'nb' in cfg['midlevels']:
                print 'r + nb'
                train_and_classify([root_pkl['kernels'],fusion_pkl['kernels']], root_pkl['labels'], C=cfg['svm_C'], metric=cfg['metric'])
            if 'r' in cfg['midlevels'] and 'n' in cfg['midlevels'] and 'b' in cfg['midlevels']:
                print 'r + n + b'
                train_and_classify([root_pkl['kernels'], tree_pkl['kernels'], branch_pkl['kernels']], root_pkl['labels'], C=cfg['svm_C'], metric=cfg['metric'])

            # print 'root + branch'
            # train_and_classify([root_pkl['kernels'], branch_pkl['kernels']], root_pkl['labels'], C=cfg['svm_C'], metric=cfg['metric'])
            # print 'root + node + branch'
            # train_and_classify([root_pkl['kernels'], tree_pkl['kernels'], branch_pkl['kernels']], root_pkl['labels'], C=cfg['svm_C'], metric=cfg['metric'])
            # print
        print 'all'

        # if len(cfg['feat_types']) > 1:
        #     train_and_classify([root_kernels], root_pkl['labels'],C=cfg['svm_C'],  metric=cfg['metric'])
        #     train_and_classify([tree_kernels], root_pkl['labels'],C=cfg['svm_C'],  metric=cfg['metric'])
        #     train_and_classify([branch_kernels], root_pkl['labels'],C=cfg['svm_C'],  metric=cfg['metric'])
        #     train_and_classify([fusion_kernels], root_pkl['labels'],C=cfg['svm_C'],  metric=cfg['metric'])
        #     train_and_classify([root_kernels, fusion_kernels], root_pkl['labels'], metric=cfg['metric'])
        #     train_and_classify([root_kernels, tree_kernels], root_pkl['labels'], metric=cfg['metric'])
        #     train_and_classify([root_kernels, branch_kernels], root_pkl['labels'], metric=cfg['metric'])
        #     train_and_classify([root_kernels, tree_kernels, branch_kernels], root_pkl['labels'], metric=cfg['metric'])
        #     print

    ctx.pop()
    ctx.detach()
    print 'done'