from scipy.io import loadmat
from os import listdir,makedirs
from os.path import isfile,join,splitext,exists
import numpy as np
from sklearn import preprocessing, svm, multiclass, metrics, cross_validation
import itertools
from joblib import delayed, Parallel
import cPickle
import sys
import random
import threading
import svmutil

# ------------------------------------------------------------------------
# Auxiliary
# ------------------------------------------------------------------------

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


def get_id_from_darwinfile(filename):
    return (str.split(filename, '-')[0]).zfill(5)


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


def atep(trees_i, trees_j, is_train=False, kernel_map=None, norm=None, nt=1, verbose=False):
    # List all the kernel points to distribute over "nt" threads
    points = []
    if is_train:
        points += [(i,i) for i in xrange(len(trees_i))]  # diagonal
        points += [p for p in itertools.combinations(np.arange(len(trees_i)),2)]  # upper-triangle combinations
    else:
        points += [ p for p in itertools.product(*[np.arange(len(trees_i)),np.arange(len(trees_j))]) ]

    random.shuffle(points)  # balances computation among threads
    n = int(len(points)/float(nt))+1  # sets the num of points per thread

    lock = ProgressLock(len(points))
    ret = Parallel(n_jobs=nt, backend='threading')(delayed(_atep)(trees_i, trees_j, \
                                                                  points[t*n:((t+1)*n if (t+1)*n < len(points) else len(points))],\
                                                                  kernel_map=kernel_map, norm=norm, lock=lock,
                                                                  verbose=(True if t == 0 else False))
                                                   for t in xrange(nt))

    # Complete the kernel from multiple threads' results
    K = np.zeros((len(trees_i),len(trees_j)), dtype=np.float32)
    K[:] = np.nan

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
        with lock:
            lock.update()
            if verbose:
                print('Progress: %.2f%%' % (lock.progress()*100.0))

    return points, res


def train_and_classify(kernels, labels, metric='acc'):
    if metric == 'acc':
        train_and_classify_acc(kernels, labels)
    elif metric == 'map':
        train_and_classify_map(kernels, labels)


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

def train_and_classify_acc(kernels, labels, C=[0.1, 1, 10, 100, 1000]):
    # Fuse different modalities by averaging
    for i,representations in enumerate(kernels):
        if not isinstance(representations, tuple):
            kernels[i] = (np.sum([(1.0/len(representations))*K for (K,_) in representations.values()],axis=0),
                          np.sum([(1.0/len(representations))*K for (_,K) in representations.values()],axis=0))

    # Validate the weight of the different mid-level representations
    if len(kernels) == 1:
        W = np.array([[1.0]])
    elif len(kernels) == 2:
        lin = np.linspace(0,1,21)
        W = np.vstack((lin, 1-lin)).T
    else:
        W = random_point_distribution(1000,len(kernels))

    labels_train, labels_test = labels  # assume tree_kernel labels coincide! assertion needed

    C_perfs = np.zeros((W.shape[0],len(C)), dtype=np.float32)
    for i in xrange(W.shape[0]):
        w = W[i,:]
        K_train = np.sum([w[k]*K for k,(K,_) in enumerate(kernels)], axis=0)
        # K_test  = np.sum([w[k]*K for k,(_,K) in enumerate(kernels)], axis=0)
        for j,c in enumerate(C):
            cv = cross_validation.StratifiedKFold(labels_train, n_folds=4)
            cv_val_perfs = []
            for v,(val_tr,val_te) in enumerate(cv):
                ovr = multiclass.OneVsRestClassifier(svm.SVC(kernel='precomputed', class_weight='balanced', C=c, random_state=42))
                ovr.fit(K_train[val_tr,:][:,val_tr], labels_train[val_tr])
                acc = metrics.accuracy_score(labels_train[val_te], ovr.predict(K_train[val_te,:][:,val_tr]))
                cv_val_perfs.append(acc)
            C_perfs[i,j] = np.mean(cv_val_perfs)

    best_idx = np.unravel_index(np.argmax(C_perfs), C_perfs.shape)
    w = W[best_idx[0],:]
    c = C[best_idx[1]]
    print w,c,np.max(C_perfs)

    ovr = multiclass.OneVsRestClassifier(svm.SVC(kernel='precomputed', class_weight='balanced', C=c, random_state=42))
    K_train = np.sum([w[k]*K for k,(K,_) in enumerate(kernels)], axis=0)
    ovr.fit(K_train, labels_train)

    K_test  = np.sum([w[k]*K for k,(_,K) in enumerate(kernels)], axis=0)
    print 'Out-of-sample acc:', metrics.accuracy_score(labels_test, ovr.predict(K_test))


def train_and_classify_map(kernels, labels, C=[0.1, 1, 10, 100, 1000]):
# Fuse different modalities by averaging
    for i,representations in enumerate(kernels):
        if not isinstance(representations, tuple):
            kernels[i] = (np.sum([(1.0/len(representations))*K for (K,_) in representations.values()],axis=0),
                          np.sum([(1.0/len(representations))*K for (_,K) in representations.values()],axis=0))

    # Validate the weight of the different mid-level representations
    if len(kernels) == 1:
        W = np.array([[1.0]])
    elif len(kernels) == 2:
        lin = np.linspace(0,1,21)
        W = np.vstack((lin, 1-lin)).T
    else:
        W = random_point_distribution(1000,len(kernels))

    labels_train, labels_test = labels  # assume tree_kernel labels coincide! assertion needed

    unique_classes = np.unique(labels_train)
    ap_classes = [None] * len(unique_classes)

    for cl in unique_classes:
        C_perfs = np.zeros((W.shape[0],len(C)), dtype=np.float32)
        for i in xrange(W.shape[0]):
            w = W[i,:]
            K_train = np.sum([w[k]*K for k,(K,_) in enumerate(kernels)], axis=0)
            K_test  = np.sum([w[k]*K for k,(_,K) in enumerate(kernels)], axis=0)
            for j,c in enumerate(C):
                cv = cross_validation.StratifiedKFold(labels_train == cl, n_folds=4)
                cv_perfs = []
                for v,(val_tr,val_te) in enumerate(cv):
                    model = svm.SVC(kernel='precomputed', class_weight='balanced', C=c, random_state=42)
                    model.fit(K_train[val_tr, :][:, val_tr], labels_train[val_tr] == cl)
                    map = metrics.average_precision_score(labels_train[val_te] == cl,
                                                          model.decision_function(K_train[val_te, :][:, val_tr]))
                    cv_perfs.append(map)
                C_perfs[i,j] = np.mean(cv_perfs)

        best_idx = np.unravel_index(np.argmax(C_perfs), C_perfs.shape)
        w = W[best_idx[0],:]
        c = C[best_idx[1]]
        print w,c

        model = svm.SVC(kernel='precomputed', class_weight='balanced', C=c, random_state=42)

        K_train = np.sum([w[k]*K for k,(K,_) in enumerate(kernels)], axis=0)
        model.fit(K_train, labels_train == cl)

        K_test  = np.sum([w[k]*K for k,(_,K) in enumerate(kernels)], axis=0)
        ap_classes[cl] = metrics.average_precision_score(labels_test == cl, model.decision_function(K_test))

    print ap_classes
    print 'Out-of-sample mAP:', np.mean(ap_classes)


# ------------------------------------------------------------------------
# main function
# ------------------------------------------------------------------------

def simpledarwintree(cfg):
    train_test_split = loadmat(join(cfg['dataset_path'], 'train_test_split.mat'))

    for pt in cfg['partitions']:

        root_kernels = {}
        tree_kernels = {}

        for feat_t in cfg['feat_types']:
            print feat_t

            vd_path = join(cfg['darws_path'], 'representation-' + str(pt) + '-vd_' + feat_t + '_non-lin')
            tree_path = join(cfg['darws_path'], 'tree_representation-' + str(pt) + '-vd_' + feat_t + '_non-lin')

            # vd_py_path = join(cfg['darws_py_path'], 'representation-' + str(pt) + '-vd_' + feat_t + '_non-lin')
            # tree_py_path = join(cfg['darws_py_path'], 'tree_representation-' + str(pt) + '-vd_' + feat_t + '_non-lin')

            try:
                # makedirs(cfg['darws_py_path'])
                # makedirs(vd_py_path)
                # makedirs(tree_py_path)
                makedirs(cfg['output_kernels_path'])
            except:
                pass

            root_kernel_name = 'kernel-' + str(pt) + '-vd_' + feat_t + '_non-lin_' + cfg['kernel_map']
            tree_kernel_name = 'tree_kernel-' + str(pt) + '-vd_' + feat_t + '_non-lin_' + cfg['kernel_map']
            root_kernel_filepath = join(cfg['output_kernels_path'], root_kernel_name + '.pkl')
            tree_kernel_filepath = join(cfg['output_kernels_path'], tree_kernel_name + '.pkl')

            if not exists(root_kernel_filepath) or not exists(tree_kernel_filepath):
                # List darwin representations/tree-representations paths from disk
                darws = [f for f in listdir(vd_path) if isfile(join(vd_path, f)) and splitext(f)[-1] == '.mat']
                darws.sort(key=get_id_from_darwinfile)

                # Load and pre-process darwin files
                # Preprocessing: posneg kernel map followed by l2-norm
                all_roots = None
                all_trees = [[] for i in xrange(len(darws))]
                for i in xrange(len(darws)):
                    print ('[%d/%d] Loading darws..' % (i+1,len(darws)))
                    print darws[i]

                    # if not exists(root_kernel_filepath):
                    root_mat = loadmat(join(vd_path, darws[i]))
                    # with open(join(vd_py_path, darws[i]), 'wb') as f:
                    #     cPickle.dump(root_mat,f)

                    waux = np.squeeze(root_mat['W'])[np.newaxis,:]
                    w = preprocessing.normalize(apply_kernel_map(waux, map=cfg['kernel_map']), norm=cfg['norm'], axis=1)
                    if all_roots is None:
                        all_roots = np.zeros((len(darws), w.shape[1]), dtype=np.float64)
                    all_roots[i,:] = w

                    if not exists(tree_kernel_filepath):
                        tree_mat = loadmat(join(tree_path, darws[i]))
                        # Wtree = np.hstack( (tree_mat['Wtree'], np.zeros((tree_mat['Wtree'].shape[0],1))) )
                        # all_trees[i] = preprocessing.normalize(apply_kernel_map(tree_mat['Wtree'],map=cfg['kernel_map']), norm=cfg['norm'], axis=1)
                        if not cfg['pre_mapping']:
                            all_trees[i] = tree_mat['Wtree']
                        else:
                            all_trees[i] = preprocessing.normalize(apply_kernel_map(tree_mat['Wtree'],map=cfg['kernel_map'],copy=False), norm=cfg['norm'], axis=1, copy=False)
                    # with open(join(tree_py_path, darws[i]), 'wb') as f:
                    #     cPickle.dump(tree_mat,f)

                all_trees = np.array(all_trees)

                # Get training and test split from our standard file
                inds_train = np.squeeze(train_test_split['cur_train_indx'][0][pt-1])-1
                inds_test = np.squeeze(train_test_split['cur_test_indx'][0][pt-1])-1
                # Filter out negative examples from test
                if cfg['negative_class'] is not None:
                    inds_test = inds_test[np.squeeze(train_test_split['labels2'][inds_test]) != cfg['negative_class']]

                labels = train_test_split['labels2']
                if all_roots.shape[0] < len(labels):
                    sys.stderr.write('Representations missing. Quitting.')
                    quit()

                if len(all_trees) < len(labels):
                    sys.stderr.write('Tree representations missing. Quitting.')
                    quit()

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

                kernel_map = None if cfg['pre_mapping'] else cfg['kernel_map']
                norm = None if cfg['pre_mapping'] else cfg['norm']

                if not exists(root_kernel_filepath):
                    # all_roots = np.sqrt(all_roots, dtype=np.complex128)
                    print('Kernel (train)...')
                    K_train = np.dot(all_roots[inds_train, :], all_roots[inds_train, :].T)
                    print('Kernel (test)...')
                    K_test  = np.dot(all_roots[inds_test, :], all_roots[inds_train, :].T)
                    print('Saving (kernel train/test)... '),
                    with open(root_kernel_filepath, 'wb') as f:
                        cPickle.dump(dict(kernel_map=kernel_map, norm=norm, \
                                          kernels=(K_train, K_test), labels=(labels_train, labels_test)), f)
                    print('DONE.')

                if not exists(tree_kernel_filepath):
                    print('Tree kernel (train)...')
                    Kn_train = atep(all_trees[inds_train], all_trees[inds_train], is_train=True,
                                    kernel_map=kernel_map, norm=norm, nt=35)
                    print('Tree kernel (test)...')
                    Kn_test  = atep(all_trees[inds_test], all_trees[inds_train],
                                    kernel_map=kernel_map, norm=norm, nt=35)
                    print('Saving (kernel train/test)... '),
                    with open(tree_kernel_filepath, 'wb') as f:
                        cPickle.dump(dict(kernel_map=kernel_map, norm=norm, \
                                          kernels=(Kn_train,Kn_test), labels=(labels_train,labels_test)), f)
                    print('DONE.')

                del all_roots
                del all_trees

            # Evaluate metric

            with open(root_kernel_filepath, 'rb') as f:
                root_pkl = cPickle.load(f)
                root_kernels[feat_t] = root_pkl['kernels']
            with open(tree_kernel_filepath, 'rb') as f:
                tree_pkl = cPickle.load(f)
                tree_kernels[feat_t] = tree_pkl['kernels']

            train_and_classify([root_pkl['kernels'], tree_pkl['kernels']], root_pkl['labels'], metric=cfg['metric'])
            # train_and_classify([root_pkl['kernels']], root_pkl['labels'], metric=cfg['metric'])

        print 'all'

        # if len(root_kernels.keys()) != len(tree_kernels.keys()):
        #     master_dict, slave_dict = root_kernels, tree_kernels\
        #         if len(root_kernels.keys()) > len(tree_kernels.keys()) else tree_kernels, root_kernels
        #     for feat_t,kernels in master_dict:
        #         if not feat_t in slave_dict:
        #             del slave_dict[feat_t]

        # train_and_classify([root_kernels, tree_kernels], root_pkl['labels'], metric=cfg['metric'])

    print 'done'