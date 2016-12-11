from os import makedirs
from os.path import join, basename, splitext, isfile
import sys
import cPickle
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from yael import ynumpy
import time
import h5py


def split_trajectory_features(data, L=15):
    split_dict = dict()

    split_dict['obj'] = data[:,:10]
    split_dict['trj'] = data[:,10:2*(L+1)]
    split_dict['hog'] = data[:,10+2*(L+1):10+2*(L+1)+96]
    split_dict['hof'] = data[:,10+2*(L+1)+96:10+2*(L+1)+96+108]
    split_dict['mbh'] = data[:,10+2*(L+1)+96+108:10+2*(L+1)+96+108+192]

    return split_dict



def build_fvtrees(trajectories_path, videofiles, traintest_inds, clusters_path, feat_types, fvtrees_path, nt=8, verbose=False):
    for p, _ in enumerate(traintest_inds):
        for j, feat_t in enumerate(feat_types):
            try:
                makedirs(join(fvtrees_path, feat_t + '-' + str(p), 'trees'))
            except OSError,e:
                pass

    train_gmm(trajectories_path, videofiles, traintest_inds, feat_types, fvtrees_path, nt=nt, verbose=verbose)

    for p, part in enumerate(traintest_inds):
        for j, feat_t in enumerate(feat_types):
            gmm_filepath = join(fvtrees_path, feat_t + '-' + str(p), 'gmm-' + str(p) + '.pkl')
            with open(gmm_filepath, 'rb') as file_handle:
                file_content = cPickle.load(file_handle)
                gmm, pca = file_content['gmm'], file_content['pca']

            print 'do stuff'



def train_gmm(trajectories_path, videofiles, traintest_inds, feat_types, fvtrees_path, nt=8, verbose=False):
    for p, part in enumerate(traintest_inds):
        D_all = None
        for j, feat_t in enumerate(feat_types):
            gmm_filepath = join(fvtrees_path, feat_t + '-' + str(p), 'gmm-' + str(p) + '.pkl')
            if not isfile(gmm_filepath):
                if D_all is None:
                    sample_filepath = join(fvtrees_path, feat_t + '-' + str(p), 'sample-' + str(p) + '.h5')
                    try:
                        with h5py.File(sample_filepath, 'r') as h5f_handle:
                            D = h5f_handle.get('data').value
                    except IOError:
                        trajs = load_tracklets_sample(trajectories_path, videofiles, part[p], 1000000, verbose=verbose)
                        with h5py.File(sample_filepath, 'w') as h5f_handle:
                            h5f_handle.create_dataset('data',data=trajs)
                    D_all = split_trajectory_features(trajs)

                start_time = time.time()
                # when loaded all features, choose feat_t ones
                D = D_all[feat_t]

                # apply preprocessing
                if feat_t == 'trj':
                    D = convert_positions_to_displacements(D)

                if feat_t != 'mbh':
                    D = preprocessing.normalize(D, norm='l1', axis=1)
                else:
                    Dx = preprocessing.normalize(D[:,:D.shape[1]/2], norm='l1', axis=1)
                    Dy = preprocessing.normalize(D[:,D.shape[1]/2:], norm='l1', axis=1)
                    D = np.hstack((Dx,Dy))

                if feat_t != 'trj':
                    D = np.sign(D) * np.sqrt(np.abs(D)) # rootsift

                # reduce dimensionality to half (PCA)
                pca = PCA(n_components=int(0.5*D.shape[1]), copy=True)
                pca.fit(D)

                # train GMMs for later FV computation
                D = np.ascontiguousarray(pca.transform(D), dtype=np.float32)
                gmm = ynumpy.gmm_learn(D, 256, nt=nt, niter=100, redo=1)

                with open(gmm_filepath, 'wb') as f:
                    cPickle.dump(dict(gmm=gmm, pca=pca), f)

                elapsed_time = time.time() - start_time
                if verbose:
                    print('[train_fv_gmms] %s -> DONE (in %.2f secs)' % (feat_t, elapsed_time))



def load_tracklets_sample(trajectories_path, videofiles, inds, num_samples, verbose=False):
    num_samples_per_vid = int(float(num_samples) / len(inds))
    D = None  # feat_t's sampled tracklets
    ptr = 0
    for j in range(0, len(inds)):
        idx = inds[j]
        # trajectories_file = join(trajectories_path, splitext(basename(videofiles[idx]))[0] + '.idt')
        trajectories_file = join(trajectories_path, splitext(basename(videofiles[idx]))[0] + '.h5')
        try:
            with h5py.File(trajectories_file, 'r') as h5f:
                d = h5f.get('data').value
                if verbose:
                    print('[load_tracklets_sample] (%d/%d) %s (num feats: %d)' % (j+1, len(inds), videofiles[idx], d.shape[1]))

            # init sample
            if D is None:
                D = np.zeros((num_samples, d.shape[1]), dtype=np.float32)
            # create a random permutation for sampling some tracklets in this vids
            randp = np.random.permutation(d.shape[0])
            if d.shape[0] > num_samples_per_vid:
                randp = randp[:num_samples_per_vid]
            D[ptr:ptr+len(randp),:] = d[randp,:]
            ptr += len(randp)
        except IOError:
            sys.stderr.write('# ERROR: missing training instance'
                                 ' {}\n'.format(trajectories_file))
            sys.stderr.flush()

    dd = D[:ptr,:]

    return dd # cut out extra reserved space

def convert_positions_to_displacements(P):
    '''
    From positions to normalized displacements
    :param D:
    :return:
    '''

    X, Y = P[:,::2], P[:,1::2]  # X (resp. Y) are odd (resp. even) columns of D
    Vx = X[:,1:] - X[:,:-1]  # get relative displacement (velocity vector)
    Vy = Y[:,1:] - Y[:,:-1]

    D = np.zeros((P.shape[0], Vx.shape[1]+Vy.shape[1]), dtype=P.dtype)
    D[:,::2]  = Vx
    D[:,1::2] = Vy

    return D