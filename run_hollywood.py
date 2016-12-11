from os import listdir,makedirs,walk,remove,system,rename
from os.path import isfile,join,splitext,exists,basename,expanduser,expandvars
import cPickle
from collections import OrderedDict
import subprocess
import sys
from joblib import delayed, Parallel
import numpy as np
import h5py

from trajectory_extraction import extract_idt_multithread
from tracklet_clustering import cluster_multithread
from midlevel_representation import build_fvtrees

# Configuration
cfg = dict(
    dataset_path = '/data/hupba/Datasets/hollywood/',
    videos_path = '/data/hupba/Datasets/hollywood/videoclips/',
    trajectories_path = '/data/hupba/Derived/improved_dense_trajectories/hollywood/',
    clusters_path = '/data/hupba/Derived/spectral_clustering/hollywood/',
    fvtrees_path ='/data/hupba/Derived/darwintree.detection/hollywood/fvtrees/',
    output_kernels_path = '/data/hupba/Derived/darwintree.detection/hollywood/',
    extractor_path = '/home/aclapes/Code/Cpp/improved_trajectory_release.abs_traj_mod/release/DenseTrackStab',
    video_ext = '.avi',
    trajfile_ext = '.idt',
    norm='l2',
    traj_len=15,
    kernel_map='posneg',
    pre_processing=False,  # kernel mapping and norm swhen loading to RAM darwin representations.
                          # if True less RAM but slower (recommended in case of kernel_map == 'posneg')
    partitions = [0,1],
    negative_class = None,
    feat_types=['mbh'],
    branch_pooling='gbl',
    svm_C=[100],
    gpu_id=4    ,
    max_depth=64
)

def g(cfg):
    annotations_path = join(cfg['dataset_path'], 'annotations')

    # load train
    train_file_path = join(annotations_path, 'train_clean.txt')
    test_file_path = join(annotations_path, 'test_clean.txt')
    with open(train_file_path, 'r') as f:
        train_annots = [line.replace('"', '') for line in f.readlines()]
    with open(test_file_path, 'r') as f:
        test_annots = [line.replace('"', '') for line in f.readlines()]

    partition_ind = []
    filenames = []
    beginends = []
    action_names = []
    for p, parti in enumerate([train_annots, test_annots]):
        for i, annot in enumerate(parti):
            partition_ind.append(p+1)
            filenames.append(annot[:annot.find(cfg['video_ext'])].replace(' ', '_') + cfg['video_ext'])
            beginends.append(np.array(annot[annot.find('(')+1:annot.find(')')].split('-'), dtype=np.int32))
            action_names.append(annot[annot.find('<')+1:annot.find('>')])

    action_id_lut = {name:i for i,name in enumerate(np.unique(action_names))}

    class_onezero_enc = np.zeros((len(action_names),len(action_id_lut)), dtype=np.int32)
    for i,name in enumerate(action_names):
        class_onezero_enc[i,action_id_lut[name]] = 1

    traintest_inds = [(np.where(np.array(partition_ind) == p)[0], np.where(np.array(partition_ind) != p)[0])]  # holdout (train_inds, test_inds)

    return filenames, beginends, class_onezero_enc, traintest_inds

if __name__ == "__main__":
    print cfg

    all_videofiles = [join(cfg['videos_path'], f) for dp, dn, fn in walk(expanduser(cfg['videos_path'])) for f in fn if f.endswith(cfg['video_ext'])]
    all_videofiles.sort()

    # extract_idt_multithread(cfg['extractor_path'], cfg['traj_len'], cfg['trajfile_ext'], all_videofiles, cfg['trajectories_path'], nt=1)
    # cluster_multithread(cfg['trajectories_path'], all_videofiles, cfg['clusters_path'], nt=12, verbose=False, visualize=False)

    filenames, beginends, class_onezero_enc, traintest_inds = g(cfg)
    clean_videofiles = [join(cfg['videos_path'], f) for f in filenames]
    build_fvtrees(cfg['trajectories_path'], clean_videofiles, traintest_inds, cfg['clusters_path'], cfg['feat_types'], cfg['fvtrees_path'], nt=8, verbose=True)

    print 'done'