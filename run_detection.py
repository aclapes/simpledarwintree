from darwintree import detection
from os import listdir,makedirs,walk,remove
from os.path import isfile,join,splitext,exists,basename,expanduser
import cPickle
from collections import OrderedDict

# Configuration
cfg = dict(
    dataset_path = '/data/hupba/Datasets/ucf_sports_actions/',
    trajectories_path = '/data/see4c/Derived/improved_dense_trajectories/ucf_sports_actions/',
    clusters_path = '/data/see4c/Derived/spectral_clustering/ucf_sports_actions/',
    darws_path = '/data/see4c/Derived/darwintree.matimpl/ucf_sports_actions/darws.mock/',
    output_kernels_path = '/data/hupba/Derived/darwintree.detection/ucf_sports_actions/',
    video_ext = '.avi',
    norm='l2',
    traj_L=15,
    kernel_map='posneg',
    pre_processing=False,  # kernel mapping and norm swhen loading to RAM darwin representations.
                          # if True less RAM but slower (recommended in case of kernel_map == 'posneg')
    partitions = [1],  # one and only partition
    negative_class = None,
    feat_types=['mbh'],
    branch_pooling='gbl',
    svm_C=[100],
    gpu_id=4    ,
    max_depth=64
)

def build_detection_groundtruth(cfg, output_gt_file):
    if exists(join(cfg['dataset_path'], output_gt_file)):
        return

    groundtruth_path = '/home/aclapes/Downloads/UCF-Sports-annotations-master/Annotations'

    video_names = [splitext(f)[0] for dp, dn, fn in walk(expanduser(cfg['dataset_path'])) for f in fn if f.endswith(cfg['video_ext'])]
    video_names.sort()

    annots_all = OrderedDict()
    for vn in video_names:
        action_name, aid = str.split(vn, '_')

        # gt_files = listdir(join(groundtruth_path, action_name, aid, 'gt'))
        gt_files = [f for dp, dn, fn in walk(expanduser(join(groundtruth_path, action_name, aid, 'gt'))) for f in fn if f.endswith('.txt')]
        gt_files.sort()
        annot = OrderedDict()
        for i,file in enumerate(gt_files):
            with open(join(groundtruth_path, action_name, aid, 'gt', file), 'rb') as f:
                lines = f.readlines()
            # split the line by tabs or spaces (depending on the file. try one first, then the other)
            line0 = str.split(lines[0], '\t')[:-1]
            if len(line0) == 0:
                line0 = str.split(lines[0], ' ')[:-1]
            annot[i+1] = (int(line0[0]), int(line0[1]), int(line0[0])+int(line0[2])-1, int(line0[1])+int(line0[3])-1)
        annots_all[vn] = annot

    with open(join(cfg['dataset_path'], output_gt_file), 'wb') as f:
        cPickle.dump(annots_all, f)

    return


if __name__ == "__main__":
    print cfg
    build_detection_groundtruth(cfg, join(cfg['dataset_path'], 'detection_gt.pkl'))
    detection(cfg)