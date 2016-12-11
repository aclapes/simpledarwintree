from simpledarwintree import simpledarwintree

# Configuration
cfg = dict(
    dataset_path = '/data/hupba/Datasets/olympic_sports/',
    darws_path = '/data/cvpr2016/Derived/darwintree.matimpl/olympic_sports/darws.tmp-8/',
    output_kernels_path = '/data/hupba/Derived/darwintree.pyimpl/olympic_sports/kernels.mock.max_depth=8/',
    norm='l2',
    kernel_map='rootsift',
    pre_processing=False,  # kernel mapping and norm when loading to RAM darwin representations.
                          # if True less RAM but slower (recommended in case of kernel_map == 'posneg')
    partitions = [1],  # one and only partition
    negative_class = None,
    metric = 'acc',  # acc or map
    feat_types=['mbh'],
    maximum_depth=None,
    branch_pooling='gbl',
    svm_C=[100],
    gpu_id=4,
    distributed=True,
    train_chunks=['A','B','C','D','E','F'],
    test_chunks=['A','B','C','D'],
    max_depth=8,
    midlevels=['r']
)

if __name__ == "__main__":
    print cfg
    simpledarwintree(cfg)