from simpledarwintree import simpledarwintree

# Configuration
cfg = dict(
    dataset_path = '/data/hupba/Datasets/highfive/',
    darws_path = '/data/see4c/Derived/darwintree.matimpl/highfive/darws/',
    output_kernels_path = '/data/hupba/Derived/darwintree.pyimpl/highfive/kernels.no_weights.max_depth=2/',
    # output_kernels_path = '/data/hupba/Derived/darwintree.pyimpl/highfive/kernels/',
    norm='l2',
    kernel_map='rootsift',
    # kernel_map = 'posneg',
    pre_processing=False,  # kernel mapping and norm when loading to RAM darwin representations.
                          # if True less RAM but slower (recommended in case of kernel_map == 'posneg')
    partitions = [1],
    negative_class = 5,
    metric = 'map', # acc or map
    feat_types=['mbh'],
    branch_pooling='gbl',
    svm_C=[100],
    gpu_id=4,
    max_depth=2,
    distributed=False,
    midlevels=['n']
    # midlevels=['r','n','b']
)

if __name__ == "__main__":
    print cfg
    simpledarwintree(cfg)