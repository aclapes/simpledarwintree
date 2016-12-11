from simpledarwintree import simpledarwintree

# Configuration
cfg = dict(
    dataset_path = '/data/hupba/Datasets/ucf_sports_actions/',
    darws_path = '/data/see4c/Derived/darwintree.matimpl/ucf_sports_actions/darws.mock/',
    output_kernels_path = '/data/hupba/Derived/darwintree.pyimpl/ucf_sports_actions/kernels.no_weights.max_depth=16',
    norm='l2',
    kernel_map='rootsift',
    pre_processing=False,  # kernel mapping and norm swhen loading to RAM darwin representations.
                          # if True less RAM but slower (recommended in case of kernel_map == 'posneg')
    partitions = [1],  # one and only partition
    negative_class = None,
    metric = 'acc',  # acc or map
    feat_types=['mbh'],
    branch_pooling='gbl',
    svm_C=[100],
    gpu_id=4    ,
    distributed=False,
    max_depth=16,
    midlevels=['r']
)

if __name__ == "__main__":
    print cfg
    simpledarwintree(cfg)