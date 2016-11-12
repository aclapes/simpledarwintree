from simpledarwintree import simpledarwintree

# Configuration
cfg = dict(
    dataset_path = '/data/hupba/Datasets/ucf_sports_actions/',
    darws_path = '/data/see4c/Derived/darwintree.matimpl/ucf_sports_actions/darws/',
    output_kernels_path = '/data/hupba/Derived/darwintree.pyimpl/ucf_sports_actions/kernels/',
    norm='l2',
    kernel_map='rootsift',
    pre_processing=False,  # kernel mapping and norm when loading to RAM darwin representations.
                          # if True less RAM but slower (recommended in case of kernel_map == 'posneg')
    partitions = [1],  # one and only partition
    negative_class = None,
    metric = 'map',  # acc or map
    feat_types=['mbh'],
    maximum_depth=None,
    branch_pooling='gbl'
)

if __name__ == "__main__":
    simpledarwintree(cfg)