from simpledarwintree import simpledarwintree

# Configuration
cfg = dict(
    dataset_path = '/data/hupba/Datasets/ucf_sports_actions/',
    darws_path = '/data/see4c/Derived/darwintree.matimpl/ucf_sports_actions/darws/',
    output_kernels_path = '/data/hupba/Derived/darwintree.pyimpl/ucf_sports_actions/kernels/',
    norm='l2',
    kernel_map='posneg',
    pre_kernel_mapping=True,  # if True less RAM but slower (recommended in case of kernel_map == 'posneg')
    partitions = [1],  # one and only partition
    negative_class = None,
    metric = 'acc',  # acc or map
    feat_types=['mbh']
)

if __name__ == "__main__":
    simpledarwintree(cfg)