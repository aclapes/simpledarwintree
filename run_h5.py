from simpledarwintree import simpledarwintree

# Configuration
cfg = dict(
    dataset_path = '/data/hupba/Datasets/highfive/',
    darws_path = '/data/see4c/Derived/darwintree.matimpl/highfive/darws/',
    output_kernels_path = '/data/hupba/Derived/darwintree.pyimpl/highfive/kernels/',
    norm='l2',
    kernel_map='posneg',
    pre_mapping=True,  # if True less RAM but slower (recommended in case of kernel_map == 'posneg')
    partitions = [1,2],
    negative_class = 5,
    metric = 'acc', # acc or map
    feat_types=['mbh']
)

if __name__ == "__main__":
    simpledarwintree(cfg)