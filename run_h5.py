from simpledarwintree import simpledarwintree

# Configuration
cfg = dict(
    darws_path = '/data/hupba/Derived/darwintree.matimpl/highfive/darws/',
    darws_py_path = '/data/hupba/Derived/darwintree.pyimpl/highfive/darws/',
    atep_path = '/data/hupba/Derived/darwintree.pyimpl/highfive/kernels/',
    dataset_path = '/data/hupba/Datasets/highfive/',
    norm='l2',
    kernel_map='posneg',
    partitions = [1,2],
    negative_class = 5,
    metric = 'acc', # acc or map
    feat_types=['mbh']
)

if __name__ == "__main__":
    simpledarwintree(cfg)