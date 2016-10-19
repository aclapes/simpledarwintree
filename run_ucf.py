from simpledarwintree import simpledarwintree

# Configuration
cfg = dict(
    darws_path = '/data/see4c/Derived/darwintree.matimpl/ucf_sports_actions/darws/',
    atep_path = '/data/hupba/Derived/darwintree.pyimpl/ucf_sports_actions/kernels/',
    dataset_path = '/data/hupba/Datasets/ucf_sports_actions/',
    norm='l2',
    kernel_map='posneg',
    partitions = [1],  # one and only partition
    negative_class = None,
    metric = 'map',  # acc or map
    feat_types=['hog','hof','mbh']
)

if __name__ == "__main__":
    simpledarwintree(cfg)