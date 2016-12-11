import numpy as np
import random
import os
from scipy.io import loadmat, matlab
import h5py
import cPickle


def load_darwins(representations, max_depth=64):
    output_tmpdir = '/data/cvpr2016/Derived/darwintree.matimpl/olympic_sports/darws.tmp-' + str(max_depth)
    try:
        os.makedirs(output_tmpdir)
    except:
        pass

    for i,filepath in enumerate(representations):
        print('%d/%d' % (i+1, len(representations)))

        output_dir = os.path.join(output_tmpdir,str.split(os.path.dirname(filepath), '/')[-1])
        try:
            os.makedirs(output_dir)
        except:
            pass

        stem, ext = os.path.splitext(os.path.basename(filepath))
        output_filepath = os.path.join(output_dir, stem+'.mat')

        # print output_filepath

        if os.path.exists(output_filepath):
            # print '-> Done'
            continue

        # print '-> Reading ',
        X = None
        try:
            mat = loadmat(filepath)  # raise not implemented ee
            if 'W' in mat:
                X = np.squeeze(mat['W'])
            elif 'Wtree' in mat:
                X = mat['Wtree']
            elif 'Btree' in mat:
                X = mat['Btree']
        except NotImplementedError:
            mat = h5py.File(filepath)
            if 'W' in mat.keys():
                X = np.squeeze(mat['W'].value)
                raise AssertionError('TODO: assume W is not stored in -v7.3')
            elif 'Wtree' in mat.keys():
                nodeid = np.squeeze(mat['nodeid'].value.T)
                Wtree = mat['Wtree'].value[:,nodeid < 2**max_depth]
                with h5py.File(output_filepath, 'w') as hf:
                    hf.create_dataset('Wtree', data=Wtree)
                    hf.create_dataset('nodeid', data=nodeid[nodeid < 2**max_depth])
                print '(Saved).'
            elif 'Btree' in mat.keys():
                nodeid = np.squeeze(mat['nodeid'].value.T)
                Btree = mat['Btree'].value[:,nodeid < 2**max_depth]
                with h5py.File(output_filepath, 'w') as hf:
                    hf.create_dataset('Btree', data=Btree)
                    hf.create_dataset('nodeid', data=nodeid[nodeid < 2**max_depth])
                print '(Saved).'
                # ---
        except ValueError:
            print 'ValueError', i, filepath
            continue
        except IOError:
            print 'IOError', i, filepath
            continue
        except matlab.miobase.MatReadError:
            print 'MatReadError', i, filepath
            continue


if __name__ == "__main__":
    darwins_path = '/data/cvpr2016/Derived/darwintree.matimpl/olympic_sports/darws.mock/tree_representation-1-vd_mbh_non-lin'
    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(darwins_path)) for f in fn if f.endswith('.mat')]
    load_darwins(files, max_depth=4)