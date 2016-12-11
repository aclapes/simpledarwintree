from os import makedirs, devnull
from os.path import join,splitext,basename,exists
import subprocess
import sys
from joblib import delayed, Parallel
import numpy as np
import h5py

def extract_idt(extractor_path, videofiles, traj_len, trajfile_ext, output_path):
    """
    Compute
    :param extractor_path:
    :param traj_len:
    :param trajfile_ext:
    :param videofiles:
    :param output_path:
    :return:
    """
    try:
        makedirs(output_path)
    except OSError, e:
        pass

    for vf in videofiles:
        args = [extractor_path, vf, ' '.join(['-L', str(traj_len)])]
        vf_name = splitext(basename(vf))[0]
        print vf_name
        outfile_path = join(output_path, vf_name + '.idt')
        if exists(join(output_path, vf_name + '.idt')):
            if not exists(join(output_path, vf_name + '.h5')):
                with open(join(output_path, vf_name + '.idt'), 'r', 1) as f:
                    d = np.array([np.array(line.strip().split('\t'), dtype=np.float32) for line in f.readlines()])
                    with h5py.File(join(output_path, vf_name + '.h5'), 'w') as h5f:
                        h5f.create_dataset('data', data=d)
            continue

        try:
            FNULL = open(devnull, 'w')
            p = subprocess.Popen(args, shell=False, universal_newlines=True, stdout=subprocess.PIPE, stderr=FNULL)
            out, _ = p.communicate()
            FNULL.close()

            lines = out.strip().split('\n')
            trajectories = np.array([np.array(line.strip().split('\t'), dtype=np.float32) for line in lines])
            with h5py.File(outfile_path, 'w') as h5f:
                h5f.create_dataset('data', data=trajectories)
        except OSError, e:
            print >>sys.stderr, "Execution failed:", e


def extract_idt_multithread(extractor_path, traj_L, trajfile_ext, videofiles, output_path, nt=4):
    """
    Use joblib for multithread on extract_dense_trajectories(...).
    :param extractor_path:
    :param traj_L:
    :param trajfile_ext:
    :param videofiles:
    :param output_path:
    :param nt:
    :return:
    """
    n = int(len(videofiles)/float(nt))+1  # sets the num of points per thread
    ret = Parallel(n_jobs=nt, backend='threading')(delayed(extract_idt)(extractor_path, videofiles[t*n:((t+1)*n if (t+1)*n < len(videofiles) else len(videofiles))], \
                                                                        traj_L, trajfile_ext, output_path)
                                                           for t in xrange(nt))
