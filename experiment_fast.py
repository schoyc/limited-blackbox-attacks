import os
import json
import shutil
import argparse
from tensorflow.python.client import device_lib
from collections import Counter
import numpy as np
import datetime
from collections import OrderedDict

import attacks_fast

BATCH_SIZE = 50
SIGMA = 1e-3
EPSILON = 0.05
SAMPLES_PER_DRAW = 50
LEARNING_RATE = 1e-2
LOG_ITERS_FACTOR = 2
IMAGENET_PATH = ''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples-per-draw', type=int, default=SAMPLES_PER_DRAW)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--target-class', type=int, help='negative => untargeted')
    parser.add_argument('--orig-class', type=int)
    parser.add_argument('--sigma', type=float, default=SIGMA)
    parser.add_argument('--epsilon', type=float, default=EPSILON)
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--img-index', type=int)
    parser.add_argument('--target-img-index', type=int)
    parser.add_argument('--out-dir', type=str, required=True,
                        help='dir to save to if not gridding; otherwise parent \
                        dir of grid directories')
    parser.add_argument('--log-iters', type=int, default=1)
    parser.add_argument('--restore', type=str, help='restore path of img')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--max-queries', type=int, default=10000)
    parser.add_argument('--save-iters', type=int, default=50)
    parser.add_argument('--plateau-drop', type=float, default=2.0)
    parser.add_argument('--min-lr-ratio', type=int, default=200)
    parser.add_argument('--plateau-length', type=int, default=5)
    parser.add_argument('--gpus', type=int, help='number of GPUs to use')
    parser.add_argument('--imagenet-path', type=str)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--max-lr', type=float, default=1e-2)
    parser.add_argument('--min-lr', type=float, default=5e-5)
    # PARTIAL INFORMATION ARGUMENTS
    parser.add_argument('--top-k', type=int, default=-1)
    parser.add_argument('--adv-thresh', type=float, default=-1.0)
    # LABEL ONLY ARGUMENTS
    parser.add_argument('--label-only', action='store_true')
    parser.add_argument('--zero-iters', type=int, default=100, help="how many points to use for the proxy score")
    parser.add_argument('--label-only-sigma', type=float, default=1e-3, help="distribution width for proxy score")
    parser.add_argument('--starting-eps', type=float, default=1.0)
    parser.add_argument('--starting-delta-eps', type=float, default=0.5)
    parser.add_argument('--min-delta-eps', type=float, default=0.1)
    parser.add_argument('--conservative', type=int, default=2, help="How conservative we should be in epsilon decay; increase if no convergence")

    # Experiment arguments
    parser.add_argument('--exp-name', type=str)
    parser.add_argument('--exp-param-range', type=float, nargs='+', help='Different values to experiment params with', required=True)
    parser.add_argument('--exp-param', type=str, help='Param to experiment with', required=True)
    parser.add_argument('--exp-param-range-2', type=float, nargs='+', help='Different values to experiment params with',
                        required=False)
    parser.add_argument('--exp-param-2', type=str, help='Param to experiment with', required=False)
    parser.add_argument('--num-exp-per-param', type=int, default=100)
    parser.add_argument('--strat-param', type=float, default=1)
    parser.add_argument('--est-strat', type=str, default="uniform")
    parser.add_argument('--img-idxs', type=str, default=None, help='Image indices to use')
    parser.add_argument('--no-detection', action='store_true')
    parser.add_argument('--idx-range', type=int, nargs=2)
    args = parser.parse_args()

    # Data checks
    if not (args.img_path is None and args.img_index is not None or
            args.img_path is not None and args.img_index is None):
        raise ValueError('can only set one of img-path, img-index')
    if args.img_path and not (args.orig_class or args.target_class):
        raise ValueError('orig and target class required with image path')
    if (args.target_class is None and args.img_index is None):
        raise ValueError('must give target class if not using index')
    assert args.samples_per_draw % args.batch_size == 0
    gpus = get_available_gpus()
    if args.gpus:
        if args.gpus > len(gpus):
            raise RuntimeError('not enough GPUs! (requested %d, found %d)' % (args.gpus, len(gpus)))
        gpus = gpus[:args.gpus]
    if not gpus:
        raise NotImplementedError('no support for using CPU-only because lazy')
    if args.batch_size % 2*len(gpus) != 0:
        raise ValueError('batch size must be divisible by 2 * number of GPUs (batch_size=%d, gpus=%d)' % (
            batch_size,
            len(gpus)
        ))

    # CLEAR THE PATH
    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)

    # PRINT PARAMS
    args_text = json.dumps(args.__dict__)
    print(args_text)

    all_results = {}
    main_results = {}
    detection_results = {}
    results_s = []
    print("Experiment with param:", str(args.exp_param))

    imgs_specified = args.img_idxs is not None
    if imgs_specified:
        img_idxs = []
        npz = np.load(args.img_idxs)
        for orig_i, target_i, target_class in zip(npz['original_idxs'], npz['target_idxs'], npz['target_classes']):
            img_idxs.append((orig_i, target_i, target_class))
        assert args.num_exp_per_param <= len(img_idxs)

    timestamp = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d_%H%M")
    os.makedirs("./experiment_results", exist_ok=True)

    results_npz = "./experiment_results/%s_%s" % (args.exp_name, timestamp)

    idx_range_specified = args.idx_range is not None
    if idx_range_specified:
        start_idx, end_idx = args.idx_range[0], args.idx_range[1]
        results_npz = "./experiment_results/%s_s%d_e%d_%s" % (args.exp_name, start_idx, end_idx, timestamp)

    idx_range = range(start_idx, end_idx) if idx_range_specified else range(args.num_exp_per_param)
    for val in args.exp_param_range:
        set_param(args, args.exp_param, val)
        for val_2 in args.exp_param_range_2:
            set_param(args, args.exp_param_2, val_2)
            print("[experiment] %s=%f, %s=%f" % (args.exp_param, val, args.exp_param_2, val_2))
            key = (val, val_2)

            attacks_fast.main(args, gpus, all_results, main_results, detection_results, results_s, key, idx_range, img_idxs)


    print("Summary:")
    for s in results_s:
        print(s)

    np.savez_compressed(results_npz, attack_results=all_results,
                        detection_results=detection_results, main_results=main_results,
                        params=np.array(args.exp_param_range), args=vars(args))


def set_param(args, param, val):
    if param == 'label-only-sigma':
        args.label_only_sigma = val
    elif param == 'sigma':
        args.sigma = val
    elif param == 'zero-iters':
        args.zero_iters = int(val)
    elif param == 'strat-param':
        args.strat_param = val
    else:
        raise ValueError("Unrecognized param!")

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

if __name__ == "__main__":
    main()
