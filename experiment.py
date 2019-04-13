import os
import json
import shutil
import argparse
from tensorflow.python.client import device_lib
from collections import Counter
import numpy as np
import datetime
from collections import OrderedDict

import attacks 

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
    s = args.num_exp_per_param // 20 if args.num_exp_per_param > 10 else 2

    imgs_specified = args.img_idxs is not None
    if imgs_specified:
        img_idxs = []
        npz = np.load(args.img_idxs)
        for orig_i, target_i, target_class in zip(npz['original_idxs'], npz['target_idxs'], npz['target_classes']):
            img_idxs.append((orig_i, target_i, target_class))
        assert args.num_exp_per_param <= len(img_idxs)

    for val in args.exp_param_range:
        set_param(args, args.exp_param, val)
        for val_2 in args.exp_param_range_2:
            set_param(args, args.exp_param_2, val_2)
            num_iters = []
            results = []
            infos = []
            detections_success = {}
            detections_failure = {}
            l2_distortions = []
            num_detections_success = {}
            num_detections_failure = {}

            print("[experiment] %s=%f, %s=%f" % (args.exp_param, val, args.exp_param_2, val_2))
            key = (val, val_2)

            for i in range(args.num_exp_per_param):
                # if i % s == 0:

                if imgs_specified:
                    orig_i, target_i, target_class = img_idxs[i]
                    args.img_index, args.target_img_index = int(orig_i), int(target_i)
                    args.target_class = int(target_class)

                success, retval, info, detectors, distortion = attacks.main(args, gpus)

                result = retval
                detections, num_detections = detections_success, num_detections_success
                if success:
                    result = 1
                    num_iters.append(retval)
                    l2_distortions.append(distortion)
                else:
                    result = min(retval, 0)
                    detections, num_detections = detections_failure, num_detections_failure

                for d_name, detector in detectors.detectors.items():
                    if d_name not in detections:
                        detections[d_name] = []

                    if d_name not in num_detections:
                        num_detections[d_name] = []

                    detections[d_name].append((detector.history, detector.detected_dists))
                    num_detections[d_name].append(len(detector.get_detections()))

                print("[run] Experiment %d/%d: result %d" % (i, args.num_exp_per_param, result))

                results.append(result)
                infos.append(info)

                if i % s == 0:
                  c = Counter(results)
                  # num_iters = np.array(num_iters)
                  n_i = np.array(num_iters)
                  print(str(key), "\t", str(np.mean(n_i)), str(np.median(n_i)), str(np.std(n_i)), json.dumps(c))

            mean_distortion = np.mean(l2_distortions)
            success_rate = len(num_iters) / args.num_exp_per_param
            mean_iters = np.mean(num_iters)

            c = Counter(results)
            num_iters = np.array(num_iters)
            result_s = " ".join([str(key), "\t", str(np.mean(num_iters)), str(np.median(num_iters)), str(np.std(num_iters)), json.dumps(c)])
            results_s.append(result_s)
            print(result_s)
            all_results[key] = (results, num_iters, l2_distortions, infos)

            main_det_results = {}
            for d_name in detections_success:
                detection_results[key + (d_name, )] = (detections_success[d_name], detections_failure[d_name])

                mean_detections_success = np.mean(num_detections_success[d_name])
                mean_detections_failure = np.mean(num_detections_failure[d_name])
                mean_detections = np.mean(num_detections_success[d_name] + num_detections_failure[d_name])
                main_det_results[d_name] = (mean_detections, mean_detections_success, mean_detections_failure)

            main_results[key] = (success_rate, mean_distortion, mean_iters, main_det_results)

            print("Summary so far:")
            for result_s in results_s:
                print(result_s)

    print("Summary:")
    for s in results_s:
        print(s)
    timestamp = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d_%H%M")
    os.makedirs("./experiment_results", exist_ok=True)
    np.savez_compressed("./experiment_results/%s_%s" % (args.exp_name, timestamp), attack_results=all_results,
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
