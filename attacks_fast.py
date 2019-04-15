from PIL import Image
import numpy as np
import tensorflow as tf

from tools.utils import *
import json
import pdb
import os
import sys
import time
import scipy.misc
import PIL
import tools.estimation_strats as est_strats
import tools.detection as detection

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.logging_utils import *

# from tools.inception_v3_imagenet import model
# from tools.imagenet_labels import label_to_name
# from tools.tf_sample_cifar10 import model
from tools.resnet_v1_cifar10 import model
from tensorflow.keras.datasets import cifar10
from collections import Counter

import datetime
timestamp = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d")

IMAGENET_PATH=""
NUM_LABELS=10
NUM_CHANNELS = 3
SIZE = 32

# SESSION INITIALIZATION
config_sess = tf.ConfigProto()
config_sess.gpu_options.allow_growth = True
# config_sess.gpu_options.per_process_gpu_memory_fraction = 0.5
# sess = tf.Session(config=config_sess)
sess = tf.InteractiveSession(config=config_sess)
# sess = tf.InteractiveSession()

def main(args, gpus):
    # SESSION INITIALIZATION
    # config_sess = tf.ConfigProto()
    # config_sess.gpu_options.allow_growth = True
    # config_sess.gpu_options.per_process_gpu_memory_fraction = 0.5
    # sess = tf.Session(config=config_sess)
    # sess = tf.InteractiveSession(config=config_sess)
    # sess = tf.InteractiveSession()

    # INITIAL IMAGE AND CLASS SELECTION
    (_, _), (x_test, y_test) = cifar10.load_data()

    batch_size = args.batch_size
    out_dir = os.path.join(args.out_dir, timestamp, '')
    # out_dir = args.out_dir
    print(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    epsilon = args.epsilon

    batch_per_gpu = batch_size // len(gpus)
    log_iters = args.log_iters
    current_lr = args.learning_rate
    queries_per_iter = args.samples_per_draw
    max_iters = int(np.ceil(args.max_queries // queries_per_iter))
    max_lr = args.max_lr
    # ----- partial info params -----
    k = args.top_k
    goal_epsilon = epsilon
    adv_thresh = args.adv_thresh

    # ----- label only params -----
    label_only = args.label_only
    zero_iters = args.zero_iters

    img_shape = (SIZE, SIZE, NUM_CHANNELS)

    x = tf.placeholder(tf.float32, img_shape)
    target_class_ph = tf.placeholder(tf.int64, shape=())

    eval_logits, eval_preds = model(sess, tf.expand_dims(x, 0))
    eval_percent_adv = tf.equal(eval_preds[0], target_class_ph)

    # TENSORBOARD SETUP
    # empirical_loss = tf.placeholder(dtype=tf.float32, shape=())
    # lr_placeholder = tf.placeholder(dtype=tf.float32, shape=())
    # loss_vs_queries = tf.summary.scalar('empirical loss vs queries', empirical_loss)
    # loss_vs_steps = tf.summary.scalar('empirical loss vs step', empirical_loss)
    # lr_vs_queries = tf.summary.scalar('lr vs queries', lr_placeholder)
    # lr_vs_steps = tf.summary.scalar('lr vs step', lr_placeholder)
    # writer = tf.summary.FileWriter(out_dir, graph=sess.graph)
    log_file = open(os.path.join(out_dir, 'log.txt'), 'w+')
    with open(os.path.join(out_dir, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)

    labels_ph = tf.placeholder(tf.float32, shape=(batch_per_gpu, NUM_LABELS))
    # LOSS FUNCTION
    def standard_loss(eval_points, noise):
        logits, preds = model(sess, eval_points)
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph)
        return losses, noise, None

    conf_est_strat = est_strats.get_strat(args.est_strat, args.strat_param)
    def label_only_loss(eval_points, noise):
        noised_eval_points = tf.zeros((batch_per_gpu,))
        # tiled_points = tf.tile(tf.expand_dims(eval_points, 0), [zero_iters,1,1,1,1])
        # noised_eval_im = tiled_points + \
        #         tf.random_uniform(tf.shape(tiled_points), minval=-1, \
        #         maxval=1)*args.label_only_sigma
        noised_eval_im = conf_est_strat.generate_samples(eval_points, zero_iters, img_shape)
        reshaped_noised_eval_im = tf.reshape(noised_eval_im, (-1,) + img_shape)

        logits, preds = model(sess, reshaped_noised_eval_im)
        vals, inds = tf.nn.top_k(logits, k=k)
        real_inds = tf.reshape(inds, (zero_iters, batch_per_gpu, -1))
        rank_range = tf.range(start=k, limit=0, delta=-1, dtype=tf.float32)
        tiled_rank_range = tf.tile(tf.reshape(rank_range, (1, 1, k)), [zero_iters, batch_per_gpu, 1])
        batches_in = tf.where(tf.equal(real_inds, target_class_ph),
                tiled_rank_range, tf.zeros(tf.shape(tiled_rank_range)))

        # print("[debug]", "lo_tiled", tiled_points.shape, "lo_logits",
        #       logits.shape, "inds", inds.shape, "real_inds", real_inds.shape,
        #       "rank_range", rank_range.shape, "tiled_rank_range", tiled_rank_range.shape,
        #       "batches_in", batches_in.shape)
        return 1 - tf.reduce_mean(batches_in, [0, 2]), noise, reshaped_noised_eval_im

    def partial_info_loss(eval_points, noise):
        logits, preds = model(sess, eval_points)
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph)
        vals, inds = tf.nn.top_k(logits, k=k)
        # inds is batch_size x k
        good_inds = tf.where(tf.equal(inds, tf.constant(target_class_ph))) # returns (# true) x 3
        good_images = good_inds[:,0] # inds of img in batch that worked
        losses = tf.gather(losses, good_images)
        noise = tf.gather(noise, good_images)
        return losses, noise, None

    # GRADIENT ESTIMATION GRAPH
    grad_estimates = []
    final_losses = []
    all_eval_points = []
    loss_fn = label_only_loss if label_only else \
                (partial_info_loss if k < NUM_LABELS else standard_loss)
    for img_index, device in enumerate(gpus):
        with tf.device(device):
            # print('loading on gpu %d of %d' % (img_index+1, len(gpus)))
            noise_pos = tf.random_normal((batch_per_gpu//2,) + img_shape)
            noise = tf.concat([noise_pos, -noise_pos], axis=0)
            eval_points = x + args.sigma * noise
            # print("Eval points shape:", eval_points.shape)
            losses, noise, noised_eval_points = loss_fn(eval_points, noise)
        losses_tiled = tf.tile(tf.reshape(losses, (-1, 1, 1, 1)), (1,) + img_shape)
        grad_estimates.append(tf.reduce_mean(losses_tiled * noise, axis=0)/args.sigma)
        final_losses.append(losses)
        all_eval_points.append(noised_eval_points)
        print("[debug]", "eval_points", eval_points.shape, "losses", losses.shape, "losses_tiled", losses_tiled.shape)
    grad_estimate = tf.reduce_mean(grad_estimates, axis=0)
    final_losses = tf.concat(final_losses, axis=0)
    all_eval_points = tf.concat(all_eval_points, axis=0)


    # GRADIENT ESTIMATION EVAL
    def get_grad(pt, spd, bs, target_class):
        num_batches = spd // bs
        losses = []
        grads = []
        points = []
        feed_dict = {x: pt, target_class_ph: target_class}
        for _ in range(num_batches):
            loss, dl_dx_, pts = sess.run([final_losses, grad_estimate, all_eval_points], feed_dict)
            losses.append(np.mean(loss))
            grads.append(dl_dx_)
            points.append(pts)
        return np.array(losses).mean(), np.mean(np.array(grads), axis=0), np.concatenate(points, axis=0)

    # CONCURRENT VISUALIZATION
    if args.visualize:
        with tf.device('/cpu:0'):
            render_feed = tf.placeholder(tf.float32, img_shape)
            render_exp = tf.expand_dims(render_feed, axis=0)
            render_logits, _ = model(sess, render_exp)

    assert out_dir[-1] == '/'

    # STEP CONDITION (important for partial-info attacks)
    def robust_in_top_k(t_, prop_adv_,k_):
        if k == NUM_LABELS:
            return True
        for i in range(1):
            n = np.random.rand(*prop_adv_.shape)*args.sigma
            eval_logits_ = sess.run(eval_logits, {x: prop_adv_})[0]
            if not t_ in eval_logits_.argsort()[-k_:][::-1]:
                return False
        return True
        

    # queries = []
    # query_distances = []
    # current_query, prev_query = adv, prev_adv

    # Debugging
    def debug_time(start_time, comment):
        duration = (time.time() - start_time) / 60.
        # print("[debug]", "time %.4f" % duration, comment)


    def run_attack(args):
        # HISTORY VARIABLES (for backtracking and momentum)
        num_queries = 0
        g = 0
        # prev_adv = adv
        last_ls = []

        # Detector
        run_detection = not args.no_detection
        detector = detection.ExperimentDetectors(active=run_detection)

        specified_imgs = args.img_index is not None and args.target_img_index is not None
        img_index = np.random.randint(0, x_test.shape[0]) if not specified_imgs else args.img_index

        if args.img_path:
            initial_img = np.asarray(Image.open(args.img_path).resize((SIZE, SIZE)))
            orig_class = args.orig_class
            initial_img = initial_img.astype(np.float32) / 255.0
        else:
            x, y = x_test[img_index, None][0], y_test[img_index][0]
            orig_class = y
            initial_img = x
            initial_img = initial_img.astype(np.float32) / 255.0
            print('[info] chose test image (%d) of class (%d)' % (img_index, orig_class))

        # PARAMETER SETUP
        if args.target_class is None and not specified_imgs:
            target_class = pseudorandom_target(img_index, NUM_LABELS, orig_class)
            print('[info] chose pseudorandom target class: %d' % target_class)
        else:
            target_class = args.target_class

        original_i = img_index

        lower = np.clip(initial_img - args.epsilon, 0., 1.)
        upper = np.clip(initial_img + args.epsilon, 0., 1.)
        adv = initial_img.copy() if not args.restore else \
            np.clip(np.load(args.restore), lower, upper)

        target_i = -1
        k = args.top_k
        if k > 0:
            if target_class == -1:
                raise ValueError("Partial-information attack is a targeted attack.")
            # adv = image_of_class(target_class, IMAGENET_PATH)
            if specified_imgs:
                target_i = args.target_img_index
                adv = x_test[target_i, None][0]
            else:
                mask = (y_test == target_class).flatten()
                x_test_target_class = x_test[mask]

                target_i = np.random.randint(0, x_test_target_class.shape[0])
                adv = x_test_target_class[target_i, None][0]

            adv = adv.astype(np.float32) / 255.0
            epsilon = args.starting_eps
            delta_epsilon = args.starting_delta_eps
        else:
            k = NUM_LABELS

        print("[info] original_i=%d, target_i=%d" % (original_i, target_i))

        # TARGET CLASS SELECTION
        if target_class < 0:
            one_hot_vec = one_hot(orig_class, NUM_LABELS)
        else:
            one_hot_vec = one_hot(target_class, NUM_LABELS)
        labels = np.repeat(np.expand_dims(one_hot_vec, axis=0),
                           repeats=batch_per_gpu, axis=0)
        is_targeted = 1 if target_class >= 0 else -1

        # MAIN LOOP
        # cur_query_adv, prev_query_adv = adv, prev_adv

        success, retval, info = False, args.max_queries, (timestamp, original_i, target_i, orig_class, target_class)
        distortion = None
        advs = []
        entire_start = time.time()
        for img_index in range(max_iters):
            start = time.time()
            if args.visualize:
                render_frame(sess, adv, img_index, render_logits, render_feed, out_dir)

            # CHECK IF WE SHOULD STOP
            temp_t = time.time()
            padv = sess.run(eval_percent_adv, feed_dict={x: adv, target_class_ph: target_class})
            # print("Processing adv single...")
            detector.process_query(adv, num_queries)
            # print("Finished...")
            debug_time(temp_t, "Single query check/detect adv")
            if padv == 1 and epsilon <= goal_epsilon:
                print('[log] early stopping at iteration %d, num queries %d' % (img_index, num_queries))
                success, retval = True, num_queries
                distortion = np.linalg.norm(initial_img - adv)
                break

            prev_g = g
            # print("Estimating the gradient...")
            temp_t = time.time()
            l, g, queries = get_grad(adv, args.samples_per_draw, batch_size, target_class)
            # print("Finished...")
            debug_time(temp_t, "Estimate gradient")

            # Detection
            temp_t = time.time()
            # print("Processing grad est queries...")
            detector.process(queries, num_queries)
            # print("Finished...")
            debug_time(temp_t, "Process grad est queries")

            temp_t = time.time()

            # SIMPLE MOMENTUM
            g = args.momentum * prev_g + (1.0 - args.momentum) * g

            # PLATEAU LR ANNEALING
            last_ls.append(l)
            last_ls = last_ls[-args.plateau_length:]
            if last_ls[-1] > last_ls[0] \
               and len(last_ls) == args.plateau_length:
                if max_lr > args.min_lr:
                    # print("[log] Annealing max_lr")
                    max_lr = max(max_lr / args.plateau_drop, args.min_lr)
                last_ls = []

            # SEARCH FOR LR AND EPSILON DECAY
            current_lr = max_lr
            proposed_adv = adv - is_targeted * current_lr * np.sign(g)
            prop_de = 0.0
            if l < adv_thresh and epsilon > goal_epsilon:
                prop_de = delta_epsilon
            while current_lr >= args.min_lr:
                # PARTIAL INFORMATION ONLY
                if k < NUM_LABELS:
                    proposed_epsilon = max(epsilon - prop_de, goal_epsilon)
                    lower = np.clip(initial_img - proposed_epsilon, 0, 1)
                    upper = np.clip(initial_img + proposed_epsilon, 0, 1)
                # GENERAL LINE SEARCH
                proposed_adv = adv - is_targeted * current_lr * np.sign(g)
                proposed_adv = np.clip(proposed_adv, lower, upper)
                num_queries += 1

                # Detection
                # print("Processing loop query...")
                detector.process_query(proposed_adv, num_queries)
                # print("Finished...")

                if robust_in_top_k(target_class, proposed_adv, k):
                    if prop_de > 0:
                        delta_epsilon = max(prop_de, 0.1)
                        last_ls = []

                    # prev_adv = adv
                    adv = proposed_adv
                    epsilon = max(epsilon - prop_de/args.conservative, goal_epsilon)

                    break
                elif current_lr >= args.min_lr*2:
                    current_lr = current_lr / 2
                    #print("[log] backtracking lr to %3f" % (current_lr,))
                else:
                    prop_de = prop_de / 2
                    if prop_de == 0:
                        # raise ValueError("Did not converge.")
                        print("[error] Did not converge!")
                        return False, -1, info, detector, distortion
                    if prop_de < 2e-3:
                        prop_de = 0
                    current_lr = max_lr
                    # if img_index % 50 == 0:
                    #     print("[log] backtracking eps to %3f" % (epsilon-prop_de,))

            # current_query, prev_query = adv, prev_adv

            debug_time(temp_t, "Parameter tuning/decay")
            # BOOK-KEEPING STUFF
            num_queries += args.samples_per_draw * (zero_iters if label_only else 1)

            log_text = 'Step %05d: loss %.4f lr %.2E eps %.3f (time %.4f)' % (img_index, l, \
                            current_lr, epsilon, time.time() - start)
            log_file.write(log_text + '\n')
            if img_index % (max_iters // 100) == 0:
                print(log_text)

            # if img_index % log_iters == 0:
            #     lvq, lvs, lrvq, lrvs = sess.run([loss_vs_queries, loss_vs_steps,
            #                                      lr_vs_queries, lr_vs_steps], {
            #                                          empirical_loss:l,
            #                                          lr_placeholder:current_lr
            #                                      })
            #     writer.add_summary(lvq, num_queries)
            #     writer.add_summary(lrvq, num_queries)
            #     writer.add_summary(lvs, img_index)
            #     writer.add_summary(lrvs, img_index)

            # if (img_index+1) % args.save_iters == 0 and args.save_iters > 0:
            #     np.save(os.path.join(out_dir, '%s.npy' % (img_index+1)), adv)
            #     scipy.misc.imsave(os.path.join(out_dir, '%s.png' % (img_index+1)), adv)

        # print("[error] hit max iterations")
        #     if img_index < 100:
        #         advs.append(adv)
        #
        # np.savez_compressed(os.path.join(out_dir, "trajectory_%d_to_%d.npz" % (original_i, target_i)), np.array(advs))

        # print("Average query distance:", np.mean(query_distances))

        log_output(sess, eval_logits, eval_preds, x, adv, initial_img, \
                target_class, out_dir, orig_class, num_queries)

        # import datetime
        # timestamp = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d_%H%M")
        # np.savez("query_distances_i%d_%d_o%d_t%d_iters_%d_%d_%s" % (original_i, target_i, orig_class, target_class, img_index, max_iters, timestamp), dists=query_distances)

        print("[detection]: params=%d,%f; num_queries=%d, result=%s,%d" % (args.zero_iters, args.strat_param, num_queries, str(success), retval))
        num_detections = [(d_name, len(detector.get_detections())) for d_name, detector in detector.detectors.items()]
        print("[detection]: num_queries",  num_queries, "num_detections", num_detections)
        return success, retval, info, detector, distortion

    def run_experiment(args, all_results, main_results, detection_results, results_s, key, idx_range, img_idxs):
        num_iters = []
        results = []
        infos = []
        detections_success = {}
        detections_failure = {}
        l2_distortions = []
        num_detections_success = {}
        num_detections_failure = {}

        imgs_specified = args.img_idxs is not None
        s = args.num_exp_per_param // 20 if args.num_exp_per_param > 10 else 2

        for i in idx_range:
            # if i % s == 0:

            if imgs_specified:
                orig_i, target_i, target_class = img_idxs[i]
                args.img_index, args.target_img_index = int(orig_i), int(target_i)
                args.target_class = int(target_class)

            success, retval, info, detectors, distortion = run_attack(args)

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
        result_s = " ".join(
            [str(key), "\t", str(np.mean(num_iters)), str(np.median(num_iters)), str(np.std(num_iters)), json.dumps(c)])
        results_s.append(result_s)
        print(result_s)
        all_results[key] = (results, num_iters, l2_distortions, infos)

        main_det_results = {}
        for d_name in detections_success:
            detection_results[key + (d_name,)] = (detections_success[d_name], detections_failure[d_name])

            mean_detections_success = np.mean(num_detections_success[d_name])
            mean_detections_failure = np.mean(num_detections_failure[d_name])
            mean_detections = np.mean(num_detections_success[d_name] + num_detections_failure[d_name])
            main_det_results[d_name] = (mean_detections, mean_detections_success, mean_detections_failure)

        main_results[key] = (success_rate, mean_distortion, mean_iters, main_det_results)

        print("Summary so far:")
        for result_s in results_s:
            print(result_s)

if __name__ == '__main__':
    main()
