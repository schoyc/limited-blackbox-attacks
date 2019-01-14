from PIL import Image
import numpy as np
import tensorflow as tf

# SESSION INITIALIZATION
config_sess = tf.ConfigProto()
config_sess.gpu_options.allow_growth = True
config_sess.gpu_options.per_process_gpu_memory_fraction = 0.5
# sess = tf.Session(config=config_sess)
sess = tf.InteractiveSession(config=config_sess)


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
from tools.tf_sample_cifar10 import model
from tensorflow.keras.datasets import cifar10

import datetime
timestamp = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d_%H%M")

IMAGENET_PATH=""
NUM_LABELS=10
SIZE = 32

def main(args, gpus):
    # INITIAL IMAGE AND CLASS SELECTION
    (_, _), (x_test, y_test) = cifar10.load_data()

    specified_imgs = args.img_index is not None and args.target_img_index is not None
    img_index = np.random.randint(0, x_test.shape[0]) if specified_imgs else args.img_index

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

    batch_size = args.batch_size
    out_dir = os.path.join(args.out_dir, timestamp, '')
    print(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    epsilon = args.epsilon
    lower = np.clip(initial_img - args.epsilon, 0., 1.)
    upper = np.clip(initial_img + args.epsilon, 0., 1.)
    adv = initial_img.copy() if not args.restore else \
            np.clip(np.load(args.restore), lower, upper)
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
    target_i = -1
    if k > 0:
        if target_class == -1:
            raise ValueError("Partial-information attack is a targeted attack.")
        # adv = image_of_class(target_class, IMAGENET_PATH)

        mask = (y_test == target_class).flatten()
        x_test_target_class = x_test[mask]
        target_i = np.random.randint(0, x_test_target_class.shape[0]) if not specified_imgs else args.target_img_index
        adv = x_test_target_class[target_i, None][0]
        adv = adv.astype(np.float32) / 255.0
        epsilon = args.starting_eps
        delta_epsilon = args.starting_delta_eps
    else:
        k = NUM_LABELS
    # ----- label only params -----
    label_only = args.label_only
    zero_iters = args.zero_iters

    print("[info] original_i=%d, target_i=%d" % (original_i, target_i))

    # TARGET CLASS SELECTION
    if target_class < 0:
        one_hot_vec = one_hot(orig_class, NUM_LABELS)
    else:
        one_hot_vec = one_hot(target_class, NUM_LABELS)
    labels = np.repeat(np.expand_dims(one_hot_vec, axis=0),
                       repeats=batch_per_gpu, axis=0)
    is_targeted = 1 if target_class >= 0 else -1

    # QUERY DISTANCE RECORDING
    adversarial_query_dists = []
    conf_est_query_dists = []
    grad_est_query_dists = []

    def query_dist(cur_query, prev_query):
        # RECORD DISTANCE BETWEEN QUERIES
        l2_dist = np.linalg.norm(cur_query - prev_query)
        # print("[log] distance from prev. query: %d" % l2_dist)
        return l2_dist


    x = tf.placeholder(tf.float32, initial_img.shape)
    eval_logits, eval_preds = model(sess, tf.expand_dims(x, 0))
    eval_percent_adv = tf.equal(eval_preds[0], tf.constant(target_class, tf.int64))

    # TENSORBOARD SETUP
    empirical_loss = tf.placeholder(dtype=tf.float32, shape=())
    lr_placeholder = tf.placeholder(dtype=tf.float32, shape=())
    loss_vs_queries = tf.summary.scalar('empirical loss vs queries', empirical_loss)
    loss_vs_steps = tf.summary.scalar('empirical loss vs step', empirical_loss)
    lr_vs_queries = tf.summary.scalar('lr vs queries', lr_placeholder)
    lr_vs_steps = tf.summary.scalar('lr vs step', lr_placeholder)
    writer = tf.summary.FileWriter(out_dir, graph=sess.graph)
    log_file = open(os.path.join(out_dir, 'log.txt'), 'w+')
    with open(os.path.join(out_dir, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)

    # LOSS FUNCTION
    def standard_loss(eval_points, noise):
        logits, preds = model(sess, eval_points)
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        return losses, noise, None

    conf_est_strat = est_strats.get_strat(args.est_strat, args.strat_param)
    def label_only_loss(eval_points, noise):
        noised_eval_points = tf.zeros((batch_per_gpu,))
        # tiled_points = tf.tile(tf.expand_dims(eval_points, 0), [zero_iters,1,1,1,1])
        # noised_eval_im = tiled_points + \
        #         tf.random_uniform(tf.shape(tiled_points), minval=-1, \
        #         maxval=1)*args.label_only_sigma
        noised_eval_im = conf_est_strat.generate_samples(eval_points, zero_iters, initial_img.shape)
        reshaped_noised_eval_im = tf.reshape(noised_eval_im, (-1,) + initial_img.shape)

        logits, preds = model(sess, reshaped_noised_eval_im)
        vals, inds = tf.nn.top_k(logits, k=k)
        real_inds = tf.reshape(inds, (zero_iters, batch_per_gpu, -1))
        rank_range = tf.range(start=k, limit=0, delta=-1, dtype=tf.float32)
        tiled_rank_range = tf.tile(tf.reshape(rank_range, (1, 1, k)), [zero_iters, batch_per_gpu, 1])
        batches_in = tf.where(tf.equal(real_inds, target_class), 
                tiled_rank_range, tf.zeros(tf.shape(tiled_rank_range)))

        # print("[debug]", "lo_tiled", tiled_points.shape, "lo_logits",
        #       logits.shape, "inds", inds.shape, "real_inds", real_inds.shape,
        #       "rank_range", rank_range.shape, "tiled_rank_range", tiled_rank_range.shape,
        #       "batches_in", batches_in.shape)
        return 1 - tf.reduce_mean(batches_in, [0, 2]), noise, reshaped_noised_eval_im

    def partial_info_loss(eval_points, noise):
        logits, preds = model(sess, eval_points)
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        vals, inds = tf.nn.top_k(logits, k=k)
        # inds is batch_size x k
        good_inds = tf.where(tf.equal(inds, tf.constant(target_class))) # returns (# true) x 3
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
            noise_pos = tf.random_normal((batch_per_gpu//2,) + initial_img.shape)
            noise = tf.concat([noise_pos, -noise_pos], axis=0)
            eval_points = x + args.sigma * noise
            # print("Eval points shape:", eval_points.shape)
            losses, noise, noised_eval_points = loss_fn(eval_points, noise)
        losses_tiled = tf.tile(tf.reshape(losses, (-1, 1, 1, 1)), (1,) + initial_img.shape)
        grad_estimates.append(tf.reduce_mean(losses_tiled * noise, axis=0)/args.sigma)
        final_losses.append(losses)
        all_eval_points.append(noised_eval_points)
        print("[debug]", "eval_points", eval_points.shape, "losses", losses.shape, "losses_tiled", losses_tiled.shape)
    grad_estimate = tf.reduce_mean(grad_estimates, axis=0)
    final_losses = tf.concat(final_losses, axis=0)
    all_eval_points = tf.concat(all_eval_points, axis=0)


    # GRADIENT ESTIMATION EVAL
    def get_grad(pt, spd, bs):
        num_batches = spd // bs
        losses = []
        grads = []
        points = []
        feed_dict = {x: pt}
        for _ in range(num_batches):
            loss, dl_dx_, pts = sess.run([final_losses, grad_estimate, all_eval_points], feed_dict)
            losses.append(np.mean(loss))
            grads.append(dl_dx_)
            points.append(points)
        return np.array(losses).mean(), np.mean(np.array(grads), axis=0), np.concatenate(points, axis=0)

    # CONCURRENT VISUALIZATION
    if args.visualize:
        with tf.device('/cpu:0'):
            render_feed = tf.placeholder(tf.float32, initial_img.shape)
            render_exp = tf.expand_dims(render_feed, axis=0)
            render_logits, _ = model(sess, render_exp)

    assert out_dir[-1] == '/'

    # HISTORY VARIABLES (for backtracking and momentum)
    num_queries = 0
    g = 0
    prev_adv = adv
    last_ls = []

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
        

    queries = []
    query_distances = []
    current_query, prev_query = adv, prev_adv

    # Detector
    detector = detection.Detector(threshold=1.44, K=50)

    # MAIN LOOP
    cur_query_adv, prev_query_adv = adv, prev_adv
    success, retval, info = False, args.max_queries, (timestamp, original_i, target_i, orig_class, target_class)
    advs = []
    entire_start = time.time()
    for img_index in range(max_iters):
        start = time.time()
        if args.visualize:
            render_frame(sess, adv, img_index, render_logits, render_feed, out_dir)

        # Record query distance
        prev_query_adv = cur_query_adv
        cur_query_adv = adv
        d = query_dist(cur_query_adv, prev_query_adv)
        adversarial_query_dists.append(d)
        # print("[info] query dist:", d)

        # CHECK IF WE SHOULD STOP
        padv = sess.run(eval_percent_adv, feed_dict={x: adv})
        print("Processing adv single...")
        detector.process_query(adv, num_queries)
        print("Finished...")
        if padv == 1 and epsilon <= goal_epsilon:
            print('[log] early stopping at iteration %d, num queries %d' % (img_index, num_queries))
            success, retval = True, num_queries
            break

        prev_g = g
        l, g, queries = get_grad(adv, args.samples_per_draw, batch_size)

        # Detection
        print("Processing grad est queries...")
        detector.process(queries, num_queries)
        print("Finished...")

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

            # Record query distance
            prev_query_adv = cur_query_adv
            cur_query_adv = proposed_adv
            adversarial_query_dists.append(query_dist(cur_query_adv, prev_query_adv))

            # Detection
            print("Processing loop query...")
            detector.process_query(proposed_adv, num_queries)
            print("Finished...")

            if robust_in_top_k(target_class, proposed_adv, k):
                if prop_de > 0:
                    delta_epsilon = max(prop_de, 0.1)
                    last_ls = []

                prev_adv = adv
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
                    return False, -1, info
                if prop_de < 2e-3:
                    prop_de = 0
                current_lr = max_lr
                # if img_index % 50 == 0:
                #     print("[log] backtracking eps to %3f" % (epsilon-prop_de,))

        current_query, prev_query = adv, prev_adv

        # BOOK-KEEPING STUFF
        num_queries += args.samples_per_draw * (zero_iters if label_only else 1)

        log_text = 'Step %05d: loss %.4f lr %.2E eps %.3f (time %.4f)' % (img_index, l, \
                        current_lr, epsilon, time.time() - start)
        log_file.write(log_text + '\n')
        if img_index % (max_iters // 10) == 0:
            print(log_text)

        if img_index % log_iters == 0:
            lvq, lvs, lrvq, lrvs = sess.run([loss_vs_queries, loss_vs_steps,
                                             lr_vs_queries, lr_vs_steps], {
                                                 empirical_loss:l,
                                                 lr_placeholder:current_lr
                                             })
            writer.add_summary(lvq, num_queries)
            writer.add_summary(lrvq, num_queries)
            writer.add_summary(lvs, img_index)
            writer.add_summary(lrvs, img_index)

        if (img_index+1) % args.save_iters == 0 and args.save_iters > 0:
            np.save(os.path.join(out_dir, '%s.npy' % (img_index+1)), adv)
            scipy.misc.imsave(os.path.join(out_dir, '%s.png' % (img_index+1)), adv)

    print("[error] hit max iterations")
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
    print("[detection]:", num_queries, str(detector.history))
    return success, retval, info

if __name__ == '__main__':
    main()
