# -*- coding: utf-8 -*-
"""Training script for the WaveNet network on the VCTK corpus.

This script trains a network with the WaveNet using data from the VCTK corpus,
which can be freely downloaded at the following site (~10 GB):
http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import argparse
from datetime import datetime
import json
import os
import sys
import time

import tensorflow as tf
from tensorflow.python.client import timeline

from wavenet import WaveNetModel, AudioReader, optimizer_factory
from wavenet.util.visual import figure_joint_skeleton

from analysis.Corr_Dim import fnn, Tao, Dim_Corr
from analysis.Fourier_utils import *

BATCH_SIZE = 1
DATA_DIRECTORY = './VCTK-Corpus'
LOGDIR_ROOT = './logdir'
CHECKPOINT_EVERY = 500
NUM_STEPS = int(1e5)
LEARNING_RATE = 1e-3
para_id = 2
WAVENET_PARAMS = './RNN_params/rnnnet_params'+str(para_id)+'.json'
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SAMPLE_SIZE = None
L2_REGULARIZATION_STRENGTH = 0
SILENCE_THRESHOLD = 0.3
EPSILON = 0.001
MOMENTUM = 0.9
MAX_TO_KEEP = 5
METADATA = False
def figure_hand_back(uvd_pt,uvd_pt1,uvd_pt2,path,test_num):
    #uvd_pt = np.reshape(uvd_pt, (20, 3))
    uvd_pt = uvd_pt.reshape(-1, 3)
    uvd_pt1 = uvd_pt1.reshape(-1, 3)
    uvd_pt2 = uvd_pt2.reshape(-1, 3)
    fig = plt.figure(1)
    fig.clear()
    ax = plt.subplot(111, projection='3d')

    fig_color = ['c', 'm', 'y', 'g', 'r']

    ax.scatter(uvd_pt[0, 0], uvd_pt[0, 1], uvd_pt[0, 2], s=10, c='b')
    ax.scatter(uvd_pt[1, 0], uvd_pt[1, 1], uvd_pt[1, 2], s=10, c='b')
    ax.scatter(uvd_pt[2, 0], uvd_pt[2, 1], uvd_pt[2, 2], s=10, c='b')

    ax.plot([uvd_pt[0, 0], uvd_pt[1, 0]],
            [uvd_pt[0, 1], uvd_pt[1, 1]],
            [uvd_pt[0, 2], uvd_pt[1, 2]], color='b', linewidth=1)
    ax.plot([uvd_pt[1, 0], uvd_pt[2, 0]],
            [uvd_pt[1, 1], uvd_pt[2, 1]],
            [uvd_pt[1, 2], uvd_pt[2, 2]], color='b', linewidth=1)
    ax.plot([uvd_pt[2, 0], uvd_pt[0, 0]],
            [uvd_pt[2, 1], uvd_pt[0, 1]],
            [uvd_pt[2, 2], uvd_pt[0, 2]], color='b', linewidth=1)

    plt.ylim(-300, 300)
    plt.xlim(-300, 300)
    ax.set_zlim(-300, 300)

    ax.scatter(uvd_pt1[0, 0], uvd_pt1[0, 1], uvd_pt1[0, 2], s=10, c='g')
    ax.scatter(uvd_pt1[1, 0], uvd_pt1[1, 1], uvd_pt1[1, 2], s=10, c='g')
    ax.scatter(uvd_pt1[2, 0], uvd_pt1[2, 1], uvd_pt1[2, 2], s=10, c='g')

    ax.plot([uvd_pt1[0, 0], uvd_pt1[1, 0]],
            [uvd_pt1[0, 1], uvd_pt1[1, 1]],
            [uvd_pt1[0, 2], uvd_pt1[1, 2]], color='g', linewidth=1)
    ax.plot([uvd_pt1[1, 0], uvd_pt1[2, 0]],
            [uvd_pt1[1, 1], uvd_pt1[2, 1]],
            [uvd_pt1[1, 2], uvd_pt1[2, 2]], color='g', linewidth=1)
    ax.plot([uvd_pt1[2, 0], uvd_pt1[0, 0]],
            [uvd_pt1[2, 1], uvd_pt1[0, 1]],
            [uvd_pt1[2, 2], uvd_pt1[0, 2]], color='g', linewidth=1)

    ax.scatter(uvd_pt2[0, 0], uvd_pt2[0, 1], uvd_pt2[0, 2], s=10, c='r')
    ax.scatter(uvd_pt2[1, 0], uvd_pt2[1, 1], uvd_pt2[1, 2], s=10, c='r')
    ax.scatter(uvd_pt2[2, 0], uvd_pt2[2, 1], uvd_pt2[2, 2], s=10, c='r')

    ax.plot([uvd_pt2[0, 0], uvd_pt2[1, 0]],
            [uvd_pt2[0, 1], uvd_pt2[1, 1]],
            [uvd_pt2[0, 2], uvd_pt2[1, 2]], color='r', linewidth=1)
    ax.plot([uvd_pt2[1, 0], uvd_pt2[2, 0]],
            [uvd_pt2[1, 1], uvd_pt2[2, 1]],
            [uvd_pt2[1, 2], uvd_pt2[2, 2]], color='r', linewidth=1)
    ax.plot([uvd_pt2[2, 0], uvd_pt2[0, 0]],
            [uvd_pt2[2, 1], uvd_pt2[0, 1]],
            [uvd_pt2[2, 2], uvd_pt2[0, 2]], color='r', linewidth=1)

    plt.savefig(path+str(test_num).zfill(7)+".png")

def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many wav files to process at once. Default: ' + str(BATCH_SIZE) + '.')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the VCTK corpus.')
    parser.add_argument('--store_metadata', type=bool, default=METADATA,
                        help='Whether to store advanced debugging information '
                        '(execution time, memory consumption) for use with '
                        'TensorBoard. Default: ' + str(METADATA) + '.')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory in which to store the logging '
                        'information for TensorBoard. '
                        'If the model already exists, it will restore '
                        'the state and will continue training. '
                        'Cannot use with --logdir_root and --restore_from.')
    parser.add_argument('--logdir_root', type=str, default=None,
                        help='Root directory to place the logging '
                        'output and generated model. These are stored '
                        'under the dated subdirectory of --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Directory in which to restore the model from. '
                        'This creates the new model under the dated directory '
                        'in --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--checkpoint_every', type=int,
                        default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after. Default: ' + str(CHECKPOINT_EVERY) + '.')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps. Default: ' + str(NUM_STEPS) + '.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training. Default: ' + str(LEARNING_RATE) + '.')
    parser.add_argument('--wavenet_params', type=str, default=WAVENET_PARAMS,
                        help='JSON file with the network parameters. Default: ' + WAVENET_PARAMS + '.')
    parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE,
                        help='Concatenate and cut audio samples to this many '
                        'samples. Default: ' + str(SAMPLE_SIZE) + '.')
    parser.add_argument('--l2_regularization_strength', type=float,
                        default=L2_REGULARIZATION_STRENGTH,
                        help='Coefficient in the L2 regularization. '
                        'Default: False')
    parser.add_argument('--silence_threshold', type=float,
                        default=SILENCE_THRESHOLD,
                        help='Volume threshold below which to trim the start '
                        'and the end from the training set samples. Default: ' + str(SILENCE_THRESHOLD) + '.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=optimizer_factory.keys(),
                        help='Select the optimizer specified by this option. Default: adam.')
    parser.add_argument('--momentum', type=float,
                        default=MOMENTUM, help='Specify the momentum to be '
                        'used by sgd or rmsprop optimizer. Ignored by the '
                        'adam optimizer. Default: ' + str(MOMENTUM) + '.')
    parser.add_argument('--histograms', type=_str_to_bool, default=False,
                        help='Whether to store histogram summaries. Default: False')
    parser.add_argument('--gc_channels', type=int, default=None,
                        help='Number of global condition channels. Default: None. Expecting: Int')
    parser.add_argument('--max_checkpoints', type=int, default=MAX_TO_KEEP,
                        help='Maximum amount of checkpoints that will be kept alive. Default: '
                             + str(MAX_TO_KEEP) + '.')
    return parser.parse_args()


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def get_default_logdir(logdir_root):
    logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING)
    return logdir


def validate_directories(args):
    """Validate and arrange directory related arguments."""

    # Validation
    if args.logdir and args.logdir_root:
        raise ValueError("--logdir and --logdir_root cannot be "
                         "specified at the same time.")

    if args.logdir and args.restore_from:
        raise ValueError(
            "--logdir and --restore_from cannot be specified at the same "
            "time. This is to keep your previous model from unexpected "
            "overwrites.\n"
            "Use --logdir_root to specify the root of the directory which "
            "will be automatically created with current date and time, or use "
            "only --logdir to just continue the training from the last "
            "checkpoint.")

    # Arrangement
    logdir_root = args.logdir_root
    if logdir_root is None:
        logdir_root = LOGDIR_ROOT

    logdir = args.logdir
    if logdir is None:
        logdir = get_default_logdir(logdir_root)+'_par' + str(para_id)
        print('Using default logdir: {}'.format(logdir))

    restore_from = args.restore_from
    if restore_from is None:
        # args.logdir and args.restore_from are exclusive,
        # so it is guaranteed the logdir here is newly created.
        restore_from = logdir

    return {
        'logdir': logdir,
        'logdir_root': args.logdir_root,
        'restore_from': restore_from
    }

from rnnnet.model import Model

istest = True

def main():
    args = get_arguments()

    try:
        directories = validate_directories(args)
    except ValueError as e:
        print("Some arguments are wrong:")
        print(str(e))
        return

    logdir = directories['logdir']
    restore_from = directories['restore_from']

    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    is_overwritten_training = logdir != restore_from

    with open(args.wavenet_params, 'r') as f:
        wavenet_params = json.load(f)

    # Create coordinator.
    coord = tf.train.Coordinator()

    # Load raw waveform from VCTK corpus.
    with tf.name_scope('create_inputs'):
        # Allow silence trimming to be skipped by specifying a threshold near
        # zero.
        silence_threshold = args.silence_threshold if args.silence_threshold > \
                                                      EPSILON else None
        gc_enabled = args.gc_channels is not None
        reader = AudioReader(
            args.data_dir,
            coord,
            sample_rate=0,
            gc_enabled=gc_enabled,
            receptive_field=0,
            sample_size=args.sample_size,
            silence_threshold=silence_threshold)
        audio_batch = reader.dequeue(args.batch_size)
        if gc_enabled:
            gc_id_batch = reader.dequeue_gc(args.batch_size)
        else:
            gc_id_batch = None

    # Create network.
    model = Model(wavenet_params,input_batch=audio_batch)
    loss = model.loss
    # chen_test end
    optimizer = optimizer_factory[args.optimizer](
                    learning_rate=args.learning_rate,
                    momentum=args.momentum)
    trainable = tf.trainable_variables()

    optim = optimizer.minimize(loss, var_list=trainable)

    # Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()

    # Set up session
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    init = tf.global_variables_initializer()
    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=args.max_checkpoints)

    try:
        saved_global_step = load(saver, sess, restore_from)
        if is_overwritten_training or saved_global_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            saved_global_step = -1

    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)

    step = None
    last_saved_step = saved_global_step
    try:
        pass_loss = 0.0
        for step in range(saved_global_step + 1, args.num_steps):

            if istest:
                start_time = time.time()

                loss_value, output_value, label_value = sess.run([model.loss, model.logits, model.original_labels])
                label_value  = label_value[0,:,:]

                np.savetxt("/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/pakinson/shake_tra_test_result/RNN2/"
                           + "loss/" + str(step).zfill(5) + ".txt", np.array([loss_value]))
                np.savetxt("/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/pakinson/shake_tra_test_result/RNN2/"
                           + "result/" + str(step).zfill(5) + ".txt", output_value)
                np.savetxt("/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/pakinson/shake_tra_test_result/RNN2/"
                           + "target/" + str(step).zfill(5) + ".txt", label_value)

                duration = time.time() - start_time
                print('step {:d} - loss = {:.3f}, ({:.3f} sec/step)'
                      .format(step, loss_value, duration))
            else:
                start_time = time.time()
                if args.store_metadata and step % 5000 == 0:

                    # Slow run that stores extra information for debugging.
                    print('Storing metadata')
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    summary, loss_value, _ = sess.run(
                        [summaries, loss, optim],
                        options=run_options,
                        run_metadata=run_metadata)
                    writer.add_summary(summary, step)
                    writer.add_run_metadata(run_metadata,
                                            'step_{:04d}'.format(step))
                    tl = timeline.Timeline(run_metadata.step_stats)
                    timeline_path = os.path.join(logdir, 'timeline.trace')
                    with open(timeline_path, 'w') as f:
                        f.write(tl.generate_chrome_trace_format(show_memory=True))

                else:
                    # chen_test
                    #summary, loss_value, _, shape_input_batch_v,  shape_encoded_input_v,  shape_encoded_v, shape_network_input_v\
                    #    = sess.run([summaries, loss, optim, shape_input_batch,  shape_encoded_input,  shape_encoded, shape_network_input])
                    """
                    audio_batch_v, label_v, raw_output_v = sess.run([network_input, network_label,raw_output])
                    loss_value = 0
                    print(audio_batch_v.shape)
                    print(label_v.shape)
                    print(raw_output_v.shape)
                    onefile_data_pose_r = audio_batch_v[0, 52:, :]
                    shape = onefile_data_pose_r.shape
                    onefile_data_pose_r = onefile_data_pose_r.reshape(shape[0], -1, 3)
                    #onefile_data_pose_r = onefile_data_pose_r[:, 0:20, :]
                    for test_read_i in range(shape[0]):
                        figure_joint_skeleton(onefile_data_pose_r[test_read_i, :, :],
                        "/home/chen/Documents/tensorflow-wavenet-master/wavenet/test/image" + "/" + str(step) + "/", test_read_i)
                    """
                    loss_value, _ = sess.run([loss, optim])
                    #writer.add_summary(summary, step)

                    """
                    shape = prediction_v.shape
                    x = np.linspace(0, shape[0], shape[0])
                    y1, y2 = prediction_v[:, 2], target_output_v[:, 2]
                    plt.clf()
                    plt.plot(x, y1)
                    plt.plot(x, y2)
                    plt.ylim(-80, 80)
                    plt.xlim(-0, 500)
                    plt.title('Fitting chart')
                    plt.xlabel('time')
                    plt.ylabel('angel')
                    plt.savefig("/home/chen/Documents/tensorflow-wavenet-master/images/train_result/" + str(loss_value) + ".png")
                    """
                    # chen_test end

                duration = time.time() - start_time

                print('step {:d} - loss = {:.3f}, ({:.3f} sec/step)'
                      .format(step, loss_value, duration))
                # chen_test
                #print(shape_input_batch_v,  shape_encoded_input_v,  shape_encoded_v, shape_network_input_v)
                # chen_test end
                if step % args.checkpoint_every == 0:
                    save(saver, sess, logdir, step)
                    last_saved_step = step

    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
    finally:
        if step > last_saved_step:
            save(saver, sess, logdir, step)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()


