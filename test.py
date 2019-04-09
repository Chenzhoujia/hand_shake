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
import json,math
import os
import sys
import time
import matplotlib.pyplot as pl
import tensorflow as tf
from tensorflow.python.client import timeline

from wavenet import WaveNetModel, optimizer_factory, AudioReader
from wavenet.model import create_variable, create_bias_variable
from wavenet.util.visual import figure_joint_skeleton

from analysis.Corr_Dim import fnn, Tao, Dim_Corr
from analysis.Fourier_utils import *

BATCH_SIZE = 1
DATA_DIRECTORY = './VCTK-Corpus'
LOGDIR_ROOT = './logdir'
CHECKPOINT_EVERY = 500
NUM_STEPS = int(1e5)
LEARNING_RATE = 1e-4
WAVENET_PARAMS = ['./WAVE_params/wavenet_params_1.json','./WAVE_params/wavenet_params_2.json','./WAVE_params/wavenet_params_3.json']
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
    #parser.add_argument('--wavenet_params', type=list, default=WAVENET_PARAMS,
    #                    help='JSON file with the network parameters. Default: ' + WAVENET_PARAMS + '.')
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
        logdir = get_default_logdir(logdir_root)
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

istest = False
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

    wavenet_params = []
    for file_name in WAVENET_PARAMS:
        with open(file_name, 'r') as f:
            wavenet_params.append(json.load(f))

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
            sample_rate=wavenet_params[2]['sample_rate'],
            gc_enabled=gc_enabled,
            receptive_field=WaveNetModel.calculate_receptive_field(wavenet_params[2]["filter_width"],
                                                                    wavenet_params[2]["dilations"],
                                                                    wavenet_params[2]["scalar_input"],
                                                                    wavenet_params[2]["initial_filter_width"]),
            sample_size=args.sample_size,
            silence_threshold=silence_threshold,
            pad = False,
            path = "/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/pakinson/degree_dataset/3",
            rand = False)
        audio_batch = reader.dequeue(args.batch_size)
        audio_batch_file = reader.dequeue_str(args.batch_size)
        audio_batch = tf.squeeze(audio_batch)

        one_receptive_field = WaveNetModel.calculate_receptive_field(wavenet_params[2]["filter_width"],
                                                                     wavenet_params[2]["dilations"],
                                                                     wavenet_params[2]["scalar_input"],
                                                                     wavenet_params[2]["initial_filter_width"])

        audio_batch = tf.pad(audio_batch, [[one_receptive_field, 0], [0, 0]],
                       'constant')
        audio_batch = tf.expand_dims(audio_batch, 0)


        if gc_enabled:
            gc_id_batch = reader.dequeue_gc(args.batch_size)
        else:
            gc_id_batch = None

    # Create network.
    net = [WaveNetModel(
        batch_size=args.batch_size,
        dilations=one_params["dilations"],
        filter_width=one_params["filter_width"],
        residual_channels=one_params["residual_channels"],
        dilation_channels=one_params["dilation_channels"],
        skip_channels=one_params["skip_channels"],
        quantization_channels=one_params["quantization_channels"],
        use_biases=one_params["use_biases"],
        scalar_input=one_params["scalar_input"],
        initial_filter_width=one_params["initial_filter_width"],
        histograms=args.histograms,
        global_condition_channels=args.gc_channels,
        global_condition_cardinality=reader.gc_category_cardinality,
        namespace = str(one_params_i))

        for one_params_i,one_params in enumerate(wavenet_params)]

    post_par = []
    for one_params_i, one_params in enumerate(wavenet_params):
        with tf.variable_scope('postprocessing_'+'stage_id_'+str(one_params_i)):
            current = dict()
            current['postprocess1'] = create_variable(
                'postprocess1',
                [1, 64, 32])
            current['postprocess2'] = create_variable(
                'postprocess2',
                [1, 32, 3])

            current['postprocess1_bias'] = create_bias_variable(
                'postprocess1_bias',
                [32])
            current['postprocess2_bias'] = create_bias_variable(
                'postprocess2_bias',
                [3])
            post_par.append(current)

    if args.l2_regularization_strength == 0:
        args.l2_regularization_strength = None
    #compute
    loss_list = []
    optimizer = optimizer_factory[args.optimizer](
        learning_rate=args.learning_rate,
        momentum=args.momentum)
    optim_list = []
    raw_output_list = []
    audio_batch_list = []
    loss_all_list = []
    for one_params_i, _ in enumerate(wavenet_params):
        with tf.name_scope('stage_' + str(one_params_i) + '_postcompute'):

            if one_params_i==0:
                raw_output, network_label = net[one_params_i].pre_loss(input_batch=audio_batch,
                                                                        global_condition_batch=gc_id_batch,
                                                                        l2_regularization_strength=args.l2_regularization_strength)
                audio_batch_list.append(audio_batch)
            else:
                #将前一步骤的补偿结果作用在下一步的输入
                raw_output = tf.pad(raw_output, [[one_receptive_field-1, 0], [0, 0]])
                raw_output = tf.concat([raw_output, raw_output, raw_output, raw_output], axis=1)
                raw_output = tf.pad(raw_output, [[0, 0], [0, 6]])
                raw_output = tf.expand_dims(raw_output, 0)
                audio_batch = audio_batch - raw_output
                audio_batch_list.append(audio_batch)

                raw_output, network_label = net[one_params_i].pre_loss(input_batch=audio_batch,
                                                                        global_condition_batch=gc_id_batch,
                                                                        l2_regularization_strength=args.l2_regularization_strength)

            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            w1 = post_par[one_params_i]['postprocess1']
            w2 = post_par[one_params_i]['postprocess2']
            b1 = post_par[one_params_i]['postprocess1_bias']
            b2 = post_par[one_params_i]['postprocess2_bias']

            raw_output = tf.nn.relu(raw_output)
            raw_output = tf.nn.conv1d(raw_output, w1, stride=1, padding="SAME")
            raw_output = tf.add(raw_output, b1)

            raw_output = tf.nn.relu(raw_output)
            raw_output = tf.nn.conv1d(raw_output, w2, stride=1, padding="SAME")
            raw_output = tf.add(raw_output, b2)

            raw_output = tf.squeeze(raw_output)

            raw_output_list.append(raw_output)
            network_label = tf.squeeze(network_label)

    # Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()

    # Set up session
    #config = tf.ConfigProto(log_device_placement=False)
    #config.gpu_options.allow_growth = True
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=tf_config)
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

    from tqdm import tqdm
    try:
        pass_loss = 0.0
        for step in tqdm(range(90400)):
            '''
            if step < args.num_steps*0.0:
                muti_step_id = 0
            elif step < args.num_steps*0.99:
                muti_step_id = 1
            else:
                muti_step_id = 2
            '''
            muti_step_id = 1
            start_time = time.time()

            audio_batch_list_ori_v, audio_batch_list_now_v,\
            raw_output_list_ori_v, raw_output_list_now_v,\
                file_name= \
                sess.run([audio_batch_list[muti_step_id-1],audio_batch_list[muti_step_id],  #label input
                          raw_output_list[muti_step_id-1], raw_output_list[muti_step_id],
                          audio_batch_file])

            audio_batch_list_ori_v = audio_batch_list_ori_v[0, one_receptive_field:, :]
            audio_batch_list_now_v = audio_batch_list_now_v[0, one_receptive_field:, :]
            #writer.add_summary(summary, step)


            if step%1==0:
                wave_result0 = raw_output_list_ori_v[(one_receptive_field+1):, :]   #t'
                wave_result1 = raw_output_list_now_v[(one_receptive_field+1):, :] + wave_result0 #t'+(t-t')'
                wave_target = audio_batch_list_ori_v[:,9:12][one_receptive_field:, :]   #t
                wave_input = audio_batch_list_ori_v[:, 0:3][one_receptive_field:, :]    #t+v

                num_sampe = np.size(wave_target,axis=0)
                total_tremor = np.zeros(3)
                remove_tremor0 = np.zeros(3)
                remove_tremor1 = np.zeros(3)
                #统计补偿的震颤的比例：
                for tmp_over1_i in range(num_sampe):
                    for xyz_i in range(3):
                        # 累加振幅
                        total_tremor[xyz_i] += abs(wave_target[tmp_over1_i, xyz_i])
                        # 计算有效的消除震颤的部分——附加后震颤幅度变小：方向一致且幅度小于2×target
                        if wave_target[tmp_over1_i, xyz_i] * wave_result0[tmp_over1_i, xyz_i] > 0:
                            if abs(wave_result0[tmp_over1_i, xyz_i]) < abs(wave_target[tmp_over1_i, xyz_i]):
                                remove_tremor0[xyz_i] += abs(wave_result0[tmp_over1_i, xyz_i])

                            elif abs(wave_result0[tmp_over1_i, xyz_i]) / 2 < abs(wave_target[tmp_over1_i, xyz_i]):
                                remove_tremor0[xyz_i] += 2 * abs(wave_target[tmp_over1_i, xyz_i]) - abs(
                                    wave_result0[tmp_over1_i, xyz_i])

                        if wave_target[tmp_over1_i, xyz_i] * wave_result1[tmp_over1_i, xyz_i] > 0:
                            if abs(wave_result1[tmp_over1_i, xyz_i]) < abs(wave_target[tmp_over1_i, xyz_i]):
                                remove_tremor1[xyz_i] += abs(wave_result1[tmp_over1_i, xyz_i])

                            elif abs(wave_result1[tmp_over1_i, xyz_i]) / 2 < abs(wave_target[tmp_over1_i, xyz_i]):
                                remove_tremor1[xyz_i] += 2 * abs(wave_target[tmp_over1_i, xyz_i]) - abs(
                                    wave_result1[tmp_over1_i, xyz_i])
                file_name = file_name[0].split('/')[-2] #'044'
                # print(total_tremor)
                # print(remove_tremor0)
                # print(remove_tremor1)
                # print(np.sum(remove_tremor0) / np.sum(total_tremor))
                # print(np.sum(remove_tremor1) / np.sum(total_tremor))

                wave_result0 = wave_input - wave_result0
                wave_result1 = wave_input - wave_result1
                wave_target = wave_input - wave_target

                wave_analysis = np.array([np.sum(remove_tremor0) / np.sum(total_tremor), np.sum(remove_tremor1) / np.sum(total_tremor)])
                # /home/chen/Documents/tensorflow-wavenet-master/analysis/test
                # 清道夫的工作
                test_path = "/home/chen/Documents/tensorflow-wavenet-master/analysis/test/3/" + file_name + ".txt"
                #if not os.path.exists(test_path):
                #    os.makedirs(test_path)
                #    print(test_path)

                if not os.path.exists(test_path):
                    np.savetxt(test_path, wave_analysis)
                else:
                    wave_analysis_his = np.loadtxt(test_path)
                    wave_analysis = np.vstack((wave_analysis_his, wave_analysis))
                    np.savetxt(test_path, wave_analysis)

                # 绘制3D图
                if step % 100 == 0:
                    fig = plt.figure(1)
                    fig.clear()
                    ax1 = plt.subplot(231, projection='3d')
                    ax2 = plt.subplot(232, projection='3d')
                    ax3 = plt.subplot(233, projection='3d')
                    ax4 = plt.subplot(234, projection='3d')
                    ax5 = plt.subplot(235, projection='3d')
                    ax6 = plt.subplot(236, projection='3d')
                    lim_show = np.size(wave_input,axis = 0)
                    lim_show = int(math.floor(lim_show/5))
                    ax1.plot(wave_input[:, 0], wave_input[:, 1], wave_input[:, 2], linewidth=0.3)
                    ax4.plot(wave_target[:, 0], wave_target[:, 1], wave_target[:, 2], linewidth=0.3)
                    ax2.plot(wave_result0[:, 0], wave_result0[:, 1], wave_result0[:, 2], linewidth=0.3)
                    ax2.set_title(str(np.sum(remove_tremor0) / np.sum(total_tremor)))
                    ax5.plot(wave_result1[:, 0], wave_result1[:, 1], wave_result1[:, 2], linewidth=0.3)
                    ax5.set_title(str(np.sum(remove_tremor1) / np.sum(total_tremor)))

                    ax3.plot(wave_result0[-lim_show:, 0], wave_result0[-lim_show:, 1], wave_result0[-lim_show:, 2], linewidth=0.3)
                    ax3.plot(wave_target[-lim_show:, 0], wave_target[-lim_show:, 1], wave_target[-lim_show:, 2], linewidth=0.3)
                    ax6.plot(wave_result1[-lim_show:, 0], wave_result1[-lim_show:, 1], wave_result1[-lim_show:, 2], linewidth=0.3)
                    ax6.plot(wave_target[-lim_show:, 0], wave_target[-lim_show:, 1], wave_target[-lim_show:, 2], linewidth=0.3)

                    plt.savefig("/home/chen/Documents/tensorflow-wavenet-master/images/test_result/" + str(step).zfill(7) + ".png")

            #if step % args.checkpoint_every == 0:
            #    save(saver, sess, logdir, step)
            #    last_saved_step = step

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


