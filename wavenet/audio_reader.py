import fnmatch
import os
import random
import re
import threading
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import librosa
import numpy as np
import tensorflow as tf

FILE_PATTERN = r'p([0-9]+)_([0-9]+)\.wav'

def figure_hand_back(uvd_pt,uvd_pt2,path,test_num):
    #uvd_pt = np.reshape(uvd_pt, (20, 3))
    uvd_pt = uvd_pt.reshape(-1, 3)
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

    plt.ylim(-300, 100)
    plt.xlim(-300, 100)
    ax.set_zlim(-300, 100)

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
def get_category_cardinality(files):
    id_reg_expression = re.compile(FILE_PATTERN)
    min_id = None
    max_id = None
    for filename in files:
        matches = id_reg_expression.findall(filename)[0]
        id, recording_id = [int(id_) for id_ in matches]
        if min_id is None or id < min_id:
            min_id = id
        if max_id is None or id > max_id:
            max_id = id

    return min_id, max_id


def randomize_files(files):
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]
def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_generic_audio(directory, sample_rate):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    id_reg_exp = re.compile(FILE_PATTERN)
    print("files length: {}".format(len(files)))
    randomized_files = randomize_files(files)
    for filename in randomized_files:
        ids = id_reg_exp.findall(filename)
        if not ids:
            # The file name does not match the pattern containing ids, so
            # there is no id.
            category_id = None
        else:
            # The file name matches the pattern for containing ids.
            category_id = int(ids[0][0])
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        yield audio, filename, category_id
def load_generated_pose(directory, sample_rate):
    '''Generator that yields audio waveforms from the directory.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*.txt'):
            files.append(os.path.join(root, filename))
    FILE_PATTERN = r'^[a-zA-Z0-9_-]+\.txt$'
    id_reg_exp = re.compile(FILE_PATTERN)
    print("files length: {}".format(len(files)))
    randomized_files = randomize_files(files)
    for filename in randomized_files:
        ids = id_reg_exp.findall(filename)
        if not ids:
            # The file name does not match the pattern containing ids, so
            # there is no id.
            category_id = None
        else:
            # The file name matches the pattern for containing ids.
            category_id = int(ids[0][0])
        #audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        #audio = audio.reshape(-1, 1)
        onefile_data_pose_r = np.loadtxt(filename)
        shape = onefile_data_pose_r.shape
        onefile_data_pose_r = onefile_data_pose_r.reshape(shape[0], 22, 3)
        for i in range(20):
            onefile_data_pose_r[:,i,:] = onefile_data_pose_r[:,i,:]  - onefile_data_pose_r[:,19,:]
        onefile_data_pose_r = onefile_data_pose_r.reshape(shape[0],shape[1])
        #onefile_data_pose_r = onefile_data_pose_r[:, 0:shape[1] - 2, :]
        #for test_read_i in range(shape[0]):
        #    figure_joint_skeleton(onefile_data_pose_r[test_read_i, :, :], path + "/" + seqid + "/", test_read_i)
        yield onefile_data_pose_r, filename, category_id
def load_generated_tra(directory, sample_rate):
    '''Generator that yields audio waveforms from the directory.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*.txt'):
            files.append(os.path.join(root, filename))
    print("files_gt length: {}".format(len(files)))
    randomized_files = randomize_files(files)
    for filename in randomized_files:
        onefile_randomized_files = np.loadtxt(filename)
        #shape = onefile_randomized_files.shape
        #path_view_txt = "/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/pakinson/shake_tra/view/"
        #for test_read_i in range(shape[0]):
        #    figure_hand_back(onefile_randomized_files[test_read_i, 0:9], onefile_randomized_files[test_read_i, 9:],
        #                     path_view_txt, test_read_i)
        yield onefile_randomized_files, filename, None
def trim_silence(audio, threshold, frame_length=2048):
    '''Removes silence at the beginning and end of a sample.'''
    if audio.size < frame_length:
        frame_length = audio.size
    energy = librosa.feature.rmse(audio, frame_length=frame_length)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


def not_all_have_id(files):
    ''' Return true iff any of the filenames does not conform to the pattern
        we require for determining the category id.'''
    id_reg_exp = re.compile(FILE_PATTERN)
    for file in files:
        ids = id_reg_exp.findall(file)
        if not ids:
            return True
    return False


class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 audio_dir,
                 coord,
                 sample_rate,
                 gc_enabled,
                 receptive_field,
                 sample_size=None,
                 silence_threshold=None,
                 queue_size=32):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.coord = coord
        self.sample_size = sample_size
        self.receptive_field = receptive_field
        self.silence_threshold = silence_threshold
        self.gc_enabled = gc_enabled
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32'],
                                         shapes=[(None, 9*2)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder])

        if self.gc_enabled:
            self.id_placeholder = tf.placeholder(dtype=tf.int32, shape=())
            self.gc_queue = tf.PaddingFIFOQueue(queue_size, ['int32'],
                                                shapes=[()])
            self.gc_enqueue = self.gc_queue.enqueue([self.id_placeholder])

        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        files = find_files(audio_dir)
        if not files:
            raise ValueError("No audio files found in '{}'.".format(audio_dir))
        if self.gc_enabled and not_all_have_id(files):
            raise ValueError("Global conditioning is enabled, but file names "
                             "do not conform to pattern having id.")
        # Determine the number of mutually-exclusive categories we will
        # accomodate in our embedding table.
        if self.gc_enabled:
            _, self.gc_category_cardinality = get_category_cardinality(files)
            # Add one to the largest index to get the number of categories,
            # since tf.nn.embedding_lookup expects zero-indexing. This
            # means one or more at the bottom correspond to unused entries
            # in the embedding lookup table. But that's a small waste of memory
            # to keep the code simpler, and preserves correspondance between
            # the id one specifies when generating, and the ids in the
            # file names.
            self.gc_category_cardinality += 1
            print("Detected --gc_cardinality={}".format(
                  self.gc_category_cardinality))
        else:
            self.gc_category_cardinality = None

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output

    def dequeue_gc(self, num_elements):
        return self.gc_queue.dequeue_many(num_elements)

    def thread_main(self, sess):
        stop = False
        # Go through the dataset multiple times
        while not stop:
            # chen_test data
            #iterator = load_generated_pose("/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/shake", self.sample_rate)
            #iterator = load_generic_audio(self.audio_dir, self.sample_rate)
            iterator = load_generated_tra("/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/pakinson/shake_tra", self.sample_rate)
            for audio, filename, category_id in iterator:
                if self.coord.should_stop():
                    stop = True
                    break
                if self.silence_threshold is not None:
                    # Remove silence
                    audio = trim_silence(audio[:, 0], self.silence_threshold)
                    audio = audio.reshape(-1, 1)
                    if audio.size == 0:
                        print("Warning: {} was ignored as it contains only "
                              "silence. Consider decreasing trim_silence "
                              "threshold, or adjust volume of the audio."
                              .format(filename))

                audio = np.pad(audio, [[self.receptive_field, 0], [0, 0]],
                               'constant')

                if self.sample_size:
                    # Cut samples into pieces of size receptive_field +
                    # sample_size with receptive_field overlap
                    while len(audio) > self.receptive_field:
                        piece = audio[:(self.receptive_field +
                                        self.sample_size), :]
                        sess.run(self.enqueue,
                                 feed_dict={self.sample_placeholder: piece})
                        audio = audio[self.sample_size:, :]
                        if self.gc_enabled:
                            sess.run(self.gc_enqueue, feed_dict={
                                self.id_placeholder: category_id})
                else:
                    sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: audio})
                    if self.gc_enabled:
                        sess.run(self.gc_enqueue,
                                 feed_dict={self.id_placeholder: category_id})

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            # chen_test
            # target = self.thread_main(sess)
            # chen_test_end
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
