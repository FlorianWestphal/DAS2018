#!/usr/bin/python -u

import tensorflow as tf
import numpy as np

import random
import time
import sys

import binarization as bn

if len(sys.argv) != 5:
    print 'usage:', sys.argv[0], '<dataset> <model> <target> <mode string>'
    exit(1)

def suitable_batch_size(batch_size):
    if batch_size < 600:
        return batch_size, 1
    for i in reversed(range(1, 33)):
        if batch_size % i == 0:
            batch_size = batch_size / i
            num = i
            break
    return batch_size, num

dataset = sys.argv[1]
model = sys.argv[2]
target_folder = sys.argv[3]
mode_string = sys.argv[4]
config = bn.config.Configuration()

# adjust configuration based on mode
if mode_string == 'dyn_wgt_sc1_16_201e':
    config.hdfs_path = "/home2/flw/binarization/recurrent_binarization/dibco_sc1_fp16.hdfs5"
elif mode_string == 'dyn_wgt_sc2_16_201e':
    pass
elif mode_string == 'dyn_wgt_sc2_4equal_201e':
    config.hdfs_path = "/home2/flw/binarization/recurrent_binarization/dibco_sc2_fp4.hdfs5"
    config.input_factor = 20
elif mode_string == 'dyn_wgt_sc2_64_201e':
    config.hdfs_path = "/home2/flw/binarization/recurrent_binarization/dibco_sc2_fp64.hdfs5"
elif mode_string == 'dyn_wgt_sc4_16_201e':
    config.hdfs_path = "/home2/flw/binarization/recurrent_binarization/dibco_sc4_fp16.hdfs5"
elif mode_string == 'no_wgt_sc2_16_201e':
    config.dynamic_weights = False
    config.positive_weight = 1.0
elif mode_string == 'stat_wgt05_sc2_16_201e':
    config.dynamic_weights = False
    config.positive_weight = 0.5
elif mode_string == 'stat_wgt20_sc2_16_201e':
    config.dynamic_weights = False
    config.positive_weight = 2.0

with bn.storage.StorageReader(config.hdfs_path) as store:

    imgs = store.dataset_images(dataset)
    inputs = store.inputs()
    seq_length = store.patch_size()**2 / inputs
    conv = bn.converter.FromSeqConverter(store.patch_size())

    for img in imgs:
        org, x, y, batch_size = store.read_img(dataset, img)        
        batch_size, num = suitable_batch_size(batch_size)

        target = tf.placeholder(tf.float32, [batch_size, seq_length, inputs])
        tar = np.empty([batch_size, seq_length, store.inputs()])
        data = tf.placeholder(tf.float32, [2, batch_size, seq_length, 3 * inputs])
        org = org.reshape(2, num, -1, seq_length,
                            org[0].shape[-1]).swapaxes(0,1)

        binarizer = bn.binarizer.Binarizer(data, target, 
                                        n_hidden = inputs * config.input_factor,
                                        input_factor = config.input_factor,
                                        positive_weight = config.positive_weight)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, model)
            start = time.time()
            binarized = None
            for i in range(num):
                tmp = sess.run(binarizer.binarize, {data: org[i], target: tar})
                if binarized is None:
                    binarized = tmp
                else:
                    binarized = np.concatenate((binarized, tmp))
            duration = time.time() - start
            print '{},{}'.format(img, duration)
            conv.convert(binarized, x, y, '{0}/{1}.bmp'.format(target_folder,
                                                                img))
        tf.reset_default_graph()
