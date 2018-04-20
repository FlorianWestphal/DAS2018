#!/usr/bin/python -u

import tensorflow as tf
import numpy as np

import random
import time
import sys

import binarization as bn

if len(sys.argv) != 3:
    print 'usage:', sys.argv[0], '<config file> <seed>'
    exit(1)

def init_iterators(store, sample_size, seed):
    test_it = store.test_iterator()
    
    random.seed(seed)
    train_it = store.random_train_iterator(sample_size)
    valid_it = store.random_validation_iterator(sample_size)
    
    return train_it, valid_it, test_it

seed = int(sys.argv[2])
config_path = sys.argv[1]

config = bn.bin_config.BinConfig(config_path)
config.sample_seed = seed
print config

train, validation, test = config.configured_datasets()
print 'train set: {}'.format(train)
print 'validation set: {}'.format(validation)

batch_size = config.batches
with_weights = config.dynamic_weights

with bn.storage.StorageReader(config.hdfs_path, train=train, test=test,
        validation=validation, batch_size=batch_size, with_weights=with_weights) as store:

    inputs = store.inputs()
    seq_length = store.patch_size()**2 / inputs
    # create data placeholders

    target = tf.placeholder(tf.float32, [batch_size, seq_length, inputs])
    if config.dynamic_weights:
        weights = tf.placeholder(tf.float32, [batch_size, seq_length, inputs])
    else:
        weights = None
    data = tf.placeholder(tf.float32, [2, batch_size, seq_length, 3 * inputs])

    binarizer = bn.binarizer.Binarizer(data, target, 
                                    n_hidden = inputs * config.input_factor,
                                    input_factor = config.input_factor,
                                    positive_weight = config.positive_weight,
                                    weights = weights)

    errors = []
    losses = []
    epoch_times = []
    epoch = config.epochs#5000
    lowest_error = 100
    lowest_loss = float('inf')

    saver = tf.train.Saver()

    train_it, valid_it, test_it = init_iterators(store,
                                                    config.sample_size,
                                                    config.sample_seed) 

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(config.log_path + '/train', sess.graph)
        valid_writer = tf.summary.FileWriter(config.log_path + '/validation')
        
        if config.load_path is None:
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, config.load_path)
        
        for i in range(epoch):
            start_time = time.time()
            cnt = 0
            for (org, gt, wgts) in train_it:
                if config.dynamic_weights:
                    summary, _ = sess.run([merged, binarizer.optimize],
                                        {data: org, target: gt, weights: wgts})
                else:
                    summary, _ = sess.run([merged, binarizer.optimize],{data: org, target: gt})
                idx = i * 1000 + cnt
                train_writer.add_summary(summary, idx)
                cnt += 1
            train_it.reset()
            print "Epoch - ",str(i), "time:", str(time.time() - start_time)
            if i % 10 == 0:
                if config.store_path is not None:
		    saver.save(sess, config.store_path)
                cnt = 0
                for (org, gt, wgts) in valid_it:
                    if config.dynamic_weights:
                        summary, incorrect, loss = sess.run([merged,
                                        binarizer.error, binarizer.loss], 
                                        {data: org, target: gt, weights: gt})
                    else:
                        summary, incorrect, loss = sess.run([merged,
                                        binarizer.error, binarizer.loss], 
                                        {data: org, target: gt})
                    errors.append(incorrect)
                    losses.append(loss)
                    idx = i * 1000 + cnt
                    valid_writer.add_summary(summary, idx)
                    cnt += 1        

                incorrect = np.mean(errors)
                loss = np.mean(losses)
                print('Epoch {:2d} error {:3.5f}% loss: {:3.5f}'.format(i + 1,
                                                        100 * incorrect, loss))
                valid_it.reset()
                if incorrect < lowest_error and config.store_path is not None:
                    saver.save(sess, config.store_path + '_best_err')
                    lowest_error = incorrect
                if loss < lowest_loss and config.store_path is not None:
                    saver.save(sess, config.store_path + '_best_loss')
                    lowest_loss = loss
                del errors[:]
                del losses[:]

        if config.store_path is not None:
            saver.save(sess, config.store_path)

        for (org, gt, wgts) in test_it:
            if config.dynamic_weights:
                incorrect, loss = sess.run([binarizer.error, binarizer.loss], 
                                    {data: org, target: gt, weights: gt})
            else:
                incorrect, loss = sess.run([binarizer.error, binarizer.loss], 
                                    {data: org, target: gt})
            errors.append(incorrect)
            losses.append(loss)

        incorrect = np.mean(errors)
        loss = np.mean(losses)
        print('Epoch {:2d} error {:3.5f}% loss: {:3.5f}'.format(i + 1, 100 *
                                                            incorrect, loss))
