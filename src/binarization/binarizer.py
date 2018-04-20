import tensorflow as tf
import numpy as np

from . import helper

class Binarizer(object):

    def __init__(self, data, target, n_hidden, input_factor,
                    positive_weight=None, weights=None):
        if (n_hidden % int(target.get_shape()[-1])) > 0:
            msg = "n_hidden ({0}) must be multiple of input size ({1})"
            msg = msg.format(n_hidden, int(target.get_shape()[-1]))
            raise ValueError(msg)  
        
        target_out = int(target.get_shape()[2])
	
        self.out_size = target_out
	self.out_h = int(np.sqrt(target_out))
        self.n_hidden = n_hidden
        self.input_factor = input_factor
        self.positive_weight = positive_weight
        self.weights = weights
        self.data = data
        self.target = target
        self.seq_length = [int(target.get_shape()[1]) for i in
                                        range(int(target.get_shape()[0]))]

        # init functions to build computational graph
        self.prediction
        self.optimize
        self.error
        self.loss
        self.binarize


    def _lstm_input_layer(self):
        # create Grid LSTM layer for processing image top left to bottom right
        with tf.variable_scope("top-left"):
            lstm_fw_cell_tl = tf.contrib.grid_rnn.GridRNNCell(
                                        num_units=self.n_hidden,
                                        num_dims=4, input_dims=[0,1,2],
                                        output_dims=0,
                                        priority_dims=0)
            lstm_bw_cell_tl = tf.contrib.grid_rnn.GridRNNCell(
                                        num_units=self.n_hidden,
                                        num_dims=4, input_dims=[0,1,2],
                                        output_dims=0,
                                        priority_dims=0)

            outputs_tl, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_tl,
                                                lstm_bw_cell_tl,
                                                self.data[0], dtype=tf.float32,
                                                sequence_length=self.seq_length,
                                                swap_memory=True)
            # extract separate outputs
            output_fw_tl, output_bw_tl = outputs_tl

        # create Grid LSTM layer for processing image bottom left to top right
        with tf.variable_scope("bottom-left"):
            lstm_fw_cell_bl = tf.contrib.grid_rnn.GridRNNCell(
                                        num_units=self.n_hidden,
                                        num_dims=4, input_dims=[0,1,2],
                                        output_dims=0,
                                        priority_dims=0)
            lstm_bw_cell_bl = tf.contrib.grid_rnn.GridRNNCell(
                                        num_units=self.n_hidden,
                                        num_dims=4, input_dims=[0,1,2],
                                        output_dims=0,
                                        priority_dims=0)

            outputs_bl, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_bl,
                                                lstm_bw_cell_bl,
                                                self.data[1], dtype=tf.float32,
                                                sequence_length=self.seq_length,
                                                swap_memory=True)
            # extract separate outputs
            output_fw_bl, output_bw_bl = outputs_bl

        output_fw_bl = self._flip_patches(output_fw_bl, 'fw-bl')
        output_bw_bl = self._flip_patches(output_bw_bl, 'bw-bl')
        # concatenate outputs of forward and backward pass
        return tf.concat([output_fw_tl, output_bw_tl, output_fw_bl,
                                output_bw_bl], 2)

    def _output_shape(self, out_size, factor):
        out_height = int(np.sqrt(out_size/factor))
        out_width = out_height * factor
        return out_height, out_width

    def _output_number(self, seq_len):
        return int(np.sqrt(seq_len))

    def _flip_patches(self, tensor, name):
        with tf.variable_scope('flip-' + name):
            batch_size, seq_len, out_size = tensor.get_shape().as_list()
            out_h, out_w = self._output_shape(out_size, self.input_factor)
            output_number = self._output_number(seq_len)
            # prepare for flipping
            tmp = tf.reshape(tensor, [batch_size, output_number, -1, out_h, out_w])
            tmp = tf.transpose(tmp, [0,1,3,2,4])
            tmp = tf.reshape(tmp, [batch_size, out_h * output_number,
                                    out_w * output_number])
            # flip
            tmp = tf.reverse_v2(tmp, [1])
            # transform back
            tmp = tf.reshape(tmp, [batch_size, output_number, out_h, -1, out_w])
            tmp = tf.transpose(tmp, [0,1,3,2,4])
            return tf.reshape(tmp, [batch_size, seq_len, out_size])

    def _lstm_hidden_layer(self, data):
        out_size = int(self.target.get_shape()[2])

        lstm_fw_cell = tf.contrib.grid_rnn.GridRNNCell(
                                    num_units=out_size,
                                    num_dims=2, input_dims=[0],
                                    output_dims=0,
                                    priority_dims=0)
        lstm_bw_cell = tf.contrib.grid_rnn.GridRNNCell(
                                    num_units=out_size,
                                    num_dims=2, input_dims=[0],
                                    output_dims=0,
                                    priority_dims=0)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                            lstm_bw_cell,
                                            data, dtype=tf.float32,
                                            sequence_length=self.seq_length,
                                            swap_memory=True)

        output_fw, output_bw = outputs
        return tf.concat([output_fw, output_bw], 2)

    def _output_layer(self, data):
        max_length = int(self.target.get_shape()[1])
        out_size = int(self.target.get_shape()[2])
        in_size = int(data.get_shape()[2])

        # create softmax layer on top of LSTM layer
        weight = tf.Variable(tf.truncated_normal([in_size, out_size]))
        bias = tf.Variable(tf.constant(0.1, shape=[out_size]))

        # reshape to perform matrix multiplication between the output at each time step
        # and the weight matrix
        out = tf.reshape(data, [-1, in_size])
        #prediction = tf.nn.softmax(tf.matmul(out, weight) + bias)
        # map all output values between 0 and 1
        #prediction = tf.sigmoid(tf.matmul(out, weight) + bias)
        prediction = tf.matmul(out, weight) + bias

        # restore shape to separate time steps of different batch items
        prediction = tf.reshape(prediction, [-1, max_length, out_size])
        tf.summary.histogram('activations', prediction)
        return prediction

    @helper.define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def prediction(self):
        out = self._lstm_input_layer()
        out2 = self._lstm_hidden_layer(out)
        return self._output_layer(out2)

    def _loss(self, feedback=False):
        batch_size, seq_len, out_size = self.target.get_shape().as_list()
        target = tf.reshape(self.target, [-1, out_size])
        prediction = tf.reshape(self.prediction, [-1, out_size])
        tf.summary.histogram('predicted-class', prediction)

        if self.positive_weight is None:
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                                            labels = target,
                                            logits = prediction)
            if self.weights is not None:
                weights = tf.reshape(self.weights, [-1, out_size])
                loss = tf.multiply(loss, weights)
        else:
            loss = tf.nn.weighted_cross_entropy_with_logits(
                                            targets = target,
                                            logits = prediction,
                                            pos_weight = self.positive_weight)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('loss', loss)
        return loss

    @helper.define_scope
    def optimize(self):
        target = self._loss()
        # setup optimizer
        optimizer = tf.train.AdamOptimizer()
        return optimizer.minimize(target)

    def _store_image(self, tensor, name):
        with tf.variable_scope('store-'+name):
            batch_size, seq_len, out_size = tensor.get_shape().as_list()
            img =  tf.reshape(tensor, [-1, self.out_size])
            output_number = self._output_number(seq_len)
            img = tf.cast(img, tf.uint8)
            img = tf.reshape(img, [batch_size, output_number, -1, self.out_h,
                                    self.out_h])
            img = tf.transpose(img, [0,1,3,2,4])
            dim = output_number * self.out_h
            img = tf.reshape(img, [batch_size, dim, dim, 1])
            img = img * 255
            tf.summary.image(name, img)

    @helper.define_scope
    def error(self):
        self._store_image(self.prediction, 'predicted-patch')
        self._store_image(self.target, 'target-patch')

        # find which class each pixel has in target and predicted vector
        target =  tf.reshape(self.target, [-1, self.out_size])
        prediction =  tf.reshape(self.prediction, [-1, self.out_size])
        # compare if the class indices match
        prediction = tf.round(prediction)
        prediction = tf.clip_by_value(prediction, 0, 1)
        mistakes = tf.not_equal(target, prediction)
        error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

        tf.summary.scalar('error', error)
        return error

    @helper.define_scope
    def loss(self):
        loss = self._loss()
        return loss

    @helper.define_scope
    def binarize(self):
        batch_size, seq_len, out_size = self.prediction.get_shape().as_list()
        prediction = tf.reshape(tf.sigmoid(self.prediction), [-1, self.out_size])
        result = tf.round(prediction)
        return tf.reshape(result, [batch_size, seq_len, -1])






