import tensorflow as tf

class AFD_RNN(object):

    time_step = 150
    class_num = 8
    num_units = 64
    senor_data_num = 6
    batch_size = 64

    def __init__(self):
        self.build_net_graph()

    def build_net_graph(self):
        self.x = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.senor_data_num])
        self.label = tf.placeholder(tf.float32, [self.class_num])

        self.add_input_layer()
        self.add_rnn_layer()
        self.add_output_layer()

    def add_input_layer(self):
        input_x = tf.reshape(self.x, [-1, self.senor_data_num])
        weights_x = self.get_variable_weights([self.senor_data_num, self.num_units], 'input_weights')
        biases_x = self.get_variable_biases([self.num_units], 'input_biases')
        self.x_output = tf.reshape(tf.matmul(input_x, weights_x) + biases_x, [-1, self.time_step, self.num_units])

    def add_rnn_layer(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units)
        self.cell_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)

        # outputs shape =[batch_size, max_time, cell_state_size]
        # LSTM final_state shape = [2, batch_size, cell_state_size]
        self.cell_outputs, self.final_state = tf.nn.dynamic_rnn(lstm_cell,
                                                           self.x_output,
                                                           initial_state=self.cell_state,
                                                           time_major=False)
    def add_output_layer(self):
        outputs = tf.reshape(self.cell_outputs, [-1, self.time_step, self.num_units])
        weights_outputs = self.get_variable_weights([self.num_units, self.class_num], 'outputs_weights')
        biases_outputs = self.get_variable_biases([self.class_num], 'outputs_biases')
        self.predict = tf.reshape(tf.add(tf.matmul(outputs, weights_outputs), biases_outputs),
                                  [self.batch_size, self.time_step, self.class_num])

    def get_variable_weights(self, shape, name):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1), dtype=tf.float32, name=name)

    def get_variable_biases(self, shape, name):
        return tf.Variable(tf.constant(0.1, shape), dtype=tf.float32, name=name)