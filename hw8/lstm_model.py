import tensorflow as tf
import keras.layers as L


class LstmBidirectionalBilinearAttentionTranslationModel:
    def __init__(self, name, inp_voc, out_voc,
                 emb_size, hid_size,):

        self.name = name
        self.inp_voc = inp_voc
        self.out_voc = out_voc

        with tf.variable_scope(name):
            self.emb_inp = L.Embedding(len(inp_voc), emb_size)
            self.emb_out = L.Embedding(len(out_voc), emb_size)
            self.enc0 = tf.contrib.rnn.LSTMCell(int(hid_size / 2), state_is_tuple=False)
            self.dec_start = L.Dense(hid_size * 2)
            self.dec0 = tf.contrib.rnn.LSTMCell(hid_size, state_is_tuple=False)
            self.logits = L.Dense(len(out_voc))
            self.W_omega = tf.Variable(tf.random_normal([hid_size, hid_size], stddev=0.1))
            self.b_omega = tf.Variable(tf.random_normal([hid_size], stddev=0.1))


            # run on dummy output to .build all layers (and therefore create weights)
            inp = tf.placeholder('int32', [None, None])
            out = tf.placeholder('int32', [None, None])
            all_h, h0 = self.encode(inp)
            h1 = self.decode(h0, out[:,0], all_h)
            # h2 = self.decode(h1,out[:,1]) etc.

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)


    def encode(self, inp, **flags):
        """
        Takes symbolic input sequence, computes initial state
        :param inp: matrix of input tokens [batch, time]
        :return: a list of initial decoder state tensors
        """
        inp_lengths = infer_length(inp, self.inp_voc.eos_ix)
        inp_emb = self.emb_inp(inp)

        (output_fw, output_bw), (enc_last_fw, enc_last_bw) = tf.nn.bidirectional_dynamic_rnn(
                          self.enc0, self.enc0, inp_emb,
                          sequence_length=inp_lengths,
                          dtype = inp_emb.dtype)
        
        outputs = tf.concat([output_fw, output_bw], -1)
        enc_last = tf.concat([enc_last_fw, enc_last_bw], -1)
        print(outputs, enc_last)
        # concatenated = tf.concat([outputs, reversed_outputs], 2)
        # outputs - [B, T, D]
        dec_start = self.dec_start(enc_last)
        # dec_start = enc_last
        return outputs, [dec_start]

    def decode(self, prev_state, prev_tokens, all_h, **flags):
        """
        Takes previous decoder state and tokens, returns new state and logits
        :param prev_state: a list of previous decoder state tensors
        :param prev_tokens: previous output tokens, an int vector of [batch_size]
        :return: a list of next decoder state tensors, a tensor of logits [batch,n_tokens]
        """

        [prev_dec] = prev_state
        s = prev_dec[:, :64]
        s_c = prev_dec[:, 64:]

        prev_emb = self.emb_out(prev_tokens[:,None])[:,0]
        
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        b_t_d = tf.shape(all_h)
        d_a = tf.shape(self.W_omega)
        all_h_w_dot = tf.reshape(tf.matmul(tf.reshape(all_h, [-1, b_t_d[2]]), self.W_omega),
                       shape=[b_t_d[0], b_t_d[1], d_a[1]])
        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        all_h_w_s_dot = tf.reshape(tf.matmul(all_h_w_dot, tf.expand_dims(s, -1)), shape=[b_t_d[0], b_t_d[1]]) # (B,T) shape
        alphas = tf.nn.softmax(all_h_w_s_dot)    # (B,T) shape also
        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(all_h * tf.expand_dims(alphas, -1), 1)
        
        new_prev_emb = tf.concat([prev_emb, output], 1)
        # new_dec_out - [B, D]
        new_dec_out, new_dec_state = self.dec0(new_prev_emb, prev_dec)
        
        # ahh_h - [B, T, D]
        output_logits = self.logits(new_dec_out)
        # output_logits = self.logits(new_dec_out)
        return [new_dec_state], output_logits

    def symbolic_score(self, inp, out, eps=1e-30, **flags):
        """
        Takes symbolic int32 matrices of hebrew words and their english translations.
        Computes the log-probabilities of all possible english characters given english prefices and hebrew word.
        :param inp: input sequence, int32 matrix of shape [batch,time]
        :param out: output sequence, int32 matrix of shape [batch,time]
        :return: log-probabilities of all possible english characters of shape [bath,time,n_tokens]

        NOTE: log-probabilities time axis  is synchronized with out
        In other words, logp are probabilities of __current__ output at each tick, not the next one
        therefore you can get likelihood as logprobas * tf.one_hot(out,n_tokens)
        """
        all_h, first_state = self.encode(inp,**flags)

        batch_size = tf.shape(inp)[0]
        bos = tf.fill([batch_size],self.out_voc.bos_ix)
        first_logits = tf.log(tf.one_hot(bos, len(self.out_voc)) + eps)

        def step(blob, y_prev):
            h_prev = blob[:-1]
            h_new, logits = self.decode(h_prev, y_prev, all_h, **flags)
            return list(h_new) + [logits]

        results = tf.scan(step,initializer=list(first_state)+[first_logits],
                          elems=tf.transpose(out))

        # gather state and logits, each of shape [time,batch,...]
        states_seq, logits_seq = results[:-1], results[-1]

        # add initial state and logits
        logits_seq = tf.concat((first_logits[None], logits_seq),axis=0)
        states_seq = [tf.concat((init[None], states), axis=0)
                      for init, states in zip(first_state, states_seq)]

        #convert from [time,batch,...] to [batch,time,...]
        logits_seq = tf.transpose(logits_seq, [1, 0, 2])
        states_seq = [tf.transpose(states, [1, 0] + list(range(2, states.shape.ndims)))
                      for states in states_seq]

        return tf.nn.log_softmax(logits_seq)

    def symbolic_translate(self, inp, greedy=False, max_len = None, eps = 1e-30, **flags):
        """
        takes symbolic int32 matrix of hebrew words, produces output tokens sampled
        from the model and output log-probabilities for all possible tokens at each tick.
        :param inp: input sequence, int32 matrix of shape [batch,time]
        :param greedy: if greedy, takes token with highest probablity at each tick.
            Otherwise samples proportionally to probability.
        :param max_len: max length of output, defaults to 2 * input length
        :return: output tokens int32[batch,time] and
                 log-probabilities of all tokens at each tick, [batch,time,n_tokens]
        """
        all_h, first_state = self.encode(inp, **flags)

        batch_size = tf.shape(inp)[0]
        bos = tf.fill([batch_size],self.out_voc.bos_ix)
        first_logits = tf.log(tf.one_hot(bos, len(self.out_voc)) + eps)
        max_len = tf.reduce_max(tf.shape(inp)[1])*2

        def step(blob,t):
            h_prev, y_prev = blob[:-2], blob[-1]
            h_new, logits = self.decode(h_prev, y_prev, all_h, **flags)
            y_new = tf.argmax(logits,axis=-1) if greedy else tf.multinomial(logits,1)[:,0]
            return list(h_new) + [logits, tf.cast(y_new,y_prev.dtype)]

        results = tf.scan(step, initializer=list(first_state) + [first_logits, bos],
                          elems=[tf.range(max_len)])

        # gather state, logits and outs, each of shape [time,batch,...]
        states_seq, logits_seq, out_seq = results[:-2], results[-2], results[-1]

        # add initial state, logits and out
        logits_seq = tf.concat((first_logits[None],logits_seq),axis=0)
        out_seq = tf.concat((bos[None], out_seq), axis=0)
        states_seq = [tf.concat((init[None], states), axis=0)
                      for init, states in zip(first_state, states_seq)]

        #convert from [time,batch,...] to [batch,time,...]
        logits_seq = tf.transpose(logits_seq, [1, 0, 2])
        out_seq = tf.transpose(out_seq)
        states_seq = [tf.transpose(states, [1, 0] + list(range(2, states.shape.ndims)))
                      for states in states_seq]

        return out_seq, tf.nn.log_softmax(logits_seq)



### Utility functions ###

def initialize_uninitialized(sess = None):
    """
    Initialize unitialized variables, doesn't affect those already initialized
    :param sess: in which session to initialize stuff. Defaults to tf.get_default_session()
    """
    sess = sess or tf.get_default_session()
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

def infer_length(seq, eos_ix, time_major=False, dtype=tf.int32):
    """
    compute length given output indices and eos code
    :param seq: tf matrix [time,batch] if time_major else [batch,time]
    :param eos_ix: integer index of end-of-sentence token
    :returns: lengths, int32 vector of shape [batch]
    """
    axis = 0 if time_major else 1
    is_eos = tf.cast(tf.equal(seq, eos_ix), dtype)
    count_eos = tf.cumsum(is_eos,axis=axis,exclusive=True)
    lengths = tf.reduce_sum(tf.cast(tf.equal(count_eos,0),dtype),axis=axis)
    return lengths

def infer_mask(seq, eos_ix, time_major=False, dtype=tf.float32):
    """
    compute mask given output indices and eos code
    :param seq: tf matrix [time,batch] if time_major else [batch,time]
    :param eos_ix: integer index of end-of-sentence token
    :returns: mask, float32 matrix with '0's and '1's of same shape as seq
    """
    axis = 0 if time_major else 1
    lengths = infer_length(seq, eos_ix, time_major=time_major)
    mask = tf.sequence_mask(lengths, maxlen=tf.shape(seq)[axis], dtype=dtype)
    if time_major: mask = tf.transpose(mask)
    return mask


def select_values_over_last_axis(values, indices):
    """
    Auxiliary function to select logits corresponding to chosen tokens.
    :param values: logits for all actions: float32[batch,tick,action]
    :param indices: action ids int32[batch,tick]
    :returns: values selected for the given actions: float[batch,tick]
    """
    assert values.shape.ndims == 3 and indices.shape.ndims == 2
    batch_size, seq_len = tf.shape(indices)[0], tf.shape(indices)[1]
    batch_i = tf.tile(tf.range(0,batch_size)[:, None],[1,seq_len])
    time_i = tf.tile(tf.range(0,seq_len)[None, :],[batch_size,1])
    indices_nd = tf.stack([batch_i, time_i, indices], axis=-1)

    return tf.gather_nd(values,indices_nd)



