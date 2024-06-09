import os
import random
import numpy as np
try:
    import tensorflow as tf
except:
    pass
from utils import set_seed, log, set_tf_seed
from utils.imbFun import *
# tf.reset_default_graph()
def get_FLAGS():
    ''' Define parameter flags '''
    FLAGS = tf.app.flags.FLAGS

    tf.app.flags.DEFINE_integer('lrate_decay_num', 100, """NUM_ITERATIONS_PER_DECAY. """)
    tf.app.flags.DEFINE_integer('output_delay', 100, """Number of iterations between log/loss outputs. """)
    tf.app.flags.DEFINE_string('loss', 'log', """Which loss function to use (l1/l2/log)""")
    tf.app.flags.DEFINE_integer('n_in', 3, """Number of representation layers. """)
    tf.app.flags.DEFINE_integer('n_out', 3, """Number of regression layers. """)
    tf.app.flags.DEFINE_float('p_alpha', 1e-3, """The ratio of treament prediction loss. """)
    tf.app.flags.DEFINE_float('p_beta', 1e-1, """The ratio of disentangle loss. """)
    tf.app.flags.DEFINE_float('p_gamma', 1e-2, """The ratio of imb loss. """)
    tf.app.flags.DEFINE_float('p_lambda', 1e-4, """The ratio of regularization loss. """)
    tf.app.flags.DEFINE_integer('rep_weight_decay', 0, """Whether to penalize representation layers with weight decay""")
    tf.app.flags.DEFINE_float('dropout_in', 1.0, """Input layers dropout keep rate. """)
    tf.app.flags.DEFINE_float('dropout_out', 1.0, """Output layers dropout keep rate. """)
    tf.app.flags.DEFINE_string('nonlin', 'elu', """Kind of non-linearity. Default relu. """)
    tf.app.flags.DEFINE_float('lrate', 1e-3, """Learning rate. """)
    tf.app.flags.DEFINE_float('decay', 0.3, """RMSProp decay. """)
    tf.app.flags.DEFINE_integer('batch_size', 100, """Batch size. """)
    tf.app.flags.DEFINE_integer('dim_in', 256, """Pre-representation layer dimensions. """)
    tf.app.flags.DEFINE_integer('dim_out', 128, """Post-representation layer dimensions. """)
    tf.app.flags.DEFINE_integer('batch_norm', 0, """Whether to use batch normalization. """)
    tf.app.flags.DEFINE_string('normalization', 'divide', """How to normalize representation (after batch norm). none/bn_fixed/divide/project """)
    tf.app.flags.DEFINE_float('rbf_sigma', 0.1, """RBF MMD sigma """)
    tf.app.flags.DEFINE_integer('experiments', 2, """Number of experiments. """)
    tf.app.flags.DEFINE_integer('iterations', 3000, """Number of iterations. """)
    tf.app.flags.DEFINE_float('weight_init', 0.1, """Weight initialization scale. """)
    tf.app.flags.DEFINE_float('lrate_decay', 0.97, """Decay of learning rate every 100 iterations """)
    tf.app.flags.DEFINE_integer('wass_iterations', 10, """Number of iterations in Wasserstein computation. """)
    tf.app.flags.DEFINE_float('wass_lambda', 10.0, """Wasserstein lambda. """)
    tf.app.flags.DEFINE_integer('wass_bpt', 1, """Backprop through T matrix? """)
    tf.app.flags.DEFINE_integer('varsel', 0, """Whether the first layer performs variable selection. """)
    tf.app.flags.DEFINE_integer('sparse', 0, """Whether data is stored in sparse format (.x, .y). """)
    tf.app.flags.DEFINE_integer('use_p_correction', 0, """Whether to use population size p(t) in mmd/disc/wass.""")
    tf.app.flags.DEFINE_string('optimizer', 'Adam', """Which optimizer to use. (RMSProp/Adagrad/GradientDescent/Adam)""")
    tf.app.flags.DEFINE_string('imb_fun', 'wass', """Which imbalance penalty to use (mmd_lin/mmd_rbf/mmd2_lin/mmd2_rbf/lindisc/wass). """)
    tf.app.flags.DEFINE_integer('pred_output_delay', 200, """Number of iterations between prediction outputs. (-1 gives no intermediate output). """)
    tf.app.flags.DEFINE_float('val_part', 0.3, """Validation part. """)
    tf.app.flags.DEFINE_boolean('split_output', 1, """Whether to split output layers between treated and control. """)
    tf.app.flags.DEFINE_boolean('reweight_sample', 1, """Whether to reweight sample for prediction loss with average treatment probability. """)
    tf.app.flags.DEFINE_boolean('twoStage', 1, """twoStage. """)
    tf.app.flags.DEFINE_string('f', '', 'kernel')

    if FLAGS.sparse:
        import scipy.sparse as sparse

    return FLAGS

class VSDD(object):
 
    def __init__(self, x, t, y_ , p_t,s_mode1,s_mode2,FLAGS, r_alpha, r_beta, r_gamma, r_lambda,do_in, do_out, dims,pi_0=None):
        self.variables = {}
        self.wd_loss = 0

        if FLAGS.nonlin.lower() == 'elu':
            self.nonlin = tf.nn.elu
        else:
            self.nonlin = tf.nn.relu

        self._build_graph(x, t, y_ , p_t, s_mode1,s_mode2,FLAGS, r_alpha, r_beta, r_gamma, r_lambda, do_in, do_out, dims,pi_0)

    def _add_variable(self, var, name):
        ''' Adds variables to the internal track-keeper '''
        basename = name
        i = 0
        while name in self.variables:
            name = '%s_%d' % (basename, i) #@TODO: not consistent with TF internally if changed
            i += 1

        self.variables[name] = var

    def _create_variable(self, var, name):
        ''' Create and adds variables to the internal track-keeper '''

        var = tf.Variable(var, name=name)
        self._add_variable(var, name)
        return var

    def _create_variable_with_weight_decay(self, initializer, name, wd):
        ''' Create and adds variables to the internal track-keeper
            and adds it to the list of weight decayed variables '''
        var = self._create_variable(initializer, name)
        self.wd_loss += wd*tf.nn.l2_loss(var)
        return var

    def _build_graph(self, x, t, y_ , p_t, s_mode1,s_mode2,FLAGS, r_alpha, r_beta, r_gamma, r_lambda,do_in, do_out, dims,pi_0):
 

        self.x = x
        self.t = t
        self.y_ = y_
        self.t_one_hot = tf.one_hot(tf.to_int32(tf.squeeze(t)),2)
        self.y_one_hot = tf.one_hot(tf.to_int32(tf.squeeze(y_)),2)

        self.p_t = p_t
        self.s_mode1 = s_mode1
        self.s_mode2 = s_mode2
        self.r_alpha = r_alpha
        self.r_beta = r_beta
        self.r_gamma = r_gamma
        self.r_lambda = r_lambda
        self.do_in = do_in
        self.do_out = do_out
        
        dim_input = dims[0]
        dim_in = dims[1]
        dim_out = dims[2]
        
        weights_in = []
        biases_in = []
        kl = tf.keras.losses.KLDivergence()
        if FLAGS.n_in == 0 or (FLAGS.n_in == 1 and FLAGS.varsel):
            dim_in = dim_input
        if FLAGS.n_out == 0:
            if FLAGS.split_output == False:
                dim_out = dim_in+1
            else:
                dim_out = dim_in

        if FLAGS.batch_norm:
            self.bn_biases = []
            self.bn_scales = []

        ''' Construct input/representation layers '''
       
        with tf.name_scope("distangle"):
            h_rep_Z, h_rep_norm_Z, weights_in_Z, biases_in_Z = self._build_latent_graph(self.x,dim_input, dim_in, dim_out, FLAGS)
            h_rep_C, h_rep_norm_C, weights_in_C, biases_in_C = self._build_latent_graph(self.x,dim_input, dim_in, dim_out, FLAGS)
            h_rep_A, h_rep_norm_A, weights_in_A, biases_in_A = self._build_latent_graph(self.x,dim_input, dim_in, dim_out, FLAGS)
            h_rep_T, h_rep_norm_T, weights_in_T, biases_in_T = self._build_latent_graph(tf.concat((h_rep_norm_Z, h_rep_norm_C), axis=1),dim_in*2, dim_in, dim_out, FLAGS)
            h_rep_Y, h_rep_norm_Y, weights_in_Y, biases_in_Y = self._build_latent_graph(tf.concat((h_rep_norm_A, h_rep_norm_C), axis=1),dim_in*2, dim_in, dim_out, FLAGS)
            h_rep = tf.concat((h_rep_Z, h_rep_C, h_rep_A), axis=1)
            h_rep_norm = tf.concat((h_rep_norm_Z, h_rep_norm_C, h_rep_norm_A), axis=1)
        # self.zero_vector = tf.zeros([tf.shape(h_rep_norm)[0],dim_in],tf.float32)
        weights_in = weights_in_Z + weights_in_C + weights_in_A +weights_in_T+weights_in_Y
        biases_in = biases_in_Z + biases_in_C + biases_in_A + biases_in_T+ biases_in_Y
        self.weights_in_Z = weights_in_Z
        self.weights_in_C = weights_in_C
        self.weights_in_A = weights_in_A
        self.weights_in_T = weights_in_T
        self.weights_in_Y = weights_in_Y
        
        ''' Construct treatment classifier '''
        with tf.name_scope("treatment"):
            t_pre, W_t_out, W_t_pre= self._build_treatment_graph(h_rep_norm_T, dim_in,dim_out,do_out,FLAGS)
            t_z_pre, W_t_z_out, W_t_z_pre= self._build_treatment_graph(h_rep_norm_Z, dim_in,dim_out,do_out,FLAGS)
            t_c_pre, W_t_c_out, W_t_c_pre= self._build_treatment_graph(h_rep_norm_C, dim_in,dim_out,do_out,FLAGS)
            self.W_t_out = W_t_out
            self.W_t_pre = W_t_pre
            self.W_t_z_out = W_t_z_out
            self.W_t_z_pre = W_t_z_pre
            self.W_t_c_out = W_t_c_out
            self.W_t_c_pre = W_t_c_pre
            sigma_t = tf.nn.softmax(t_pre)   
            sigma_t_c = tf.nn.softmax(t_c_pre)
            sigma_t_z = tf.nn.softmax(t_z_pre)   
          


            s = sigma_t[:,1]
        ''' Construct outcome classifier '''
        with tf.name_scope("outcome"):
            
            i0,i1,y_pre, y0, y1, weights_out, weights_pred, _, _, _, _ = self._build_output_graph(
            h_rep_norm_Y, t, dim_in, dim_out, do_out, FLAGS)
            _,_,y_pre_c, y0_c, y1_c, weights_out_c, weights_pred_c, _, _, _, _ = self._build_output_graph(
            h_rep_norm_C, t, dim_in, dim_out, do_out, FLAGS)
            _,_,y_pre_a, y0_a, y1_a, weights_out_a, weights_pred_a, _, _, _, _ = self._build_output_graph(
            h_rep_norm_A, t, dim_in, dim_out, do_out, FLAGS)
            sigma_y = tf.nn.softmax(y_pre)   
            sigma_y_0 = tf.nn.softmax(y0)
            sigma_y_1 = tf.nn.softmax(y1)
            sigma_y_c = tf.nn.softmax(y_pre_c)
            sigma_y_a = tf.nn.softmax(y_pre_a)
  
        
        ''' Compute loss '''
        ### prediction loss
        ## treatment loss
        risk_t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=t_pre, labels=self.t_one_hot))
        ## outcome loss
        pi0 = tf.reduce_sum(tf.multiply(self.t_one_hot, sigma_t_c),-1)
        if pi_0 == None:  # pi_0 not provided from file
            self.pi_0 = pi0
        else:
            self.pi_0 = pi_0       

        # Compute sample reweighting '''
        if FLAGS.reweight_sample:
            w_t = t / (2. * p_t)
            w_c = (1. - t) / (2. * (1. - p_t))            
            sample_weight = 1. * (1. + (1. - self.pi_0) / self.pi_0 * (p_t / (1. - p_t)) ** (2. * t - 1.)) * (w_t + w_c)

        else:
            sample_weight = 1.0
        self.sample_weight = sample_weight
        if FLAGS.twoStage:
     
            y = (self.s_mode1*s + self.s_mode2*(1-s))*sigma_y_1[:,1]+((1-self.s_mode1)*s + (1-self.s_mode2)*(1-s))*sigma_y_0[:,1]
                
        else:
            y = sigma_y[:,1]  
                  
        res = y_*tf.log(y) + (1.0-y_)*tf.log(1.0-y)

    
        risk = -tf.reduce_mean(sample_weight*res)
        pred_error = -tf.reduce_mean(res)
        
    
        ### disentangle loss
        ## disentangle A
        dis_a,imb_dist = self._calculate_disc(h_rep_norm_A, r_gamma, FLAGS)
        ## disentangle C
        ce_cy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pre_c, labels=self.y_one_hot))
        ce_ay = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pre_a, labels=self.y_one_hot))
        kl_ca = (kl(sigma_y_c, sigma_y_a)+kl(sigma_y_a, sigma_y_c))/2
        kl_cy = kl(sigma_y, sigma_y_c)
        kl_ay = kl(sigma_y, sigma_y_a)
        dis_c = ce_cy+ce_ay+kl_ca+kl_cy+kl_ay
        ## disentangle Z
        ce_zt = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=t_z_pre, labels=self.t_one_hot))
        ce_ct = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=t_c_pre, labels=self.t_one_hot))
        kl_cz = (kl(sigma_t_c, sigma_t_z)+kl(sigma_t_z, sigma_t_c))/2
        kl_ct = kl(sigma_t, sigma_t_c)
        kl_zt = kl(sigma_t, sigma_t_z)
        dis_z = ce_zt+ce_ct+kl_cz+kl_ct+kl_zt        
        

        ''' Regularization '''
        if FLAGS.p_lambda>0 and FLAGS.rep_weight_decay:
            for i in range(0, FLAGS.n_in):
                if not (FLAGS.varsel and i==0): # No penalty on W in variable selection
                    self.wd_loss += tf.nn.l2_loss(weights_in[i])   
        
        
        

        ''' Total error '''
        tot_error = risk

        if FLAGS.p_alpha>0:
            tot_error = tot_error + r_alpha * risk_t
        if FLAGS.p_beta>0:
            tot_error = tot_error + r_beta * (dis_c+dis_z)
        
        if FLAGS.p_gamma>0:
            tot_error = tot_error + dis_a

        if FLAGS.p_lambda>0:
            tot_error = tot_error + r_lambda*self.wd_loss


        if FLAGS.varsel:
            self.w_proj = tf.placeholder("float", shape=[dim_input], name='w_proj')
            self.projection = weights_in[0].assign(self.w_proj)

        self.output = y
        self.tot_loss = tot_error
        self.imb_loss = dis_a
        self.imb_dist = imb_dist
        self.t_loss = risk_t
        self.indep_loss = dis_z+dis_c
        self.pred_loss = pred_error
        self.weights_in = weights_in
        self.weights_out = weights_out
        self.weights_pred = weights_pred
        self.h_rep = h_rep
        self.h_rep_norm = h_rep_norm
        self.i0=i0
        self.i1=i1
        self.y0=y0
        self.y1=y1
        self.t_pre = t_pre
        self.biases_in = biases_in
        self.sigma_y = sigma_y
        self.sigma_t = sigma_t
        self.kl_zc = kl_cz
        self.h_rep_norm_C = h_rep_norm_C
        self.h_rep_norm_A = h_rep_norm_A
        self.h_rep_norm_Z = h_rep_norm_Z
    
    def _calculate_disc(self, h_rep_norm, coef, FLAGS):
        t = self.t

        if FLAGS.use_p_correction:
            p_ipm = self.p_t
        else:
            p_ipm = 0.5

        if FLAGS.imb_fun == 'mmd2_rbf':
            imb_dist = mmd2_rbf(h_rep_norm, t, p_ipm, FLAGS.rbf_sigma)
            imb_error = coef * imb_dist
        elif FLAGS.imb_fun == 'mmd2_lin':
            imb_dist = mmd2_lin(h_rep_norm, t, p_ipm)
            imb_error = coef * mmd2_lin(h_rep_norm, t, p_ipm)
        elif FLAGS.imb_fun == 'mmd_rbf':
            imb_dist = tf.abs(mmd2_rbf(h_rep_norm, t, p_ipm, FLAGS.rbf_sigma))
            imb_error = safe_sqrt(tf.square(coef) * imb_dist)
        elif FLAGS.imb_fun == 'mmd_lin':
            imb_dist = mmd2_lin(h_rep_norm, t, p_ipm)
            imb_error = safe_sqrt(tf.square(coef) * imb_dist)
        elif FLAGS.imb_fun == 'wass':
            imb_dist, imb_mat = wasserstein(h_rep_norm, t, p_ipm, lam=FLAGS.wass_lambda, its=FLAGS.wass_iterations,
                                            sq=False, backpropT=FLAGS.wass_bpt)
            imb_error = coef * imb_dist
            self.imb_mat = imb_mat  # FOR DEBUG
        elif FLAGS.imb_fun == 'wass2':
            imb_dist, imb_mat = wasserstein(h_rep_norm, t, p_ipm, lam=FLAGS.wass_lambda, its=FLAGS.wass_iterations,
                                            sq=True, backpropT=FLAGS.wass_bpt)
            imb_error = coef * imb_dist
            self.imb_mat = imb_mat  # FOR DEBUG
        else:
            imb_dist = lindisc(h_rep_norm, t, p_ipm)
            imb_error = coef * imb_dist

        return imb_error, imb_dist
    def _build_latent_graph(self, input, dim_input, dim_in, dim_out, FLAGS):
        weights_in = []
        biases_in = []

        h_in = [input]
        for i in range(0, FLAGS.n_in):
            if i == 0:
                ''' If using variable selection, first layer is just rescaling'''
                if FLAGS.varsel:
                    weights_in.append(tf.Variable(1.0 / dim_input * tf.ones([dim_input])))
                else:
                    weights_in.append(tf.Variable(
                        tf.random_normal([dim_input, dim_in], stddev=FLAGS.weight_init / np.sqrt(dim_input))))
            else:
                weights_in.append(
                    tf.Variable(tf.random_normal([dim_in, dim_in], stddev=FLAGS.weight_init / np.sqrt(dim_in))))

            ''' If using variable selection, first layer is just rescaling'''
            if FLAGS.varsel and i == 0:
                biases_in.append([])
                h_in.append(tf.mul(h_in[i], weights_in[i]))
            else:
                biases_in.append(tf.Variable(tf.zeros([1, dim_in])))
                z = tf.matmul(h_in[i], weights_in[i]) + biases_in[i]

                if FLAGS.batch_norm:
                    batch_mean, batch_var = tf.nn.moments(z, [0])

                    if FLAGS.normalization == 'bn_fixed':
                        z = tf.nn.batch_normalization(z, batch_mean, batch_var, 0, 1, 1e-3)
                    else:
                        self.bn_biases.append(tf.Variable(tf.zeros([dim_in])))
                        self.bn_scales.append(tf.Variable(tf.ones([dim_in])))
                        z = tf.nn.batch_normalization(z, batch_mean, batch_var, self.bn_biases[-1], self.bn_scales[-1],1e-3)

                h_in.append(self.nonlin(z))
                h_in[i + 1] = tf.nn.dropout(h_in[i + 1], self.do_in)

        h_rep = h_in[len(h_in) - 1]

        if FLAGS.normalization == 'divide':
            h_rep_norm = h_rep / safe_sqrt(tf.reduce_sum(tf.square(h_rep), axis=1, keep_dims=True))
        else:
            h_rep_norm = 1.0 * h_rep

        return h_rep, h_rep_norm, weights_in, biases_in

    def _build_output(self, rep, dim_in, dim_out, do_out, FLAGS):
        rep =[rep]
   
        
        dims = [dim_in] + ([dim_out]*FLAGS.n_out)

        weights_out = []; biases_out = []

        for i in range(0, FLAGS.n_out):
            wo = self._create_variable_with_weight_decay(
                    tf.random_normal([dims[i], dims[i+1]],
                        stddev=FLAGS.weight_init/np.sqrt(dims[i])),
                    'w_out_%d' % i, 1.0)
            weights_out.append(wo)

            biases_out.append(tf.Variable(tf.zeros([1,dim_out])))
            z = tf.matmul(rep[i], weights_out[i]) + biases_out[i]
           
            # No batch norm on output because p_cf != p_f

            rep.append(self.nonlin(z))
           
            rep[i+1] = tf.nn.dropout(rep[i+1], do_out)
            

        weights_pred = self._create_variable(tf.random_normal([dim_out,2],
            stddev=FLAGS.weight_init/np.sqrt(dim_out)), 'w_pred')
        bias_pred = self._create_variable(tf.zeros([1,2]), 'b_pred')

        if FLAGS.varsel or FLAGS.n_out == 0:
            self.wd_loss += tf.nn.l2_loss(tf.slice(weights_pred,[0,0],[dim_out-1,1])) #don't penalize treatment coefficient
        else:
            self.wd_loss += tf.nn.l2_loss(weights_pred)

        ''' Construct linear classifier '''
        h_pred = rep[-1]

        y = tf.matmul(h_pred, weights_pred)+bias_pred
      

        return y,weights_out, weights_pred

    def _build_output_graph(self, rep, t, dim_in, dim_out, do_out, FLAGS):
        ''' Construct output/regression layers '''
        if FLAGS.split_output:

            i0 = tf.to_int32(tf.where(t < 1)[:,0])
            i1 = tf.to_int32(tf.where(t > 0)[:,0])
            
            
            rep0 = tf.gather(rep, i0)
            rep1 = tf.gather(rep, i1)

            y0, weights_out0, weights_pred0 = self._build_output(rep, dim_in, dim_out, do_out, FLAGS)
            y1,weights_out1, weights_pred1 = self._build_output(rep, dim_in, dim_out, do_out, FLAGS)

            y = tf.dynamic_stitch([i0, i1], [tf.gather(y0, i0), tf.gather(y1, i1)])
        

            weights_out = weights_out0 + weights_out1
            weights_pred = weights_pred0 + weights_pred1
        else:

            h_input = tf.concat(1,[rep, t])
   
            y, weights_out, weights_pred = self._build_output(h_input,dim_in+1, dim_out, do_out, FLAGS)
            
            h0_input = tf.concat(1,[rep, t-t])

            y0, weights_out0, weights_pred0 = self._build_output(h0_input,  dim_in+1, dim_out, do_out, FLAGS)
            
            h1_input = tf.concat(1,[rep, t-t+1])
      
            y1, weights_out1, weights_pred1 = self._build_output(h1_input,  dim_in+1, dim_out, do_out, FLAGS)

        return  i0,i1,y, y0, y1, weights_out, weights_pred, weights_out0, weights_pred0, weights_out1, weights_pred1
    
    def _build_treatment_graph(self, rep,dim_in,dim_out,do_out,FLAGS):

        rep =[rep]
        
        dims = [dim_in] + ([128,64])

        weights_out = []; biases_out = []

        for i in range(0, 2):
            wo = self._create_variable_with_weight_decay(
                    tf.random_normal([dims[i], dims[i+1]],
                        stddev=FLAGS.weight_init/np.sqrt(dims[i])),
                    'w_t_out_%d' % i, 1.0)
            weights_out.append(wo)

            biases_out.append(tf.Variable(tf.zeros([1,dims[i+1]])))
            z = tf.matmul(rep[i], weights_out[i]) + biases_out[i]
            # No batch norm on output because p_cf != p_f

            rep.append(self.nonlin(z))
            rep[i+1] = tf.nn.dropout(rep[i+1], do_out)

        weights_pred = self._create_variable(tf.random_normal([64,2],
            stddev=FLAGS.weight_init/np.sqrt(64)), 'w_t_pred')
        bias_pred = self._create_variable(tf.zeros([1,2]), 'b_t_pred')

        if FLAGS.varsel or FLAGS.n_out == 0:
            self.wd_loss += tf.nn.l2_loss(tf.slice(weights_pred,[0,0],[dim_out-1,1])) #don't penalize treatment coefficient
        else:
            self.wd_loss += tf.nn.l2_loss(weights_pred)

        ''' Construct linear classifier '''
        h_pred = rep[-1]
   
        t_pre = tf.matmul(h_pred, weights_pred)+bias_pred


        return t_pre, weights_out, weights_pred
    


def trainNet(Net, sess, train_first, train_data, val_data, test_data, FLAGS, logfile, _logfile, exp):
    n_train = len(train_data['x'])
    p_treated = np.mean(train_data['t'])

    dict_factual = {Net.x: train_data['x'], Net.t: train_data['t'], Net.y_: train_data['yf'], \
            Net.s_mode1: 1., Net.s_mode2: 0.,Net.do_in: 1.0, Net.do_out: 1.0, Net.r_alpha: FLAGS.p_alpha, \
            Net.r_beta: FLAGS.p_beta,Net.r_gamma: FLAGS.p_gamma, \
            Net.r_lambda: FLAGS.p_lambda, Net.p_t: p_treated}

    dict_valid = {Net.x: val_data['x'], Net.t: val_data['t'], Net.y_: val_data['yf'], \
            Net.s_mode1: 1., Net.s_mode2: 0., Net.do_in: 1.0, Net.do_out: 1.0, Net.r_alpha: FLAGS.p_alpha, \
            Net.r_beta: FLAGS.p_beta,Net.r_gamma: FLAGS.p_gamma, \
            Net.r_lambda: FLAGS.p_lambda, Net.p_t: p_treated}

    dict_cfactual = {Net.x: train_data['x'],  Net.t: 1-train_data['t'], Net.y_: train_data['ycf'], \
            Net.s_mode1: 0., Net.s_mode2: 1., Net.do_in: 1.0, Net.do_out: 1.0}

    ''' Initialize TensorFlow variables '''
    sess.run(tf.global_variables_initializer())
    objnan = False

    

    obj_val_best = 99999
    obj_val = {'best':99999, 'ate_train': None, 'ate_test': None, 'itr': 0,
            'hat_yf_train': None, 'hat_ycf_train': None, 'hat_mu0_train': None, 'hat_mu1_train': None , 
            'hat_yf_test': None, 'hat_ycf_test': None, 'hat_mu0_test': None, 'hat_mu1_test': None }

    final   = {'best':99999, 'ate_train': None, 'ate_test': None, 'itr': 0,
            'hat_yf_train': None, 'hat_ycf_train': None, 'hat_mu0_train': None, 'hat_mu1_train': None , 
            'hat_yf_test': None, 'hat_ycf_test': None, 'hat_mu0_test': None, 'hat_mu1_test': None }

    ''' Train for multiple iterations '''
    for i in range(FLAGS.iterations):
        ''' Fetch sample '''
        I = random.sample(range(0, n_train), FLAGS.batch_size)
        x_batch = train_data['x'][I,:]
        t_batch = train_data['t'][I]
        y_batch = train_data['yf'][I]
    
  
        if not objnan:
            sess.run(train_first, feed_dict={Net.x: x_batch, Net.t: t_batch, \
                Net.y_: y_batch, Net.s_mode1: 1, Net.s_mode2: 0, Net.do_in: FLAGS.dropout_in, Net.do_out: FLAGS.dropout_out, \
                Net.r_alpha: FLAGS.p_alpha, Net.r_lambda: FLAGS.p_lambda, \
                Net.r_beta: FLAGS.p_beta,Net.r_gamma: FLAGS.p_gamma, \
                Net.p_t: p_treated})
  

        ''' Project variable selection weights '''
        if FLAGS.varsel:
            wip = simplex_project(sess.run(Net.weights_in[0]), 1)
            sess.run(Net.projection, feed_dict={Net.w_proj: wip})

        if i % FLAGS.output_delay == 0 or i==FLAGS.iterations-1:
            obj_loss = sess.run(Net.tot_loss, feed_dict=dict_factual)
          
            rep = sess.run(Net.h_rep_norm, feed_dict={Net.x: train_data['x'], Net.do_in: 1.0})
        


            valid_obj = sess.run(Net.tot_loss, feed_dict=dict_valid)

            if np.isnan(obj_loss):
                log(logfile,'Experiment %d: Objective is NaN. Skipping.' % exp)
                log(_logfile,'Experiment %d: Objective is NaN. Skipping.' % exp,False)
                objnan = True
            train_ate = train_data['yf']-train_data['ycf']
            train_ate[train_data['t']<1] = -train_ate[train_data['t']<1]
            train_ate_value = np.mean(train_ate)

            y_pred_f = sess.run([Net.output], feed_dict={Net.x: train_data['x'],  \
                Net.t: train_data['t'], Net.s_mode1: 1., Net.s_mode2: 0., Net.do_in: 1.0, Net.do_out: 1.0})
            y_pred_cf = sess.run([Net.output], feed_dict={Net.x: train_data['x'], \
                Net.t: 1-train_data['t'], Net.s_mode1: 0., Net.s_mode2: 1., Net.do_in: 1.0, Net.do_out: 1.0})
            y_pred_mu0= sess.run([Net.output], feed_dict={\
                Net.x: train_data['x'], Net.t: train_data['t']-train_data['t'], Net.s_mode1: 0., Net.s_mode2: 0., Net.do_in: 1.0, \
                Net.do_out: 1.0})
            y_pred_mu1 = sess.run([Net.output], feed_dict={\
                Net.x: train_data['x'], Net.t: 1-train_data['t']+train_data['t'], Net.s_mode1: 1., Net.s_mode2: 1., Net.do_in: 1.0, \
                Net.do_out: 1.0})
        
            train_pred_ate = np.mean(y_pred_mu1) - np.mean(y_pred_mu0)

            test_ate = test_data['yf']-test_data['ycf']
            test_ate[test_data['t']<1] = -test_ate[test_data['t']<1]
            test_ate_value = np.mean(test_ate)
            y_pred_f_test = sess.run(Net.output, feed_dict={Net.x: test_data['x'],  Net.t: test_data['t'], Net.s_mode1: 1., Net.s_mode2: 0.,Net.do_in: 1.0, Net.do_out: 1.0})
            y_pred_cf_test = sess.run(Net.output, feed_dict={Net.x: test_data['x'],  Net.t: 1-test_data['t'], Net.s_mode1: 0., Net.s_mode2: 1.,Net.do_in: 1.0, Net.do_out: 1.0})
            y_pred_mu0_test = sess.run(Net.output, feed_dict={Net.x: test_data['x'], Net.t: test_data['t']-test_data['t'], Net.s_mode1: 0., Net.s_mode2: 0.,Net.do_in: 1.0, Net.do_out: 1.0})
            y_pred_mu1_test = sess.run(Net.output, feed_dict={Net.x: test_data['x'], Net.t: 1-test_data['t']+test_data['t'], Net.s_mode1: 1., Net.s_mode2: 1.,Net.do_in: 1.0, Net.do_out: 1.0})
            test_pred_ate = np.mean(y_pred_mu1_test) - np.mean(y_pred_mu0_test)

            final = {'ate_train': np.abs(train_ate_value-train_pred_ate), 'ate_test': np.abs(test_ate_value-test_pred_ate), 'itr': i,
                'hat_yf_train': y_pred_f, 'hat_ycf_train': y_pred_cf, 'hat_mu0_train': y_pred_mu0, 'hat_mu1_train': y_pred_mu1, 
                'hat_yf_test': y_pred_f_test, 'hat_ycf_test': y_pred_cf_test, 'hat_mu0_test': y_pred_mu0_test, 'hat_mu1_test': y_pred_mu1_test }
            

            if valid_obj < obj_val_best:
                obj_val_best = valid_obj
                obj_val = {'best':valid_obj, 'ate_train': np.abs(train_ate_value-train_pred_ate), 'ate_test': np.abs(test_ate_value-test_pred_ate), 'itr': i,
                    'hat_yf_train': y_pred_f, 'hat_ycf_train': y_pred_cf, 'hat_mu0_train': y_pred_mu0, 'hat_mu1_train': y_pred_mu1, 
                    'hat_yf_test': y_pred_f_test, 'hat_ycf_test': y_pred_cf_test, 'hat_mu0_test': y_pred_mu0_test, 'hat_mu1_test': y_pred_mu1_test,
                   }


            loss_str = str(i) + '\tObj: %.3f,\tValObj: %.2f,\tate_train: %.3f,\tate_test: %.3f' % \
                (obj_loss, valid_obj, final['ate_train'], final['ate_test'])
            log(logfile, loss_str)
            log(_logfile, loss_str, False)

    return  obj_val, final

def run(exp, args, dataDir, resultDir, train, val, test, device):

    tf.reset_default_graph()
    random.seed(args.seed)
    tf.compat.v1.set_random_seed(args.seed)
    np.random.seed(args.seed)

    logfile = f'{resultDir}/log_VSDD.txt'
    _logfile = f'{resultDir}/VSDD.txt'
    alpha, beta, gamma,lambda_,iteration,lrate,output_delay = \
        args.syn_alpha, args.syn_beta, args.syn_gamma, \
            args.syn_lambda,args.iteration,args.lrate, \
                    args.output_delay

    try:
        FLAGS = get_FLAGS()
    except:
        FLAGS = tf.app.flags.FLAGS

    FLAGS.reweight_sample = 1
    FLAGS.p_alpha = alpha
    FLAGS.p_beta = beta
    FLAGS.p_gamma = gamma
    FLAGS.p_lambda = lambda_
    FLAGS.iterations = iteration
    FLAGS.output_delay = output_delay
    FLAGS.lrate= lrate


    if args.syn_twoStage:
        FLAGS.twoStage = 1
    else:
        FLAGS.twoStage = 0

    try:
        train.to_numpy()
        val.to_numpy()
        test.to_numpy()
    except:
        pass

    
    x_list = [train.x, val.x, test.x]

    train = {'x':x_list[0],
            't':train.t,        
            'yf':train.y,
            'ycf':train.f}
    val =   {'x':x_list[1],
            't':val.t,
            'yf':val.y,
            'ycf':val.f}
    test =  {'x':x_list[2],
            't':test.t,
            'yf':test.y,
            'ycf':test.f}

    log(logfile, f'exp:{exp}; lrate:{FLAGS.lrate}; alpha: {FLAGS.p_alpha}; \
        beta: {FLAGS.p_beta}; gamma: {FLAGS.p_gamma}; lambda: {FLAGS.p_lambda}; \
            iterations: {FLAGS.iterations}; reweight: {FLAGS.reweight_sample} ;\
            n_in: {FLAGS.n_in};n_out: {FLAGS.n_out}'\
                )
    log(_logfile, f'exp:{exp}; lrate:{FLAGS.lrate}; alpha: {FLAGS.p_alpha}; \
        beta: {FLAGS.p_beta}; gamma: {FLAGS.p_gamma}; lambda: {FLAGS.p_lambda};iterations: {FLAGS.iterations}; reweight: {FLAGS.reweight_sample}',False)

    # 1 
    ''' Initialize input placeholders '''
    x  = tf.placeholder("float", shape=[None, train['x'].shape[1]], name='x') # Features
    t  = tf.placeholder("float", shape=[None, 1], name='t')   # Treatent
    y_ = tf.placeholder("float", shape=[None, 1], name='y_')  # Outcome
    ''' Parameter placeholders '''
    r_alpha = tf.placeholder("float", name='r_alpha')
    r_beta = tf.placeholder("float", name='r_beta')
    r_gamma = tf.placeholder("float", name='r_gamma')
    r_lambda = tf.placeholder("float", name='r_lambda')
    do_in = tf.placeholder("float", name='dropout_in')
    do_out = tf.placeholder("float", name='dropout_out')
    p = tf.placeholder("float", name='p_treated')
    s_mode1 = tf.placeholder("float", name='s_mode1')
    s_mode2 = tf.placeholder("float", name='s_mode2')
    p = tf.placeholder("float", name='p_treated')

    dims = [train['x'].shape[1], FLAGS.dim_in, FLAGS.dim_out]
    Net = VSDD(x, t, y_, p, s_mode1,s_mode2,FLAGS, r_alpha, r_beta, r_gamma,r_lambda,do_in, do_out, dims)

    # 2
    ''' Start Session '''
 
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.compat.v1.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 
    
    # 3
    ''' Set up optimizer '''
 
    first_step = tf.compat.v1.Variable(0, trainable=False, name='first_step')
    second_step = tf.compat.v1.Variable(0, trainable=False, name='second_step')
    first_lr = tf.compat.v1.train.exponential_decay(FLAGS.lrate, first_step, FLAGS.lrate_decay_num, FLAGS.lrate_decay, staircase=True)
    second_lr = tf.compat.v1.train.exponential_decay(FLAGS.lrate, second_step, FLAGS.lrate_decay_num, FLAGS.lrate_decay, staircase=True)


    first_opt = None
    second_opt = None
    if FLAGS.optimizer == 'Adagrad':
        first_opt = tf.train.AdagradOptimizer(first_lr)
        second_opt = tf.train.AdagradOptimizer(second_lr)
    elif FLAGS.optimizer == 'GradientDescent':
        first_opt = tf.train.GradientDescentOptimizer(first_lr)
        second_opt = tf.train.GradientDescentOptimizer(second_lr)
    elif FLAGS.optimizer == 'Adam':
        first_opt = tf.compat.v1.train.AdamOptimizer(first_lr)
        second_opt = tf.compat.v1.train.AdamOptimizer(second_lr)
    else:
        first_opt = tf.compat.v1.train.RMSPropOptimizer(first_lr, FLAGS.decay)
        second_opt = tf.compat.v1.train.RMSPropOptimizer(second_lr, FLAGS.decay)
    ''' Unused gradient clipping '''
    D_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='distangle')
    O_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='outcome')
    T_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='treatment')

    DOT_vars = D_vars + O_vars + T_vars

    train_first = first_opt.minimize(Net.tot_loss, global_step=first_step, var_list=DOT_vars)


    obj_val, final = trainNet(Net, sess, train_first, train, val, test, FLAGS, logfile, _logfile, exp)
    

    return obj_val, final
