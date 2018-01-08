import numpy as np
import tensorflow as tf
from baselines.acktr.utils import conv, fc, dense, conv_to_fc, sample, kl_div
from baselines.common.distributions import make_pdtype
import baselines.common.tf_util as U
from pysc2.lib.features import actions


class CnnPolicy(object):
  def __init__(self,
               sess,
               ob_space,
               ac_space,
               nenv,
               nsteps,
               nstack,
               reuse=False):
    nbatch = nenv * nsteps
    nh, nw, nc = (32, 32, 3)
    ob_shape = (nbatch, nh, nw, nc * nstack)
    nact = 3  # 524
    # nsub3 = 2
    # nsub4 = 5
    # nsub5 = 10
    # nsub6 = 4
    # nsub7 = 2
    # nsub8 = 4
    # nsub9 = 500
    # nsub10 = 4
    # nsub11 = 10
    # nsub12 = 500

    # (64, 64, 13)
    # 80 * 24

    X = tf.placeholder(tf.uint8, ob_shape)  #obs
    with tf.variable_scope("model", reuse=reuse):
      with tf.variable_scope("common", reuse=reuse):
        h = conv(
            tf.cast(X, tf.float32),
            'c1',
            nf=32,
            rf=5,
            stride=1,
            init_scale=np.sqrt(2),
            pad="SAME")  # ?, 32, 32, 16
        h2 = conv(
            h,
            'c2',
            nf=64,
            rf=3,
            stride=1,
            init_scale=np.sqrt(2),
            pad="SAME")  # ?, 32, 32, 32

      with tf.variable_scope("pi1", reuse=reuse):
        h3 = conv_to_fc(h2)  # 131072
        h4 = fc(h3, 'fc1', nh=256, init_scale=np.sqrt(2))  # ?, 256
        pi_ = fc(
            h4, 'pi', nact,
            act=lambda x: x)  # ( nenv * nsteps, 524) # ?, 524
        pi = tf.nn.softmax(pi_)

        vf = fc(
            h4, 'v', 1, act=lambda x: x)  # ( nenv * nsteps, 1) # ?, 1

      # vf = tf.nn.l2_normalize(vf_, 1)

      with tf.variable_scope("xy0", reuse=reuse):
        # 1 x 1 convolution for dimensionality reduction
        pi_xy0_ = conv(
            h2, 'xy0', nf=1, rf=1, stride=1,
            init_scale=np.sqrt(2))  # (? nenv * nsteps, 32, 32, 1)
        pi_xy0__ = conv_to_fc(pi_xy0_)  # 32 x 32 => 1024
        pi_xy0 = tf.nn.softmax(pi_xy0__)


      with tf.variable_scope("xy1", reuse=reuse):
        pi_xy1_ = conv(
            h2, 'xy1', nf=1, rf=1, stride=1,
            init_scale=np.sqrt(2))  # (? nenv * nsteps, 32, 32, 1)
        pi_xy1__ = conv_to_fc(pi_xy1_)  # 32 x 32 => 1024
        pi_xy1 = tf.nn.softmax(pi_xy1__)

    v0 = vf[:, 0]
    a0 = sample(pi)
    self.initial_state = []  #not stateful

    def step(ob, *_args, **_kwargs):
      #obs, states, rewards, masks, actions, actions2, x1, y1, x2, y2, values
      _pi1, _xy0, _xy1, _v = sess.run([pi, pi_xy0, pi_xy1, v0], {X: ob})
      return _pi1, _xy0, _xy1, _v, []  #dummy state

    def value(ob, *_args, **_kwargs):
      return sess.run(v0, {X: ob})

    self.X = X
    self.pi = pi
    # self.pi_sub3 = pi_sub3
    # self.pi_sub4 = pi_sub4
    # self.pi_sub5 = pi_sub5
    # self.pi_sub6 = pi_sub6
    # self.pi_sub7 = pi_sub7
    # self.pi_sub8 = pi_sub8
    # self.pi_sub9 = pi_sub9
    # self.pi_sub10 = pi_sub10
    # self.pi_sub11 = pi_sub11
    # self.pi_sub12 = pi_sub12
    self.pi_xy0 = pi_xy0
    self.pi_xy1 = pi_xy1
    # self.pi_y0 = pi_y0
    # self.pi_x1 = pi_x1
    # self.pi_y1 = pi_y1
    # self.pi_x2 = pi_x2
    # self.pi_y2 = pi_y2
    self.vf = vf
    self.step = step
    self.value = value


  def act(self, ob):
    ac, ac_dist, logp = self._act(ob[None])
    return ac[0], ac_dist[0], logp[0]
