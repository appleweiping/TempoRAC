#@title License
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Exact notebook definitions from commit 60d48fc1b71b6126462136cad441dd76da7460d0.
# Omitted only Colab, downloads, webcam, plotting and demonstration calls.
import os
import cv2
import numpy as np
from scipy.signal import medfilt
import tensorflow.compat.v2 as tf
from tensorflow.python.training import py_checkpoint_reader

# Model definition
layers = tf.keras.layers
regularizers = tf.keras.regularizers


class ResnetPeriodEstimator(tf.keras.models.Model):
  """RepNet model."""

  def __init__(
      self,
      num_frames=64,
      image_size=112,
      base_model_layer_name='conv4_block3_out',
      temperature=13.544,
      dropout_rate=0.25,
      l2_reg_weight=1e-6,
      temporal_conv_channels=512,
      temporal_conv_kernel_size=3,
      temporal_conv_dilation_rate=3,
      conv_channels=32,
      conv_kernel_size=3,
      transformer_layers_config=((512, 4, 512),),
      transformer_dropout_rate=0.0,
      transformer_reorder_ln=True,
      period_fc_channels=(512, 512),
      within_period_fc_channels=(512, 512)):
    super(ResnetPeriodEstimator, self).__init__()

    # Model params.
    self.num_frames = num_frames
    self.image_size = image_size

    self.base_model_layer_name = base_model_layer_name

    self.temperature = temperature

    self.dropout_rate = dropout_rate
    self.l2_reg_weight = l2_reg_weight

    self.temporal_conv_channels = temporal_conv_channels
    self.temporal_conv_kernel_size = temporal_conv_kernel_size
    self.temporal_conv_dilation_rate = temporal_conv_dilation_rate

    self.conv_channels = conv_channels
    self.conv_kernel_size = conv_kernel_size
    # Transformer config in form of (channels, heads, bottleneck channels).
    self.transformer_layers_config = transformer_layers_config
    self.transformer_dropout_rate = transformer_dropout_rate
    self.transformer_reorder_ln = transformer_reorder_ln

    self.period_fc_channels = period_fc_channels
    self.within_period_fc_channels = within_period_fc_channels

    # Base ResNet50 Model.
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False, weights=None, pooling='max')
    self.base_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(self.base_model_layer_name).output)

    # 3D Conv on k Frames
    self.temporal_conv_layers = [
        layers.Conv3D(self.temporal_conv_channels,
                      self.temporal_conv_kernel_size,
                      padding='same',
                      dilation_rate=(self.temporal_conv_dilation_rate, 1, 1),
                      kernel_regularizer=regularizers.l2(self.l2_reg_weight),
                      kernel_initializer='he_normal')]
    self.temporal_bn_layers = [layers.BatchNormalization()
                               for _ in self.temporal_conv_layers]

    # Counting Module (Self-sim > Conv > Transformer > Classifier)
    self.conv_3x3_layer = layers.Conv2D(self.conv_channels,
                                        self.conv_kernel_size,
                                        padding='same',
                                        activation=tf.nn.relu)

    channels = self.transformer_layers_config[0][0]
    self.input_projection = layers.Dense(
        channels, kernel_regularizer=regularizers.l2(self.l2_reg_weight),
        activation=None)
    self.input_projection2 = layers.Dense(
        channels, kernel_regularizer=regularizers.l2(self.l2_reg_weight),
        activation=None)

    length = self.num_frames
    self.pos_encoding = tf.keras.Variable(
        tf.keras.initializers.TruncatedNormal(stddev=0.02),
        shape=[1, length, 512],
        trainable=True,
        name='pos_encoding')

    self.pos_encoding2 = tf.keras.Variable(
        tf.keras.initializers.TruncatedNormal(stddev=0.02),
        shape=[1, length, 512],
        trainable=True,
        name='pos_encoding2')

    self.transformer_layers = []
    for d_model, num_heads, dff in self.transformer_layers_config:
      self.transformer_layers.append(
          TransformerLayer(d_model, num_heads, dff,
                           self.transformer_dropout_rate,
                           self.transformer_reorder_ln))

    self.transformer_layers2 = []
    for d_model, num_heads, dff in self.transformer_layers_config:
      self.transformer_layers2.append(
          TransformerLayer(d_model, num_heads, dff,
                           self.transformer_dropout_rate,
                           self.transformer_reorder_ln))

    # Period Prediction Module.
    self.dropout_layer = layers.Dropout(self.dropout_rate)
    num_preds = self.num_frames
    self.fc_layers = []
    for channels in self.period_fc_channels:
      self.fc_layers.append(layers.Dense(
          channels, kernel_regularizer=regularizers.l2(self.l2_reg_weight),
          activation=tf.nn.relu))
    self.fc_layers.append(layers.Dense(
        num_preds, kernel_regularizer=regularizers.l2(self.l2_reg_weight)))

    # Within Period Module
    num_preds = 1
    self.within_period_fc_layers = []
    for channels in self.within_period_fc_channels:
      self.within_period_fc_layers.append(layers.Dense(
          channels, kernel_regularizer=regularizers.l2(self.l2_reg_weight),
          activation=tf.nn.relu))
    self.within_period_fc_layers.append(layers.Dense(
        num_preds, kernel_regularizer=regularizers.l2(self.l2_reg_weight)))

  def call(self, x, training=False):
    # Ensures we are always using the right batch_size during train/eval.
    batch_size = tf.shape(x)[0]
    # Conv Feature Extractor.
    x = tf.reshape(x, [-1, self.image_size, self.image_size, 3])
    x = self.base_model(x, training=training)
    h = tf.shape(x)[1]
    w = tf.shape(x)[2]
    c = tf.shape(x)[3]
    x = tf.reshape(x, [batch_size, -1, h, w, c])

    # 3D Conv to give temporal context to per-frame embeddings.
    for bn_layer, conv_layer in zip(self.temporal_bn_layers,
                                    self.temporal_conv_layers):
      x = conv_layer(x)
      x = bn_layer(x, training=training)
      x = tf.nn.relu(x)

    x = tf.reduce_max(x, [2, 3])

    # Reshape and prepare embs for output.
    final_embs = x

    # Get self-similarity matrix.
    x = get_sims(x, self.temperature)

    # 3x3 conv layer on self-similarity matrix.
    x = self.conv_3x3_layer(x)
    x = tf.reshape(x, [batch_size, self.num_frames, -1])
    within_period_x = x

    # Period prediction.
    x = self.input_projection(x)
    x += self.pos_encoding
    for transformer_layer in self.transformer_layers:
      x = transformer_layer(x, training=training)
    x = flatten_sequential_feats(x, batch_size, self.num_frames)
    for fc_layer in self.fc_layers:
      x = self.dropout_layer(x, training=training)
      x = fc_layer(x)

    # Within period prediction.
    within_period_x = self.input_projection2(within_period_x)
    within_period_x += self.pos_encoding2
    for transformer_layer in self.transformer_layers2:
      within_period_x = transformer_layer(within_period_x, training=training)
    within_period_x = flatten_sequential_feats(within_period_x,
                                               batch_size,
                                               self.num_frames)
    for fc_layer in self.within_period_fc_layers:
      within_period_x = self.dropout_layer(within_period_x, training=training)
      within_period_x = fc_layer(within_period_x)

    return x, within_period_x, final_embs

  @tf.function
  def preprocess(self, imgs):
    imgs = tf.cast(imgs, tf.float32)
    imgs -= 127.5
    imgs /= 127.5
    imgs = tf.image.resize(imgs, (self.image_size, self.image_size))
    return imgs


def get_sims(embs, temperature):
  """Calculates self-similarity between batch of sequence of embeddings."""
  batch_size = tf.shape(embs)[0]
  seq_len = tf.shape(embs)[1]
  embs = tf.reshape(embs, [batch_size, seq_len, -1])

  def _get_sims(embs):
    """Calculates self-similarity between sequence of embeddings."""
    dist = pairwise_l2_distance(embs, embs)
    sims = -1.0 * dist
    return sims

  sims = tf.map_fn(_get_sims, embs)
  sims /= temperature
  sims = tf.nn.softmax(sims, axis=-1)
  sims = tf.expand_dims(sims, -1)
  return sims


def flatten_sequential_feats(x, batch_size, seq_len):
  """Flattens sequential features with known batch size and seq_len."""
  x = tf.reshape(x, [batch_size, seq_len, -1])
  return x


# Transformer from https://www.tensorflow.org/tutorials/text/transformer .
def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.

  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    outputs: shape == (..., seq_len_q, depth_v)
    attention_weights: shape == (..., seq_len_q, seq_len_k)
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk.
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  # (..., seq_len_q, seq_len_k)
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

  outputs = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return outputs, attention_weights


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model)
  ])


class MultiHeadAttention(tf.keras.layers.Layer):
  """Multi-headed attention layer."""

  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth)."""
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(
        q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(
        k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(
        v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(
        scaled_attention,
        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(
        scaled_attention,
        (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    outputs = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return outputs, attention_weights


class TransformerLayer(tf.keras.layers.Layer):
  """Implements a single transformer layer (https://arxiv.org/abs/1706.03762).
  """

  def __init__(self, d_model, num_heads, dff,
               dropout_rate=0.1,
               reorder_ln=False):
    super(TransformerLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
    self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    self.reorder_ln = reorder_ln

  def call(self, x, training=False):
    inp_x = x

    if self.reorder_ln:
      x = self.layernorm1(x)

    # (batch_size, input_seq_len, d_model)
    attn_output, _ = self.mha(x, x, x, mask=None)
    attn_output = self.dropout1(attn_output, training=training)

    if self.reorder_ln:
      out1 = inp_x + attn_output
      x = out1
    else:
      # (batch_size, input_seq_len, d_model)
      out1 = self.layernorm1(x + attn_output)
      x = out1

    if self.reorder_ln:
      x = self.layernorm2(x)

    # (batch_size, input_seq_len, d_model)
    ffn_output = self.ffn(x)
    ffn_output = self.dropout2(ffn_output, training=training)

    if self.reorder_ln:
      out2 = out1 + ffn_output
    else:
      # (batch_size, input_seq_len, d_model)
      out2 = self.layernorm2(out1 + ffn_output)

    return out2


def pairwise_l2_distance(a, b):
  """Computes pairwise distances between all rows of a and all rows of b."""
  norm_a = tf.reduce_sum(tf.square(a), 1)
  norm_a = tf.reshape(norm_a, [-1, 1])
  norm_b = tf.reduce_sum(tf.square(b), 1)
  norm_b = tf.reshape(norm_b, [1, -1])
  dist = tf.maximum(norm_a - 2.0 * tf.matmul(a, b, False, True) + norm_b, 0.0)
  return dist


def get_repnet_model(logdir):
  """Returns a trained RepNet model.

  Args:
    logdir (string): Path to directory where checkpoint will be downloaded.

  Returns:
    model (Keras model): Trained RepNet model.
  """
  # Check if we are in eager mode.
  assert tf.executing_eagerly()

  # Define RepNet model.
  model = ResnetPeriodEstimator()
  # Pass dummy frames to build graph.
  _ = model(tf.random.uniform((1, 64, 112, 112, 3)),
            training=False)
  # tf.function for speed.
  model.call = tf.function(model.call)

  # Define checkpoint and checkpoint manager.
  ckpt = tf.train.Checkpoint(model=model)
  ckpt_manager = tf.train.CheckpointManager(
      ckpt, directory=logdir, max_to_keep=10)
  latest_ckpt = ckpt_manager.latest_checkpoint
  print('Loading from: ', latest_ckpt)
  if not latest_ckpt:
    raise ValueError('Path does not have a checkpoint to load.')
  # Restore weights.
  ckpt.restore(latest_ckpt).expect_partial()

  return model



def read_video(video_filename, width=224, height=224):
  """Read video from file."""
  cap = cv2.VideoCapture(video_filename)
  fps = cap.get(cv2.CAP_PROP_FPS)
  frames = []
  if cap.isOpened():
    while True:
      success, frame_bgr = cap.read()
      if not success:
        break
      frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
      frame_rgb = cv2.resize(frame_rgb, (width, height))
      frames.append(frame_rgb)
  frames = np.asarray(frames)
  return frames, fps



def get_score(period_score, within_period_score):
  """Combine the period and periodicity scores."""
  within_period_score = tf.nn.sigmoid(within_period_score)[:, 0]
  per_frame_periods = tf.argmax(period_score, axis=-1)
  pred_period_conf = tf.reduce_max(
      tf.nn.softmax(period_score, axis=-1), axis=-1)
  pred_period_conf = tf.where(
      tf.math.less(per_frame_periods, 3), 0.0, pred_period_conf)
  within_period_score *= pred_period_conf
  within_period_score = np.sqrt(within_period_score)
  pred_score = tf.reduce_mean(within_period_score)
  return pred_score, within_period_score


def get_counts(model, frames, strides, batch_size,
               threshold,
               within_period_threshold,
               constant_speed=False,
               median_filter=False,
               fully_periodic=False):
  """Pass frames through model and conver period predictions to count."""
  seq_len = len(frames)
  raw_scores_list = []
  scores = []
  within_period_scores_list = []

  if fully_periodic:
    within_period_threshold = 0.0

  frames = model.preprocess(frames)

  for stride in strides:
    num_batches = int(np.ceil(seq_len/model.num_frames/stride/batch_size))
    raw_scores_per_stride = []
    within_period_score_stride = []
    for batch_idx in range(num_batches):
      idxes = tf.range(batch_idx*batch_size*model.num_frames*stride,
                       (batch_idx+1)*batch_size*model.num_frames*stride,
                       stride)
      idxes = tf.clip_by_value(idxes, 0, seq_len-1)
      curr_frames = tf.gather(frames, idxes)
      curr_frames = tf.reshape(
          curr_frames,
          [batch_size, model.num_frames, model.image_size, model.image_size, 3])

      raw_scores, within_period_scores, _ = model(curr_frames)
      raw_scores_per_stride.append(np.reshape(raw_scores.numpy(),
                                              [-1, raw_scores.shape[-1]]))
      within_period_score_stride.append(np.reshape(within_period_scores.numpy(),
                                                   [-1, 1]))
    raw_scores_per_stride = np.concatenate(raw_scores_per_stride, axis=0)
    raw_scores_list.append(raw_scores_per_stride)
    within_period_score_stride = np.concatenate(
        within_period_score_stride, axis=0)
    pred_score, within_period_score_stride = get_score(
        raw_scores_per_stride, within_period_score_stride)
    scores.append(pred_score)
    within_period_scores_list.append(within_period_score_stride)

  # Stride chooser
  argmax_strides = np.argmax(scores)
  chosen_stride = strides[argmax_strides]
  raw_scores = np.repeat(
      raw_scores_list[argmax_strides], chosen_stride, axis=0)[:seq_len]
  within_period = np.repeat(
      within_period_scores_list[argmax_strides], chosen_stride,
      axis=0)[:seq_len]
  within_period_binary = np.asarray(within_period > within_period_threshold)
  if median_filter:
    within_period_binary = medfilt(np.float32(within_period_binary), 5)
    within_period_binary = within_period_binary.astype(bool)

  # Select Periodic frames
  periodic_idxes = np.where(within_period_binary)[0]

  if constant_speed:
    # Count by averaging predictions. Smoother but
    # assumes constant speed.
    scores = tf.reduce_mean(
        tf.nn.softmax(raw_scores[periodic_idxes], axis=-1), axis=0)
    max_period = np.argmax(scores)
    pred_score = scores[max_period]
    pred_period = chosen_stride * max_period
    per_frame_counts = (
        np.asarray(seq_len * [1. / pred_period]) *
        np.asarray(within_period_binary))
  else:
    # Count each frame. More noisy but adapts to changes in speed.
    pred_score = tf.reduce_mean(within_period)
    per_frame_periods = tf.argmax(raw_scores, axis=-1)
    per_frame_counts = tf.where(
        tf.math.less(per_frame_periods, 3),
        0.0,
        tf.math.divide(1.0,
                       tf.cast(chosen_stride * per_frame_periods, tf.float32)),
    )
    if median_filter:
      per_frame_counts = medfilt(per_frame_counts, 5)

    per_frame_counts *= np.asarray(within_period_binary)

    pred_period = seq_len/np.sum(per_frame_counts)

  if pred_score < threshold:
    print('No repetitions detected in video as score '
          '%0.2f is less than threshold %0.2f.'%(pred_score, threshold))
    per_frame_counts = np.asarray(len(per_frame_counts) * [0.])

  return (pred_period, pred_score, within_period,
          per_frame_counts, chosen_stride)

MAPPING_NEW_TO_OLD_LAYER_NAMES = [
    ("pos_encoding/pos_encoding","pos_encoding/.ATTRIBUTES/VARIABLE_VALUE"),
    ("pos_encoding2/pos_encoding2","pos_encoding2/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv1_conv/kernel/kernel","base_model/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv1_conv/bias/bias","base_model/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block1_preact_bn/gamma/gamma","base_model/layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block1_preact_bn/beta/beta","base_model/layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block1_preact_bn/moving_mean/moving_mean","base_model/layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block1_preact_bn/moving_variance/moving_variance","base_model/layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block1_1_conv/kernel/kernel","base_model/layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block1_1_bn/gamma/gamma","base_model/layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block1_1_bn/beta/beta","base_model/layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block1_1_bn/moving_mean/moving_mean","base_model/layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block1_1_bn/moving_variance/moving_variance","base_model/layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block1_2_conv/kernel/kernel","base_model/layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block1_2_bn/gamma/gamma","base_model/layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block1_2_bn/beta/beta","base_model/layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block1_2_bn/moving_mean/moving_mean","base_model/layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block1_2_bn/moving_variance/moving_variance","base_model/layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block1_0_conv/kernel/kernel","base_model/layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block1_0_conv/bias/bias","base_model/layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block1_3_conv/kernel/kernel","base_model/layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block1_3_conv/bias/bias","base_model/layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block2_preact_bn/gamma/gamma","base_model/layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block2_preact_bn/beta/","base_model/layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block2_preact_bn/moving_mean/moving_mean","base_model/layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block2_preact_bn/moving_variance/moving_variance","base_model/layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block2_1_conv/kernel/kernel","base_model/layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block2_1_bn/gamma/gamma","base_model/layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block2_1_bn/beta/beta","base_model/layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block2_1_bn/moving_mean/moving_mean","base_model/layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block2_1_bn/moving_variance/moving_variance","base_model/layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block2_2_conv/kernel/kernel","base_model/layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block2_2_bn/gamma/gamma","base_model/layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block2_2_bn/beta/beta","base_model/layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block2_2_bn/moving_mean/moving_mean","base_model/layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block2_2_bn/moving_variance/moving_variance","base_model/layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block2_3_conv/kernel/kernel","base_model/layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block2_3_conv/bias/bias","base_model/layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block3_preact_bn/gamma/gamma","base_model/layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block3_preact_bn/beta/beta","base_model/layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block3_preact_bn/moving_mean/moving_mean","base_model/layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block3_preact_bn/moving_variance/moving_variance","base_model/layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block3_1_conv/kernel/kernel","base_model/layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block3_1_bn/gamma/gamma","base_model/layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block3_1_bn/beta/beta","base_model/layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block3_1_bn/moving_mean/moving_mean","base_model/layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block3_1_bn/moving_variance/moving_variance","base_model/layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block3_2_conv/kernel/kernel","base_model/layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block3_2_bn/gamma/gamma","base_model/layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block3_2_bn/beta/beta","base_model/layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block3_2_bn/moving_mean/moving_mean","base_model/layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block3_2_bn/moving_variance/moving_variance","base_model/layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block3_3_conv/kernel/kernel","base_model/layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv2_block3_3_conv/bias/bias","base_model/layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block1_preact_bn/gamma/gamma","base_model/layer_with_weights-20/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block1_preact_bn/beta/beta","base_model/layer_with_weights-20/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block1_preact_bn/moving_mean/moving_mean","base_model/layer_with_weights-20/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block1_preact_bn/moving_variance/moving_variance","base_model/layer_with_weights-20/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block1_1_conv/kernel/kernel","base_model/layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block1_1_bn/gamma/gamma","base_model/layer_with_weights-22/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block1_1_bn/beta/beta","base_model/layer_with_weights-22/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block1_1_bn/moving_mean/moving_mean","base_model/layer_with_weights-22/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block1_1_bn/moving_variance/moving_variance","base_model/layer_with_weights-22/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block1_2_conv/kernel/kernel","base_model/layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block1_2_bn/gamma/gamma","base_model/layer_with_weights-24/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block1_2_bn/beta/beta","base_model/layer_with_weights-24/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block1_2_bn/moving_mean/moving_mean","base_model/layer_with_weights-24/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block1_2_bn/moving_variance/moving_variance","base_model/layer_with_weights-24/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block1_0_conv/kernel/kernel","base_model/layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block1_0_conv/bias/bias","base_model/layer_with_weights-25/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block1_3_conv/kernel/kernel","base_model/layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block1_3_conv/bias/bias","base_model/layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block2_preact_bn/gamma/gamma","base_model/layer_with_weights-27/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block2_preact_bn/beta/beta","base_model/layer_with_weights-27/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block2_preact_bn/moving_mean/moving_mean","base_model/layer_with_weights-27/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block2_preact_bn/moving_variance/moving_variance","base_model/layer_with_weights-27/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block2_1_conv/kernel/kernel","base_model/layer_with_weights-28/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block2_1_bn/gamma/gamma","base_model/layer_with_weights-29/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block2_1_bn/beta/beta","base_model/layer_with_weights-29/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block2_1_bn/moving_mean/moving_mean","base_model/layer_with_weights-29/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block2_1_bn/moving_variance/moving_variance","base_model/layer_with_weights-29/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block2_2_conv/kernel/kernel","base_model/layer_with_weights-30/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block2_2_bn/gamma/gamma","base_model/layer_with_weights-31/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block2_2_bn/beta/beta","base_model/layer_with_weights-31/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block2_2_bn/moving_mean/moving_mean","base_model/layer_with_weights-31/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block2_2_bn/moving_variance/moving_variance","base_model/layer_with_weights-31/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block2_3_conv/kernel/kernel","base_model/layer_with_weights-32/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block2_3_conv/bias/bias","base_model/layer_with_weights-32/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block3_preact_bn/gamma/gamma","base_model/layer_with_weights-33/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block3_preact_bn/beta/beta","base_model/layer_with_weights-33/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block3_preact_bn/moving_mean/moving_mean","base_model/layer_with_weights-33/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block3_preact_bn/moving_variance/moving_variance","base_model/layer_with_weights-33/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block3_1_conv/kernel/kernel","base_model/layer_with_weights-34/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block3_1_bn/gamma/gamma","base_model/layer_with_weights-35/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block3_1_bn/beta/beta","base_model/layer_with_weights-35/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block3_1_bn/moving_mean/moving_mean","base_model/layer_with_weights-35/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block3_1_bn/moving_variance/moving_variance","base_model/layer_with_weights-35/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block3_2_conv/kernel/kernel","base_model/layer_with_weights-36/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block3_2_bn/gamma/gamma","base_model/layer_with_weights-37/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block3_2_bn/beta/beta","base_model/layer_with_weights-37/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block3_2_bn/moving_mean/moving_mean","base_model/layer_with_weights-37/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block3_2_bn/moving_variance/moving_variance","base_model/layer_with_weights-37/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block3_3_conv/kernel/kernel","base_model/layer_with_weights-38/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block3_3_conv/bias/bias","base_model/layer_with_weights-38/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block4_preact_bn/gamma/gamma","base_model/layer_with_weights-39/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block4_preact_bn/beta/beta","base_model/layer_with_weights-39/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block4_preact_bn/moving_mean/moving_mean","base_model/layer_with_weights-39/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block4_preact_bn/moving_variance/moving_variance","base_model/layer_with_weights-39/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block4_1_conv/kernel/kernel","base_model/layer_with_weights-40/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block4_1_bn/gamma/gamma","base_model/layer_with_weights-41/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block4_1_bn/beta/beta","base_model/layer_with_weights-41/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block4_1_bn/moving_mean/moving_mean","base_model/layer_with_weights-41/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block4_1_bn/moving_variance/moving_variance","base_model/layer_with_weights-41/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block4_2_conv/kernel/kernel","base_model/layer_with_weights-42/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block4_2_bn/gamma/gamma","base_model/layer_with_weights-43/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block4_2_bn/beta/beta","base_model/layer_with_weights-43/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block4_2_bn/moving_mean/moving_mean","base_model/layer_with_weights-43/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block4_2_bn/moving_variance/moving_variance","base_model/layer_with_weights-43/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block4_3_conv/kernel/kernel","base_model/layer_with_weights-44/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv3_block4_3_conv/bias/bias","base_model/layer_with_weights-44/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block1_preact_bn/gamma/gamma","base_model/layer_with_weights-45/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block1_preact_bn/beta/beta","base_model/layer_with_weights-45/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block1_preact_bn/moving_mean/moving_mean","base_model/layer_with_weights-45/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block1_preact_bn/moving_variance/moving_variance","base_model/layer_with_weights-45/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block1_1_conv/kernel/kernel","base_model/layer_with_weights-46/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block1_1_bn/gamma/gamma","base_model/layer_with_weights-47/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block1_1_bn/beta/beta","base_model/layer_with_weights-47/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block1_1_bn/moving_mean/moving_mean","base_model/layer_with_weights-47/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block1_1_bn/moving_variance/moving_variance","base_model/layer_with_weights-47/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block1_2_conv/kernel/kernel","base_model/layer_with_weights-48/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block1_2_bn/gamma/gamma","base_model/layer_with_weights-49/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block1_2_bn/beta/beta","base_model/layer_with_weights-49/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block1_2_bn/moving_mean/moving_mean","base_model/layer_with_weights-49/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block1_2_bn/moving_variance/moving_variance","base_model/layer_with_weights-49/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block1_0_conv/kernel/kernel","base_model/layer_with_weights-50/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block1_0_conv/bias/bias","base_model/layer_with_weights-50/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block1_3_conv/kernel/kernel","base_model/layer_with_weights-51/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block1_3_conv/bias/bias","base_model/layer_with_weights-51/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block2_preact_bn/gamma/gamma","base_model/layer_with_weights-52/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block2_preact_bn/beta/beta","base_model/layer_with_weights-52/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block2_preact_bn/moving_mean/moving_mean","base_model/layer_with_weights-52/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block2_preact_bn/moving_variance/moving_variance","base_model/layer_with_weights-52/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block2_1_conv/kernel/kernel","base_model/layer_with_weights-53/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block2_1_bn/gamma/gamma","base_model/layer_with_weights-54/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block2_1_bn/beta/beta","base_model/layer_with_weights-54/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block2_1_bn/moving_mean/moving_mean","base_model/layer_with_weights-54/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block2_1_bn/moving_variance/moving_variance","base_model/layer_with_weights-54/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block2_2_conv/kernel/kernel","base_model/layer_with_weights-55/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block2_2_bn/gamma/gamma","base_model/layer_with_weights-56/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block2_2_bn/beta/beta","base_model/layer_with_weights-56/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block2_2_bn/moving_mean/moving_mean","base_model/layer_with_weights-56/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block2_2_bn/moving_variance/moving_variance","base_model/layer_with_weights-56/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block2_3_conv/kernel/kernel","base_model/layer_with_weights-57/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block2_3_conv/bias/bias","base_model/layer_with_weights-57/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block3_preact_bn/gamma/gamma","base_model/layer_with_weights-58/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block3_preact_bn/beta/beta","base_model/layer_with_weights-58/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block3_preact_bn/moving_mean/moving_mean","base_model/layer_with_weights-58/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block3_preact_bn/moving_variance/moving_variance","base_model/layer_with_weights-58/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block3_1_conv/kernel/kernel","base_model/layer_with_weights-59/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block3_1_bn/gamma/gamma","base_model/layer_with_weights-60/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block3_1_bn/beta/beta","base_model/layer_with_weights-60/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block3_1_bn/moving_mean/moving_mean","base_model/layer_with_weights-60/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block3_1_bn/moving_variance/moving_variance","base_model/layer_with_weights-60/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block3_2_conv/kernel/kernel","base_model/layer_with_weights-61/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block3_2_bn/gamma/gamma","base_model/layer_with_weights-62/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block3_2_bn/beta/beta","base_model/layer_with_weights-62/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block3_2_bn/moving_mean/moving_mean","base_model/layer_with_weights-62/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block3_2_bn/moving_variance/moving_variance","base_model/layer_with_weights-62/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block3_3_conv/kernel/kernel","base_model/layer_with_weights-63/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("conv4_block3_3_conv/bias/bias","base_model/layer_with_weights-63/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/conv3d_6/kernel/kernel","temporal_conv_layers/0/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/conv3d_6/bias/bias","temporal_conv_layers/0/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/batch_normalization_6/gamma/gamma","temporal_bn_layers/0/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/batch_normalization_6/beta/beta","temporal_bn_layers/0/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/batch_normalization_6/moving_mean/moving_mean","temporal_bn_layers/0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/batch_normalization_6/moving_variance/moving_variance","temporal_bn_layers/0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/conv2d_6/kernel/kernel","conv_3x3_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/conv2d_6/bias/bias","conv_3x3_layer/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/dense_66/kernel/kernel","input_projection/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/dense_66/bias/bias","input_projection/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/dense_67/kernel/kernel","input_projection2/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/dense_67/bias/bias","input_projection2/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_6/multi_head_attention_6/dense_68/kernel/kernel","transformer_layers/0/mha/wq/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_6/multi_head_attention_6/dense_68/bias/bias","transformer_layers/0/mha/wq/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_6/multi_head_attention_6/dense_69/kernel/kernel","transformer_layers/0/mha/wk/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_6/multi_head_attention_6/dense_69/bias/bias","transformer_layers/0/mha/wk/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_6/multi_head_attention_6/dense_70/kernel/kernel","transformer_layers/0/mha/wv/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_6/multi_head_attention_6/dense_70/bias/bias","transformer_layers/0/mha/wv/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_6/multi_head_attention_6/dense_71/kernel/kernel","transformer_layers/0/mha/dense/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_6/multi_head_attention_6/dense_71/bias/bias","transformer_layers/0/mha/dense/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_6/sequential_6/dense_72/kernel/kernel","transformer_layers/0/ffn/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_6/sequential_6/dense_72/bias/bias","transformer_layers/0/ffn/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_6/sequential_6/dense_73/kernel/kernel","transformer_layers/0/ffn/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_6/sequential_6/dense_73/bias/bias","transformer_layers/0/ffn/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_6/layer_normalization_12/gamma/gamma","transformer_layers/0/layernorm1/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_6/layer_normalization_12/beta/beta","transformer_layers/0/layernorm1/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_6/layer_normalization_13/gamma/gamma","transformer_layers/0/layernorm2/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_6/layer_normalization_13/beta/beta","transformer_layers/0/layernorm2/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_7/multi_head_attention_7/dense_74/kernel/kernel","transformer_layers2/0/mha/wq/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_7/multi_head_attention_7/dense_74/bias/bias","transformer_layers2/0/mha/wq/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_7/multi_head_attention_7/dense_75/kernel/kernel","transformer_layers2/0/mha/wk/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_7/multi_head_attention_7/dense_75/bias/bias","transformer_layers2/0/mha/wk/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_7/multi_head_attention_7/dense_76/kernel/kernel","transformer_layers2/0/mha/wv/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_7/multi_head_attention_7/dense_76/bias/bias","transformer_layers2/0/mha/wv/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_7/multi_head_attention_7/dense_77/kernel/kernel","transformer_layers2/0/mha/dense/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_7/multi_head_attention_7/dense_77/bias/bias","transformer_layers2/0/mha/dense/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_7/sequential_7/dense_78/kernel/kernel","transformer_layers2/0/ffn/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_7/sequential_7/dense_78/bias/bias","transformer_layers2/0/ffn/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_7/sequential_7/dense_79/kernel/kernel","transformer_layers2/0/ffn/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_7/sequential_7/dense_79/bias/bias","transformer_layers2/0/ffn/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_7/layer_normalization_14/gamma/gamma","transformer_layers2/0/layernorm1/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_7/layer_normalization_14/beta/beta","transformer_layers2/0/layernorm1/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_7/layer_normalization_15/gamma/gamma","transformer_layers2/0/layernorm2/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/transformer_layer_7/layer_normalization_15/beta/beta","transformer_layers2/0/layernorm2/beta/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/dense_80/kernel/kernel","fc_layers/0/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/dense_80/bias/bias","fc_layers/0/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/dense_81/kernel/kernel","fc_layers/1/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/dense_81/bias/bias","fc_layers/1/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/dense_82/kernel/kernel","fc_layers/2/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/dense_82/bias/bias","fc_layers/2/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/dense_83/kernel/kernel","within_period_fc_layers/0/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/dense_83/bias/bias","within_period_fc_layers/0/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/dense_84/kernel/kernel","within_period_fc_layers/1/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/dense_84/bias/bias","within_period_fc_layers/1/bias/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/dense_85/kernel/kernel","within_period_fc_layers/2/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
    ("resnet_period_estimator_6/dense_85/bias/bias","within_period_fc_layers/2/bias/.ATTRIBUTES/VARIABLE_VALUE"),
]


def load_ckpt_with_custom_layer_mapping(model, logdir, custom_mappings):
  """Remap checkpoint variables to handle Keras model name changes correctly."""
  ckpt = tf.train.Checkpoint(model=model)
  ckpt_manager = tf.train.CheckpointManager(
      ckpt, directory=logdir, max_to_keep=10)
  latest_ckpt = ckpt_manager.latest_checkpoint
  checkpoint_reader = py_checkpoint_reader.NewCheckpointReader(latest_ckpt)
  for ((new_layer_name, old_layer_name), weight) in zip(custom_mappings, model.weights):
    old_layer_name = 'model/' + old_layer_name
    old_weight = checkpoint_reader.get_tensor(old_layer_name)
    if weight.value.shape.num_elements() != np.prod(old_weight.shape):
      raise ValueError('Shape mismatch in layer (%s, %s)' % (new_layer_name, old_layer_name))
    weight.assign(old_weight)

