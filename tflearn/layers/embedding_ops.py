# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf

from .recurrent import retrieve_seq_length_op
from .. import variables as vs
from .. import utils
from .. import initializations


def embedding(incoming, input_dim, output_dim, validate_indices=False,
              weights_init='truncated_normal', trainable=True, restore=True,
              reuse=False, scope=None, name="Embedding"):
    """ Embedding.

    Embedding layer for a sequence of integer ids or floats.

    Input:
        2-D Tensor [samples, ids].

    Output:
        3-D Tensor [samples, embedded_ids, features].

    Arguments:
        incoming: Incoming 2-D Tensor.
        input_dim: list of `int`. Vocabulary size (number of ids).
        output_dim: list of `int`. Embedding size.
        validate_indices: `bool`. Whether or not to validate gather indices.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (see tflearn.initializations) Default: 'truncated_normal'.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: A name for this layer (optional). Default: 'Embedding'.

    """

    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 2, "Incoming Tensor shape must be 2-D"

    W_init = weights_init
    if isinstance(weights_init, str):
        W_init = initializations.get(weights_init)()

    with tf.variable_scope(scope, default_name=name, values=[incoming],
                           reuse=reuse) as scope:
        name = scope.name
        with tf.device('/cpu:0'):
            W = vs.variable("W", shape=[input_dim, output_dim],
                            initializer=W_init, trainable=trainable,
                            restore=restore)
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, W)

        inference = tf.cast(incoming, tf.int32)
        inference = tf.nn.embedding_lookup(W, inference,
                                           validate_indices=validate_indices)

    inference.W = W
    inference.scope = scope
    # Embedding doesn't support masking, so we save sequence length prior
    # to the lookup. Expand dim to 3d.
    shape = [-1] + inference.get_shape().as_list()[1:3] + [1]
    inference.seq_length = retrieve_seq_length_op(tf.reshape(incoming, shape))

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference

def embedding_sparse(incoming, input_dim, output_dim, incoming_weights=None,
    partition_strategy="mod", combiner="mean", weights_init="truncated_normal",
    trainable=True, restore=True, reuse=False, scope=None, name="EmbeddingSparse"):
    """ Embedding for sparse tensor

    Embedding layer for a sparse input.

    Input:
        N-D SparseTensor of int64 ids which N > 1 [ batch_size, indices... ].

    Output:
        N-D Tensor. Shape = output_dim + shape(incoming)[1: ]

    Arguments:
        incoming: Incoming N-D SparseTensor of int64 ids which N > 1.
        input_dim: list of `int`. Vocabulary size (number of ids).
        output_dim: list of `int`. Embedding size.
        incoming_weights: A SparseTensor of float / double weights,
            or None to indicate all weights should be taken to be 1.
        partition_strategy:
        combiner: A string specifying the reduction op.
            Currently "mean", "sqrtn" and "sum" are supported.
            Default: 'mean'
        max_norm:
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (see tflearn.initializations) Default: 'truncated_normal'.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: A name for this layer (optional). Default: 'EmbeddingSparse'.

    """
    W_init = weights_init
    if isinstance(weights_init, str):
        W_init = initializations.get(weights_init)()

    varis = [ incoming ]
    if incoming_weights is not None:
        varis.append(incoming_weights)

    with tf.variable_scope(scope, default_name=name, values=varis, reuse=reuse) as scope:
        name = scope.name
        with tf.device('/cpu:0'):
            W = vs.variable("W", shape=[input_dim, output_dim], initializer=W_init,
                            trainable=trainable, restore=restore)
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, W)
        inference = tf.cast(incoming, tf.int64)
        inference = tf.nn.embedding_lookup_sparse(W, incoming, incoming_weights,
                                                  partition_strategy=partition_strategy,
                                                  combiner=combiner)
    inference.W = W
    inference.scope = scope
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)
    return inference

