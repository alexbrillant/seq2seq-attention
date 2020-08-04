import tensorflow as tf
from neuraxle.base import ExecutionContext
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import dot, BatchNormalization, Activation, concatenate, Dense, LSTM, \
    TimeDistributed
from tensorflow.python.training.adam import AdamOptimizer

from neuraxle_tensorflow.tensorflow_v2 import Tensorflow2ModelStep


def create_model(step: Tensorflow2ModelStep, context: ExecutionContext):
    # shape: (batch_size, seq_length, input_dim)
    encoder_inputs = Input(
        shape=(None, step.hyperparams['input_dim']),
        batch_size=None,
        dtype=tf.dtypes.float32,
        name='encoder_inputs'
    )

    encoder_outputs, encoder_state_h, encoder_state_c = _create_encoder(encoder_inputs=encoder_inputs, step=step)

    decoder_outputs = _create_decoder(
        step=step,
        encoder_state_h=encoder_state_h,
        encoder_state_c=encoder_state_c,
        encoder_outputs=encoder_outputs
    )

    return Model([encoder_inputs], decoder_outputs)


def _create_encoder(encoder_inputs, step):
    encoder = LSTM(
        units=step.hyperparams['hidden_dim'],
        activation='relu',
        return_state=True,
        return_sequences=True
    )
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # encoder_outputs: [batch_size, hidden_dim]

    state_h = BatchNormalization(momentum=step.hyperparams['momentum'])(state_h)
    # state_h: [batch_size, hidden_dim]
    state_c = BatchNormalization(momentum=step.hyperparams['momentum'])(state_c)
    # state_c: [batch_size, hidden_dim]

    return encoder_outputs, state_h, state_c


def _create_decoder(encoder_outputs, encoder_state_h, encoder_state_c, step):
    decoder_inputs = tf.repeat(
        input=tf.expand_dims(encoder_state_h, axis=1),
        repeats=step.hyperparams['output_length'],
        axis=1
    )

    decoder = LSTM(
        units=step.hyperparams['hidden_dim'],
        activation='relu',
        return_state=False,
        return_sequences=True
    )

    decoder_outputs = decoder(decoder_inputs, initial_state=[encoder_state_h, encoder_state_c])

    decoder_outputs = _create_attention_mechanism(
        decoder_outputs=decoder_outputs,
        encoder_outputs=encoder_outputs,
        step=step
    )
    # (batch_size, output_length, output_dim)

    return decoder_outputs


def _create_attention_mechanism(decoder_outputs, encoder_outputs, step):
    attention = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
    attention = Activation('softmax')(attention)

    context = dot([attention, encoder_outputs], axes=[2, 1])
    context = BatchNormalization(momentum=step.hyperparams['momentum'])(context)

    decoder_combined_context = concatenate([context, decoder_outputs])

    projection = TimeDistributed(Dense(units=step.hyperparams['output_dim']))
    decoder_outputs = projection(decoder_combined_context)

    return decoder_outputs


def create_loss(step: Tensorflow2ModelStep, expected_outputs, predicted_outputs):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=expected_outputs, logits=predicted_outputs))
    regularizer = tf.keras.regularizers.L1L2(l1=step.hyperparams['l1'], l2=step.hyperparams['l2'])

    return regularizer(loss)


def create_optimizer(step: Tensorflow2ModelStep, context: ExecutionContext):
    return AdamOptimizer(learning_rate=step.hyperparams['learning_rate'])
