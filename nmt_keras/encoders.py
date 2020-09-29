from __future__ import print_function
from six import iteritems
try:
    import itertools.zip as zip
except ImportError:
    pass

import logging
import os
import sys
import numpy
from keras.layers import *
from keras.models import model_from_json, Model
from keras.utils import multi_gpu_model
from keras.optimizers import *
from keras.regularizers import l2, AlphaRegularizer
from keras_wrapper.cnn_model import Model_Wrapper
from keras_wrapper.extra.regularize import Regularize
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
from keras.utils.vis_utils import plot_model

def BuildEncoder(self,params):
    src_text = Input(name=self.ids_inputs[0], batch_shape=tuple([None, None]), dtype='int32')
    
    # 2. Encoder
    # 2.1. Source word embedding
    embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                            name='source_word_embedding',
                            embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                            embeddings_initializer=params['INIT_FUNCTION'],
                            trainable=self.src_embedding_weights_trainable,
                            weights=self.src_embedding_weights,
                            mask_zero=True)
    src_embedding = embedding(src_text)

    if params.get('SCALE_SOURCE_WORD_EMBEDDINGS', False):
        src_embedding = SqrtScaling(params['SOURCE_TEXT_EMBEDDING_SIZE'])(src_embedding)

    src_embedding = Regularize(src_embedding, params, name='src_embedding')

    # Get mask of source embeddings (CuDNN RNNs don't accept masks)
    src_embedding_mask = GetMask(name='source_text_mask')(src_embedding)
    src_embedding = RemoveMask()(src_embedding)

    if params['RECURRENT_INPUT_DROPOUT_P'] > 0.:
        src_embedding = Dropout(params['RECURRENT_INPUT_DROPOUT_P'])(src_embedding)

    # 2.2. BRNN encoder (GRU/LSTM)
    if params['BIDIRECTIONAL_ENCODER']:
        #CuDNNLSTM
        annotations = Bidirectional(eval(self.use_CuDNN + params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                        kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                                        recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                                        bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                                        kernel_initializer=params['INIT_FUNCTION'],
                                                                                        recurrent_initializer=params['INNER_INIT'],
                                                                                        trainable=params.get('TRAINABLE_ENCODER', True),
                                                                                        return_sequences=True),
                                    trainable=params.get('TRAINABLE_ENCODER', True),
                                    name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                    merge_mode=params.get('BIDIRECTIONAL_MERGE_MODE', 'concat'))(src_embedding)
    else:
        annotations = eval(self.use_CuDNN + params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                        kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                        recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                        bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                        kernel_initializer=params['INIT_FUNCTION'],
                                                                        recurrent_initializer=params['INNER_INIT'],
                                                                        trainable=params.get('TRAINABLE_ENCODER', True),
                                                                        return_sequences=True,
                                                                        name='encoder_' + params['ENCODER_RNN_TYPE'])(src_embedding)
    annotations = Regularize(annotations, params, name='annotations')
    # 2.3. Potentially deep encoder
    for n_layer in range(1, params['N_LAYERS_ENCODER']):

        if params['RECURRENT_INPUT_DROPOUT_P'] > 0.:
            annotations = Dropout(params['RECURRENT_INPUT_DROPOUT_P'])(annotations)

        if params['BIDIRECTIONAL_DEEP_ENCODER']:
            current_annotations = Bidirectional(eval(self.use_CuDNN + params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                                    kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                                                    recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                                                    bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                                                    kernel_initializer=params['INIT_FUNCTION'],
                                                                                                    recurrent_initializer=params['INNER_INIT'],
                                                                                                    trainable=params.get('TRAINABLE_ENCODER', True),
                                                                                                    return_sequences=True),
                                                merge_mode=params.get('BIDIRECTIONAL_MERGE_MODE', 'concat'),
                                                trainable=params.get('TRAINABLE_ENCODER', True),
                                                name='bidirectional_encoder_' + str(n_layer))(annotations)
            current_annotations = Regularize(current_annotations, params, name='annotations_' + str(n_layer))
            annotations = current_annotations if n_layer == 1 and not params['BIDIRECTIONAL_ENCODER'] else Add()([annotations, current_annotations])
        else:
            current_annotations = eval(self.use_CuDNN + params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                    kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                                    recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                                    bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                                    kernel_initializer=params['INIT_FUNCTION'],
                                                                                    recurrent_initializer=params['INNER_INIT'],
                                                                                    return_sequences=True,
                                                                                    trainable=params.get('TRAINABLE_ENCODER', True),
                                                                                    name='encoder_' + str(n_layer))(annotations)

            current_annotations = Regularize(current_annotations, params, name='annotations_' + str(n_layer))
            annotations = current_annotations if n_layer == 1 and params['BIDIRECTIONAL_ENCODER'] else Add()([annotations, current_annotations])


    # 3.2. Decoder's RNN initialization perceptrons with ctx mean
    annotations = ApplyMask(name='annotations')([annotations, src_embedding_mask])  # We may want the padded annotations
    ctx_mean = MaskedMean(name='ctx_mean')(annotations)
    if len(params['INIT_LAYERS']) > 0:
        for n_layer_init in range(len(params['INIT_LAYERS']) - 1):
            ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d' % n_layer_init,
                                kernel_initializer=params['INIT_FUNCTION'],
                                kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                bias_regularizer=l2(params['WEIGHT_DECAY']),
                                trainable=params.get('TRAINABLE_DECODER', True),
                                activation=params['INIT_LAYERS'][n_layer_init]
                                )(ctx_mean)
            ctx_mean = Regularize(ctx_mean, params, name='ctx' + str(n_layer_init))

        initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                kernel_initializer=params['INIT_FUNCTION'],
                                kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                bias_regularizer=l2(params['WEIGHT_DECAY']),
                                trainable=params.get('TRAINABLE_DECODER', True),
                                activation=params['INIT_LAYERS'][-1]
                                )(ctx_mean)
        initial_state = Regularize(initial_state, params, name='initial_state')
        #input_attentional_decoder = [state_below, annotations, initial_state]

        if 'LSTM' in params['DECODER_RNN_TYPE']:
            initial_memory = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_memory',
                                    kernel_initializer=params['INIT_FUNCTION'],
                                    kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                    bias_regularizer=l2(params['WEIGHT_DECAY']),
                                    trainable=params.get('TRAINABLE_DECODER', True),
                                    activation=params['INIT_LAYERS'][-1])(ctx_mean)
            initial_memory = Regularize(initial_memory, params, name='initial_memory')
            #input_attentional_decoder.append(initial_memory)
    encoder = Model(name='shared_encoder',inputs=src_text,outputs=[annotations,initial_state,initial_memory])
    plot_model(encoder, to_file='encoder_solo.png', show_shapes=True, show_layer_names=True)
    return encoder

def BuildDecoder(self,params):
    # 3. Decoder
    # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
    next_words = Input(name="input_state_below", batch_shape=tuple([None, None]), dtype='int32')
    annotations = Input(name="input_enc_annotations", batch_shape=tuple([None, None,256]), dtype='float')
    initial_state = Input(name="input_enc_i_state", batch_shape=tuple([None, 128]), dtype='float')
    initial_memory = Input(name="input_enc_i_memory", batch_shape=tuple([None, 128]), dtype='float')
    # 3.1.2. Target word embedding
    if params.get('TIE_EMBEDDINGS', False):
        state_below = embedding(next_words)
    else:
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trg_embedding_weights_trainable,
                                weights=self.trg_embedding_weights,
                                mask_zero=True)(next_words)

    if params.get('SCALE_TARGET_WORD_EMBEDDINGS', False):
        state_below = SqrtScaling(params['TARGET_TEXT_EMBEDDING_SIZE'])(state_below)
    state_below = Regularize(state_below, params, name='state_below')
    input_attentional_decoder = [state_below, annotations, initial_state,initial_memory]
    # 3.3. Attentional decoder
    sharedAttRNNCond = eval('Att' + params['DECODER_RNN_TYPE'] + 'Cond')(params['DECODER_HIDDEN_SIZE'],
                                                                            attention_mode=params.get('ATTENTION_MODE', 'add'),
                                                                            att_units=params.get('ATTENTION_SIZE', 0),
                                                                            kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                            recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                            conditional_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                            bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                            attention_context_wa_regularizer=l2(params['WEIGHT_DECAY']),
                                                                            attention_recurrent_regularizer=l2(params['WEIGHT_DECAY']),
                                                                            attention_context_regularizer=l2(params['WEIGHT_DECAY']),
                                                                            bias_ba_regularizer=l2(params['WEIGHT_DECAY']),
                                                                            dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                            recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                                            conditional_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                            attention_dropout=params.get('ATTENTION_DROPOUT_P', 0.),
                                                                            kernel_initializer=params['INIT_FUNCTION'],
                                                                            recurrent_initializer=params['INNER_INIT'],
                                                                            attention_context_initializer=params['INIT_ATT'],
                                                                            trainable=params.get('TRAINABLE_DECODER', True),
                                                                            return_sequences=True,
                                                                            return_extra_variables=True,
                                                                            return_states=True,
                                                                            num_inputs=len(input_attentional_decoder),
                                                                            name='decoder_Att' + params['DECODER_RNN_TYPE'] + 'Cond')

    rnn_output = sharedAttRNNCond(input_attentional_decoder)
    proj_h = rnn_output[0]
    x_att = rnn_output[1]
    alphas = rnn_output[2]
    h_state = rnn_output[3]
    if 'LSTM' in params['DECODER_RNN_TYPE']:
        h_memory = rnn_output[4]
    shared_Lambda_Permute = PermuteGeneral((1, 0, 2))

    if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0:
        alpha_regularizer = AlphaRegularizer(alpha_factor=params['DOUBLE_STOCHASTIC_ATTENTION_REG'])(alphas)

    [proj_h, shared_reg_proj_h] = Regularize(proj_h, params, shared_layers=True, name='proj_h0')

    # 3.4. Possibly deep decoder
    shared_proj_h_list = []
    shared_reg_proj_h_list = []

    h_states_list = [h_state]
    if 'LSTM' in params['DECODER_RNN_TYPE']:
        h_memories_list = [h_memory]

    for n_layer in range(1, params['N_LAYERS_DECODER']):
        current_rnn_input = [proj_h, shared_Lambda_Permute(x_att), initial_state]
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            current_rnn_input.append(initial_memory)
        shared_proj_h_list.append(eval(params['DECODER_RNN_TYPE'].replace('Conditional', '') + 'Cond')(
            params['DECODER_HIDDEN_SIZE'],
            kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
            recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
            conditional_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
            bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
            dropout=params['RECURRENT_DROPOUT_P'],
            recurrent_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
            conditional_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
            kernel_initializer=params['INIT_FUNCTION'],
            recurrent_initializer=params['INNER_INIT'],
            return_sequences=True,
            return_states=True,
            trainable=params.get('TRAINABLE_DECODER', True),
            num_inputs=len(current_rnn_input),
            name='decoder_' + params['DECODER_RNN_TYPE'].replace('Conditional', '') + 'Cond' + str(n_layer)))

        current_rnn_output = shared_proj_h_list[-1](current_rnn_input)
        current_proj_h = current_rnn_output[0]
        h_states_list.append(current_rnn_output[1])
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memories_list.append(current_rnn_output[2])
        [current_proj_h, shared_reg_proj_h] = Regularize(current_proj_h, params, shared_layers=True,
                                                            name='proj_h' + str(n_layer))
        shared_reg_proj_h_list.append(shared_reg_proj_h)

        proj_h = Add()([proj_h, current_proj_h])

    # 3.5. Skip connections between encoder and output layer
    shared_FC_mlp = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                            kernel_initializer=params['INIT_FUNCTION'],
                                            kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                            bias_regularizer=l2(params['WEIGHT_DECAY']),
                                            trainable=params.get('TRAINABLE_DECODER', True),
                                            activation='linear'),
                                    trainable=params.get('TRAINABLE_DECODER', True),
                                    name='logit_lstm')
    out_layer_mlp = shared_FC_mlp(proj_h)
    shared_FC_ctx = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                            kernel_initializer=params['INIT_FUNCTION'],
                                            kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                            bias_regularizer=l2(params['WEIGHT_DECAY']),
                                            trainable=params.get('TRAINABLE_DECODER', True),
                                            activation='linear'),
                                    trainable=params.get('TRAINABLE_DECODER', True),
                                    name='logit_ctx')
    out_layer_ctx = shared_FC_ctx(x_att)
    out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
    shared_FC_emb = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                            kernel_initializer=params['INIT_FUNCTION'],
                                            kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                            bias_regularizer=l2(params['WEIGHT_DECAY']),
                                            trainable=params.get('TRAINABLE_DECODER', True),
                                            activation='linear'),
                                    trainable=params.get('TRAINABLE_DECODER', True),
                                    name='logit_emb')
    out_layer_emb = shared_FC_emb(state_below)

    [out_layer_mlp, shared_reg_out_layer_mlp] = Regularize(out_layer_mlp, params,
                                                            shared_layers=True, name='out_layer_mlp')
    [out_layer_ctx, shared_reg_out_layer_ctx] = Regularize(out_layer_ctx, params,
                                                            shared_layers=True, name='out_layer_ctx')
    [out_layer_emb, shared_reg_out_layer_emb] = Regularize(out_layer_emb, params,
                                                            shared_layers=True, name='out_layer_emb')

    shared_additional_output_merge = eval(params['ADDITIONAL_OUTPUT_MERGE_MODE'])(name='additional_input')
    additional_output = shared_additional_output_merge([out_layer_mlp, out_layer_ctx, out_layer_emb])
    shared_activation = Activation(params.get('SKIP_VECTORS_SHARED_ACTIVATION', 'tanh'))

    out_layer = shared_activation(additional_output)

    shared_deep_list = []
    shared_reg_deep_list = []
    # 3.6 Optional deep ouput layer
    for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
        shared_deep_list.append(TimeDistributed(Dense(dimension, activation=activation,
                                                        kernel_initializer=params['INIT_FUNCTION'],
                                                        kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                                        bias_regularizer=l2(params['WEIGHT_DECAY']),
                                                        trainable=params.get('TRAINABLE_DECODER', True),
                                                        ),
                                                trainable=params.get('TRAINABLE_DECODER', True),
                                                name=activation + '_%d' % i))
        out_layer = shared_deep_list[-1](out_layer)
        [out_layer, shared_reg_out_layer] = Regularize(out_layer,
                                                        params, shared_layers=True,
                                                        name='out_layer_' + str(activation) + '_%d' % i)
        shared_reg_deep_list.append(shared_reg_out_layer)

    # 3.7. Output layer: Softmax
    shared_FC_soft = TimeDistributed(Dense(params['OUTPUT_VOCABULARY_SIZE'],
                                            activation=params['CLASSIFIER_ACTIVATION'],
                                            kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                            bias_regularizer=l2(params['WEIGHT_DECAY']),
                                            trainable=(params.get('TRAINABLE_DECODER', True) or params.get('TRAIN_ONLY_LAST_LAYER', True)),
                                            ),
                                        trainable=(params.get('TRAINABLE_DECODER', True) or params.get('TRAIN_ONLY_LAST_LAYER', True)),
                                        name="target_text_out")
                                        #name=self.ids_outputs[0])
    softout = shared_FC_soft(out_layer)

    decoder = Model(inputs=[next_words,annotations,initial_state,initial_memory],
                        outputs=softout,
                        name="target_text")
                        #name=self.ids_inputs[1])
    plot_model(decoder, to_file='decoder_solo.png', show_shapes=True, show_layer_names=True)
    return decoder