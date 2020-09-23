# -*- coding: utf-8 -*-
from keras_wrapper.dataset import Dataset, saveDataset
from data_engine.prepare_data import keep_n_captions
from nmt_keras.model_zoo import TranslationModel
import utils
from keras_wrapper.cnn_model import loadModel
from keras_wrapper.dataset import loadDataset
from keras_wrapper.extra.callbacks import *
import json
from config import load_parameters

if __name__ == "__main__":
    ds_es = Dataset('EN', 'ES', silence=False)
    ds_fr = Dataset('EN', 'FR', silence=False)
    