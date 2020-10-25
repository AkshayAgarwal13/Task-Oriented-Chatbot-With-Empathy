#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from parlai.scripts.interactive import Interactive
from parlai.scripts.display_data import DisplayData
from parlai.scripts.train_model import TrainModel
from parlai.scripts.display_model import DisplayModel
from parlai.scripts.eval_model import EvalModel

############################################################################################################
debug_flag = True # Use debug_flag = True for local runs
models_to_train = ['finetuned_ed', 'finetuned_cc', 'finetuned_ed_cc']
models_to_predict = ['pretrained_baseline', 'finetuned_ed', 'finetuned_cc', 'finetuned_ed_cc']

models_to_train = []
models_to_predict = ['pretrained_baseline']
############################################################################################################



if debug_flag:
    # Train parameters
    _max_train_time=2*60
    _validation_every_n_epochs=0.01
    _text_truncate=512
    _label_truncate=128
    _batchsize=12

    # Test parameters
    _num_examples = 5
else:
    # Train parameters
    _max_train_time=5*60*60
    _validation_every_n_epochs=0.25
    _text_truncate=512
    _label_truncate=128
    _batchsize=12

    # Test parameters
    _num_examples = -1

def init_model_dicts():
    pretrained_baseline = {
    'baseline_model' : 'transformer/generator', # Baseline model architecture used
    'init_model' : '',  # The initial model to use for fine-tuning
    'fine_tune_dataset' : '', # Dataset used for fine tuning (train.txt and valid.txt)
    'train_model_file' : '', # Path where the trained model is saved
    'predict_dataset' : 'customer_care', # Dataset used for predicting (test.txt)
    'predict_model_file' : 'zoo:tutorial_transformer_generator/model' # Path of the model which will be used for prediction
    }
    finetuned_ed = {
    'baseline_model' : 'transformer/generator', # Baseline model architecture used
    'init_model' : 'zoo:tutorial_transformer_generator/model', # The initial model to use for fine-tuning
    'fine_tune_dataset' : 'empathetic_dialogues', # Dataset used for fine tuning (train.txt and valid.txt)
    'train_model_file' : 'models/transformer_generative/finetuned_ed/model', # Path where the trained model is saved
    'predict_dataset' : 'customer_care', # Dataset used for predicting (test.txt)
    'predict_model_file' : 'models/transformer_generative/finetuned_ed/model' # Path of the model which will be used for prediction
    }
    finetuned_cc = {
    'baseline_model' : 'transformer/generator', # Baseline model architecture used
    'init_model' : 'zoo:tutorial_transformer_generator/model', # The initial model to use for fine-tuning
    'fine_tune_dataset' : 'customer_care', # Dataset used for fine tuning (train.txt and valid.txt)
    'train_model_file' : 'models/transformer_generative/finetuned_cc/model', # Path where the trained model is saved
    'predict_dataset' : 'customer_care', # Dataset used for predicting (test.txt)
    'predict_model_file' : 'models/transformer_generative/finetuned_cc/model' # Path of the model which will be used for prediction
    }
    finetuned_ed_cc = {
    'baseline_model' : 'transformer/generator', # Baseline model architecture used
    'init_model' : 'models/transformer_generative/finetuned_ed/model', # The initial model to use for fine-tuning
    'fine_tune_dataset' : 'customer_care', # Dataset used for fine tuning (train.txt and valid.txt)
    'train_model_file' : 'models/transformer_generative/finetuned_ed_cc/model', # Path where the trained model is saved
    'predict_dataset' : 'customer_care', # Dataset used for predicting (test.txt)
    'predict_model_file' : 'models/transformer_generative/finetuned_ed_cc/model' # Path of the model which will be used for prediction
    }

    model_dicts = {
    'pretrained_baseline' : pretrained_baseline,
    'finetuned_ed' : finetuned_ed,
    'finetuned_cc' : finetuned_cc,
    'finetuned_ed_cc' : finetuned_ed_cc
    }
    return model_dicts


def display_main(dataset):
    DisplayData.main(task=dataset, num_examples=5, datatype='train')


# Using a pretrained transformer generator model and fine tuning it on a particular dataset
def train_main(model_dict):
    TrainModel.main(
    # similar to before
    task=model_dict['fine_tune_dataset'],
    model=model_dict['baseline_model'],
    model_file=model_dict['train_model_file'],
    
    # initialize with a pretrained model
    init_model=model_dict['init_model'],
    
    # arguments we get from the pretrained model.
    # Unfortunately, these must be looked up separately for each model.
    n_heads=16, n_layers=8, n_positions=512, text_truncate=_text_truncate,
    label_truncate=_label_truncate, ffn_size=2048, embedding_size=512,
    activation='gelu', variant='xlm',
    dict_lower=True, dict_tokenizer='bpe',
    dict_file='zoo:tutorial_transformer_generator/model.dict',
    learn_positional_embeddings=True,
    
    # some training arguments, specific to this fine-tuning
    # use a small learning rate with ADAM optimizer
    lr=1e-5, optimizer='adam',
    warmup_updates=100,
    # early stopping on perplexity
    validation_metric='ppl',
    # train at most 10 minutes, and validate every 0.25 epochs
    max_train_time=_max_train_time, validation_every_n_epochs=_validation_every_n_epochs,
    
    # depend on your gpu. If you have a V100, this is good
    batchsize=_batchsize, fp16=True, fp16_impl='mem_efficient',
    
    # speeds up validation
    skip_generation=True,
    
    # helps us cram more examples into our gpu at a time
    dynamic_batching='full',
    )

def predict_main(model_dict, display=False):
    if display:
        DisplayModel.main(
        task=model_dict['predict_dataset'],
        model_file=model_dict['predict_model_file'],
        num_examples=2
        )
    else:
        EvalModel.main(
        task=model_dict['predict_dataset'],
        model_file=model_dict['predict_model_file'],
        metrics =  ['ppl','f1','accuracy'],
        num_examples=_num_examples
        )


if __name__ == "__main__":
    #display_main('customer_care')
    model_dicts = init_model_dicts()
    for model in models_to_train:
        if model in model_dicts:
            train_main(model_dicts[model])
        else:
            print('{} not defined in init_model_dicts()'.format(model))
    for model in models_to_predict:
        if model in model_dicts:
            predict_main(model_dicts[model])
        else:
            print('{} not defined in init_model_dicts()'.format(model))





