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
from parlai.utils.misc import nice_report
import sys
from utils import display_results_retrieval, decode_args
############################################################################################################
# run_mode = 'local' for local runs
#          = 'azuare_fast' for faster azure run
#          = 'azure' for full azure run
#          = 'display' to just display results or run predict via DisplayModel
run_mode = 'local'

# Select models to train and predict. It can be overwritten using runtime arguments
models_to_train = ['finetuned_ed', 'finetuned_cc', 'finetuned_ed_cc']
models_to_predict = ['pretrained_baseline', 'finetuned_ed', 'finetuned_cc', 'finetuned_ed_cc']

#models_to_train = []
#models_to_predict = ['finetuned_ed']
############################################################################################################

run_modes = {
    'local' : {
        'max_train_time' : 60,
        'validation_every_n_epochs' : 0.01,
        'text_truncate' : 400,
        'label_truncate' : 150,
        # Predict parameters
        'num_examples' : 10
    },
    'azure_fast' : {
        'max_train_time' : 4*60*60,
        'validation_every_n_epochs' : 0.5,
        'text_truncate' : 400,
        'label_truncate' : 150,
        # Predict parameters
        'num_examples' : -1
    },
    'azure' : {
        'max_train_time' : 8*60*60,
        'validation_every_n_epochs' : 0.5,
        'text_truncate' : 400,
        'label_truncate' : 150,
        # Predict parameters
        'num_examples' : -1
    },
    'display' : {
        'num_examples' : 10
    }
}

model_dicts = {
    'pretrained_baseline' : {
    'model_name' : 'transformer_retrieval_pretrained_baseline',
    'baseline_model' : 'transformer/biencoder', # Baseline model architecture used
    'init_model' : '',  # The initial model to use for fine-tuning
    'fine_tune_dataset' : '', # Dataset used for fine tuning (train.txt and valid.txt)
    'train_model_file' : '', # Path where the trained model is saved
    'dynamic_batching' : 'full',
    'batchsize' : 10,
    'predict_dataset' : 'customer_care', # Dataset used for predicting (test.txt)
    'predict_model_file' : 'zoo:pretrained_transformers/bi_model_huge_reddit/model' # Path of the model which will be used for prediction
    },
    'finetuned_ed' : {
    'model_name' : 'transformer_retrieval_finetuned_ed',
    'baseline_model' : 'transformer/biencoder', # Baseline model architecture used
    'init_model' : 'zoo:pretrained_transformers/bi_model_huge_reddit/model', # The initial model to use for fine-tuning
    'fine_tune_dataset' : 'empathetic_dialogues', # Dataset used for fine tuning (train.txt and valid.txt)
    'train_model_file' : 'models/transformer_retrieval/finetuned_ed/model', # Path where the trained model is saved
    'dynamic_batching' : 'full',
    'batchsize' : 10,
    'predict_dataset' : 'customer_care', # Dataset used for predicting (test.txt)
    'predict_model_file' : 'models/transformer_retrieval/finetuned_ed/model' # Path of the model which will be used for prediction
    },
    'finetuned_cc' : {
    'model_name' : 'transformer_retrieval_finetuned_cc',
    'baseline_model' : 'transformer/biencoder', # Baseline model architecture used
    'init_model' : 'zoo:pretrained_transformers/bi_model_huge_reddit/model', # The initial model to use for fine-tuning
    'fine_tune_dataset' : 'customer_care', # Dataset used for fine tuning (train.txt and valid.txt)
    'train_model_file' : 'models/transformer_retrieval/finetuned_cc/model', # Path where the trained model is saved
    'dynamic_batching' : None,
    'batchsize' : 10,
    'predict_dataset' : 'customer_care', # Dataset used for predicting (test.txt)
    'predict_model_file' : 'models/transformer_retrieval/finetuned_cc/model' # Path of the model which will be used for prediction
    },
    'finetuned_ed_cc' : {
    'model_name' : 'transformer_retrieval_finetuned_ed_cc',
    'baseline_model' : 'transformer/biencoder', # Baseline model architecture used
    'init_model' : 'models/transformer_retrieval/finetuned_ed/model', # The initial model to use for fine-tuning
    'fine_tune_dataset' : 'customer_care', # Dataset used for fine tuning (train.txt and valid.txt)
    'train_model_file' : 'models/transformer_retrieval/finetuned_ed_cc/model', # Path where the trained model is saved
    'dynamic_batching' : None,
    'batchsize' : 10,
    'predict_dataset' : 'customer_care', # Dataset used for predicting (test.txt)
    'predict_model_file' : 'models/transformer_retrieval/finetuned_ed_cc/model' # Path of the model which will be used for prediction
    }
}


def display_main(dataset):
    DisplayData.main(task=dataset, num_examples=5, datatype='train')


# Using a pretrained transformer generator model and fine tuning it on a particular dataset
def train_main(model_dict, run_mode):
    params = run_modes[run_mode]
    TrainModel.main(
    # similar to before
    task=model_dict['fine_tune_dataset'],
    model=model_dict['baseline_model'],
    model_file=model_dict['train_model_file'],
    
    # initialize with a pretrained model
    init_model=model_dict['init_model'],
    
    # arguments we get from the pretrained model.
    # Unfortunately, these must be looked up separately for each model.
    n_heads=12, n_layers=12, n_positions=1024, text_truncate=params['text_truncate'],
    label_truncate=params['label_truncate'], ffn_size=3072, embedding_size=768,
    embeddings_scale = False,
    share_encoders = False,
    dict_endtoken = '__start__',
    activation='gelu', variant='xlm',
    share_word_embeddings = False,
    dict_lower=True, dict_tokenizer='bpe',
    learn_embeddings = True,
    n_segments =2,
    dict_file='zoo:pretrained_transformers/bi_model_huge_reddit/model.dict',
    learn_positional_embeddings=True,
    candidates = 'batch',
    eval_candidates = 'batch',
    attention_dropout =  0.1,
    dropout = 0.1,
    relu_dropout = 0,
    output_scaling = 0.06,

    # some training arguments, specific to this fine-tuning
    # use a small learning rate with ADAM optimizer
    lr=1e-4, optimizer='adamax',
    warmup_updates=100,
    save_every_n_secs = -1,
    # early stopping on hits@1
    validation_metric='hits@1',
    # train at most 10 minutes, and validate every 0.25 epochs
    max_train_time=params['max_train_time'], validation_every_n_epochs=params['validation_every_n_epochs'],
    
    # depend on your gpu. If you have a V100, this is good
    batchsize=model_dict['batchsize'], fp16=True,
    
    # helps us cram more examples into our gpu at a time
    dynamic_batching=model_dict['dynamic_batching'],
    )

def predict_main(model_dict, run_mode):
    params = run_modes[run_mode]
    if run_mode=='display':
        DisplayModel.main(
        task=model_dict['predict_dataset'],
        model_file=model_dict['predict_model_file'],
        num_examples=params['num_examples'],
        skip_generation=False
        )
    else:
        return EvalModel.main(
        task=model_dict['predict_dataset'],
        model_file=model_dict['predict_model_file'],
        metrics =  ['ppl','f1','accuracy'],
        num_examples=params['num_examples'],
        eval_candidates = 'batch',
        batchsize = model_dict['batchsize'],
        report_filename='results/'+model_dict['model_name']
        )

def main(models_to_train, models_to_predict, run_mode='local'):
    predict_results = {}
    for model in models_to_train:
        if model in model_dicts:
            train_main(model_dicts[model], run_mode)
        else:
            print('{} not defined in model_dicts()'.format(model))
    for model in models_to_predict:
        if model in model_dicts:
            predict_results[model] = predict_main(model_dicts[model], run_mode)
        else:
            print('{} not defined in model_dicts()'.format(model))

if __name__ == "__main__":
    #display_main('customer_care')
    if len(sys.argv) > 1:
        models_to_train, models_to_predict, run_mode = decode_args(sys.argv[1:], run_modes, model_dicts)

    print('\n\nmodels_to_train: {0}\nmodels_to_predict: {1}\nrun_mode: {2}'.format(models_to_train, models_to_predict, run_mode))
    main(models_to_train, models_to_predict, run_mode)

    print('\n\nmodels_to_train: {0}\nmodels_to_predict: {1}\nrun_mode: {2}'.format(models_to_train, models_to_predict, run_mode))
    display_results_retrieval(model_dicts)
