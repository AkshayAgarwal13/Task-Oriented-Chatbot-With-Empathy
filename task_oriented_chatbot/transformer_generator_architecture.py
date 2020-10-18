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

pretrained_baseline = {
'baseline_model' : 'transformer/generator', # Baseline model architecture used
'init_model' : '',  # The initial model to use for fine-tuning
'fine_tune_dataset' : '', # Dataset used for fine tuning (train.txt and valid.txt)
'train_model_file' : '', # Path where the trained model is saved
'predict_dataset' : 'customer_care', # Dataset used for predicting (test.txt)
'predict_model_file' : 'zoo:tutorial_transformer_generator/model' # Path of the model which will be used for prediction
}

# Reusing the fine tuned models for ED. They seem to be in a different location. Download them into finetuned_ed/model and call predict_main
# wget https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/models/normal_transformer_pretrained.mdl  # Normal Transformer, pretrained
# wget https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/models/normal_transformer_finetuned.mdl  # Normal Transformer, fine-tuned
# wget https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/models/bert_pretrained.mdl  # BERT, pretrained
# wget https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/models/bert_finetuned.mdl  # BERT, fine-tuned
# wget https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/models/bert_finetuned_emoprepend1.mdl  # BERT, fine-tuned (EmoPrepend-1)
# wget https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/models/fasttext_empathetic_dialogues.mdl  # fastText classifier used for EmoPrepend-1
finetuned_ed = {
'baseline_model' : 'transformer/generator',
'init_model' : 'zoo:tutorial_transformer_generator/model',
'fine_tune_dataset' : 'empathetic_dialogues',
'train_model_file' : 'finetuned_ed/model',
'predict_dataset' : 'customer_care',
'predict_model_file' : 'zoo:/model'
}
finetuned_cc = {
'baseline_model' : 'transformer/generator',
'init_model' : 'zoo:tutorial_transformer_generator/model',
'fine_tune_dataset' : 'customer_care',
'train_model_file' : 'finetuned_cc/model',
'predict_dataset' : 'customer_care',
'predict_model_file' : 'finetuned_cc/model'
}
finetuned_ed_cc = {
'baseline_model' : 'transformer/generator',
'init_model' : 'finetuned_ed/model',
'fine_tune_dataset' : 'customer_care',
'train_model_file' : 'finetuned_ed_cc/model',
'predict_dataset' : 'customer_care',
'predict_model_file' : 'finetuned_ed_cc/model'
}

models_to_train = [finetuned_cc, finetuned_ed_cc]
models_to_test = [pretrained_baseline, finetuned_ed, finetuned_cc, finetuned_ed_cc]

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
    if model_dict['init_model']:
        init_model=model_dict['init_model'],
    
    # arguments we get from the pretrained model.
    # Unfortunately, these must be looked up separately for each model.
    n_heads=16, n_layers=8, n_positions=512, text_truncate=512,
    label_truncate=128, ffn_size=2048, embedding_size=512,
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
    max_train_time=10*60, validation_every_n_epochs=0.01,
    
    # depend on your gpu. If you have a V100, this is good
    batchsize=12, fp16=True, fp16_impl='mem_efficient',
    
    # speeds up validation
    skip_generation=True,
    
    # helps us cram more examples into our gpu at a time
    dynamic_batching='full',
    )

def predict_main(model_dict):
    DisplayModel.main(
    task=model_dict['predict_dataset'],
    model_file=model_dict['predict_model_file'],
    num_examples=2,
    skip_generation=False
    )


if __name__ == "__main__":
    #display_main('customer_care')
    for model in models_to_train:
        train_main(model)
    for model in models_to_predict:
        predict_main(model)












