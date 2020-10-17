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


def display_data(dataset, datatype='train'):
    DisplayData.main(task='customer_care', num_examples=5, datatype='test')


# Training a model from scratch on customer_care dataset
def train_from_scratch():
    TrainModel.main(
    # we MUST provide a filename
    model_file='from_scratch_model/model',
    # train on customer_care
    task='customer_care',
    # limit training time to 2 minutes, and a batchsize of 16
    max_train_time=2 * 60,
    batchsize=16,
    # we specify the model type as seq2seq
    model='seq2seq',
    # some hyperparamter choices. We'll use attention. We could use pretrained
    # embeddings too, with embedding_type='fasttext', but they take a long
    # time to download.
    attention='dot',
    # tie the word embeddings of the encoder/decoder/softmax.
    lookuptable='all',
    # truncate text and labels at 64 tokens, for memory and time savings
    truncate=64,
    )

# Using a pretrained transformer generator model and fine tuning it on a particular dataset
def train_fine_tune(fine_tune_dataset):
    TrainModel.main(
    # similar to before
    task=fine_tune_dataset,
    model='transformer/generator',
    model_file='fine_tuned_{}/model'.format(fine_tune_dataset),
    
    # initialize with a pretrained model
    init_model='zoo:tutorial_transformer_generator/model',
    
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

def predict_main(model, dataset):
    DisplayModel.main(
    task=dataset,
    model_file=model,
    num_examples=2,
    skip_generation=False
    )


if __name__ == "__main__":
    #display_data('customer_care', 'train')
    #train_from_scratch()
    #train_fine_tune('empathetic_dialogues')
    #train_fine_tune('customer_care')
    predict_main('fine_tuned_empathetic_dialogues/model', 'customer_care')












