"""
This file fine-tunes the customer care model on transformer architecture.
Transformer architecture with 87M parameters is used as pretrained model

To run the model create a directory "pretrained" under ./chatbots_training_eval/training/

Example:
mkdir pretrained



"""


from parlai.scripts.train_model import TrainModel

TrainModel.main(
    # similar to before
    task='empathetic_dialogues',
    #task = 'empatheticdialogues', 
    model='transformer/generator',
    model_file='pretrained_transformer_cc_empathy/model',
    
    # initialize with a pretrained model
    init_model='pretrained_transformer_cc/model',
    #init_model = '..\..\data\models\empathy',
    # arguments we get from the pretrained model.
    # Unfortunately, these must be looked up separately for each model.
    n_heads=16, n_layers=8, n_positions=512, text_truncate=100,
    label_truncate=100, ffn_size=2048, embedding_size=512,
    activation='gelu', variant='xlm',
    dict_lower=True, dict_tokenizer='bpe',
    dict_file='pretrained_transformer_cc/model.dict',
    #dict_file = 'zoo:bert/model.vocab',
    learn_positional_embeddings=True,
    
    # some training arguments, specific to this fine-tuning
    # use a small learning rate with ADAM optimizer
    lr=1e-4, optimizer='adam',
    warmup_updates=100,
    save_every_n_secs = -1,
    # early stopping on perplexity
    validation_metric='ppl',
    # train at most 10 minutes, and validate every 0.25 epochs
    max_train_time=600, validation_every_n_epochs=0.4,
    
    # depend on your gpu. If you have a V100, this is good
    batchsize=6, fp16=True, fp16_impl='mem_efficient',
    
    # speeds up validation
    skip_generation=True,
    
    # helps us cram more examples into our gpu at a time
    dynamic_batching='full',
)