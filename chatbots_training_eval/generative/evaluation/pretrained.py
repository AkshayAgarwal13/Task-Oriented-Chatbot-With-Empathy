

from parlai.scripts.eval_model import EvalModel 
EvalModel.main(
    task='customer_Care',
    #model_file='from_pretrained/model',
    model_file='zoo:tutorial_transformer_generator/model',
   # fp16 = False,
    num_examples=5,
    skip_generation=False,
   # optimizer='adam',

)