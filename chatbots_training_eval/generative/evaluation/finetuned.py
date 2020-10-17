from parlai.scripts.eval_model import EvalModel 
EvalModel.main(
    task='customer_Care',
    model_file='../training/pretrained/model',
    metrics =  ['ppl','f1','accuracy','hits@1'],
   # model_file='zoo:bert/model',
   # fp16 = False,
    num_examples=5,    
   # optimizer='adam',

)