from parlai.scripts.eval_model import EvalModel 
EvalModel.main(
    task='customer_Care',
    model_file='../training/pretrained_transformer_empathy_2/model',
    metrics =  ['ppl','f1','accuracy','hits@1'],
  
    num_examples=100,    
   # optimizer='adam',

)