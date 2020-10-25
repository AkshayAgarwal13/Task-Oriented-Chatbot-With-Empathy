from parlai.scripts.eval_model import EvalModel 
EvalModel.main(
    task='customer_Care',
    model_file='/home/xcs224u/project/Task-Oriented-Chatbot-With-Empathy/data/test_models/pretrained_transformer__ed_cc/model',
    metrics =  ['ppl','f1','accuracy','hits@1'],
   # model_file='zoo:bert/model',
   # fp16 = False,
    num_examples=200,    
   # optimizer='adam',

)