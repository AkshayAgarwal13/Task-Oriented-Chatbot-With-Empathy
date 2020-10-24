from parlai.scripts.display_model import DisplayModel
from parlai.scripts.display_model import DisplayModel

from parlai.scripts.eval_model import EvalModel 
EvalModel.main(
    task='customer_Care',
    model_file='zoo:blended_skill_talk/ed_single_task/model',
   # model_file = 'zoo:wizard_of_wikipedia/knowledge_retriever/model',
    #model_file='transformer_pretrained_cc/model',
    metrics =  ['ppl','f1','accuracy','hits@1'],
   # model_file='zoo:bert/model',
   # fp16 = False,
    num_examples=500,    
   # optimizer='adam',
    eval_candidates = 'batch',
    batchsize = 10
)