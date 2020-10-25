from parlai.scripts.display_model import DisplayModel
from parlai.scripts.display_model import DisplayModel
DisplayModel.main(
    task='customer_Care',
    model_file='/home/xcs224u/project/Task-Oriented-Chatbot-With-Empathy/data/test_models/pretrained_transformer_ed_full/model',
   # model_file='zoo:tutorial_transformer_generator/model',
    fp16 = False,
    num_examples=10,
    skip_generation=False,
    optimizer='adam',
)