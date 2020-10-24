from parlai.scripts.display_model import DisplayModel
from parlai.scripts.display_model import DisplayModel
DisplayModel.main(
    task='customer_Care',
    model_file='zoo:tutorial_transformer_generator/model',
    #model_file='zoo:bart/model',
    fp16 = False,
    num_examples=10,
    skip_generation=False,
    optimizer='adam',
)