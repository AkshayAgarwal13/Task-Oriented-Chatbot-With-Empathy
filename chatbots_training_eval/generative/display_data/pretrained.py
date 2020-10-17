from parlai.scripts.display_model import DisplayModel
from parlai.scripts.display_model import DisplayModel
DisplayModel.main(
    task='customer_Care',
    model_file='zoo:tutorial_transformer_generator/model',
    fp16 = False,
    num_examples=2,
    skip_generation=False,
    optimizer='adam',
)