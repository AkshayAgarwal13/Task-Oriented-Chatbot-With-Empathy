
# Task-Oriented-Chatbot-With-Empathy

- Akshay Agarwal, Shashank Maiya, Sonu Aggarwal


This repository is forked from [ParlAI](https://github.com/facebookresearch/ParlAI). In this project, we fine tune a language model based on emotions and show that it performs better on task-oriented chatbots compared to the one without any such fine tuning.

## Getting Started

### Installing the packages
Run the following commands to clone the repository and install all the required dependencies:

```bash
git clone https://github.com/shashankvmaiya/Task-Oriented-Chatbot-With-Empathy.git
cd Task-Oriented-Chatbot-With-Empathy; python setup.py develop
```

All needed data will be downloaded to `data`, and any non-data files if requested will be downloaded to `downloads`. If you need to clear out the space used by these files, you can safely delete these directories and any files needed will be downloaded again.

### Datasets

- [Empathetic Dialogues Dataset](https://github.com/facebookresearch/EmpatheticDialogues) is used to fine tune our baseline model, so that it generates a more empahtetic response
- [Twitter Customer Support Dataset](https://www.kaggle.com/thoughtvector/customer-support-on-twitter?select=sample.csv) is used as our core dataset to evaluate all our models. This dataset is a large, modern corpus of tweets and replies to aid innovation in natural language understanding and conversational models, and for the study of modern customer support practices and impact. This dataset has been added into the ParlAI framework. Run the below command to create a task
```bash
parlai display_data -t customer_care
```
It will download a sample train and validation file from my github folder and download the data at `data/customer_care`


### Code Organization
Our code is found under the folder [task_oriented_chatbot](https://github.com/shashankvmaiya/Task-Oriented-Chatbot-With-Empathy/tree/master/task_oriented_chatbot). The code for this is primarily obtained from [Rashkin et. al github](https://github.com/facebookresearch/EmpatheticDialogues). 


