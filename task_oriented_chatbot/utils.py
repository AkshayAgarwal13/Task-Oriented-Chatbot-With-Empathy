#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
import re
import json

def decode_args(args, run_modes, model_dicts):
    run_mode, train_flag, predict_flag = 'local', True, True
    models_to_train, models_to_predict = [], []
    for arg in args:
        if arg.lower() == 'train':
            predict_flag = False
        elif arg.lower() == 'predict':
            train_flag = False
        elif arg.lower() in run_modes:
            run_mode = arg.lower()
        elif arg.lower() in model_dicts:
            models_to_train.append(arg.lower())
            models_to_predict.append(arg.lower())
    if train_flag:
        models_to_predict = []
    if predict_flag:
        models_to_train = []
    return models_to_train, models_to_predict, run_mode

def display_results_generative(model_dicts):
    print('\n\n------------------------------------------------------------------------------------------')
    print('{0:>45}\t{1:^12}{2:^12}'.format('Models', 'ppl_valid', 'ppl_test'))
    for model in model_dicts:
        m = model_dicts[model]
        model_name = m['model_name']
        ppl_valid, ppl_test = '-', '-'
        if model == 'pretrained_baseline':
            file = 'results/'+m['model_name']
            with open(file, 'r') as json_file:
                data = json.load(json_file)
                datatype = data['opt']['datatype']
                if datatype=='valid':
                    ppl_valid = '{:.2f}'.format(data['report']['ppl'])
                else:
                    ppl_test = '{:.2f}'.format(data['report']['ppl'])

        else:
            valid_file = m['train_model_file']+'.valid'
            with open(valid_file) as f:
                for line in f:
                    if re.search('^\s*\d+', line):
                        metrics = line.split()
                        ppl_valid = metrics[8]

            test_file = m['train_model_file']+'.test'
            with open(test_file) as f:
                for line in f:
                    if re.search('^\s*\d+', line):
                        metrics = line.split()
                        ppl_test = metrics[8]
        print('{0:>45}\t{1:^12}{2:^12}'.format(model_name, ppl_valid, ppl_test))

    print('------------------------------------------------------------------------------------------')

def display_results_retrieval(model_dicts):
    print('\n\n------------------------------------------------------------------------------------------')
    print('{0:>45}\t{1:^15}{2:^15}{3:^15}{4:^15}{5:^15}{6:^15}{7:^15}{8:^15}'.\
        format('Models', 'hits@1_valid', 'hits@5_valid', 'hits@10_valid', 'hits@100_valid', \
        'hits@1_test', 'hits@5_test', 'hits@10_test', 'hits@100_test'))
    for model in model_dicts:
        m = model_dicts[model]
        model_name = m['model_name']
        hits1_valid, hits5_valid, hits10_valid, hits100_valid = '-', '-', '-', '-'
        hits1_test, hits5_test, hits10_test, hits100_test = '-', '-', '-', '-'
        if model == 'pretrained_baseline':
            file = 'results/'+m['model_name']
            with open(file, 'r') as json_file:
                data = json.load(json_file)
                datatype = data['opt']['datatype']
                if datatype=='valid':
                    hits1_valid = '{:.2f}'.format(data['report']['hits@1'])
                    hits5_valid = '{:.2f}'.format(data['report']['hits@5'])
                    hits10_valid = '{:.2f}'.format(data['report']['hits@10'])
                    hits100_valid = '{:.2f}'.format(data['report']['hits@100'])
                else:
                    hits1_test = '{:.2f}'.format(data['report']['hits@1'])
                    hits5_test = '{:.2f}'.format(data['report']['hits@5'])
                    hits10_test = '{:.2f}'.format(data['report']['hits@10'])
                    hits100_test = '{:.2f}'.format(data['report']['hits@100'])

        else:
            valid_file = m['train_model_file']+'.valid'
            with open(valid_file) as f:
                for line in f:
                    if re.search('^\s*\.?\d+', line):
                        metrics = line.split()
                        if len(metrics)>10:
                            hits1_valid, hits5_valid, hits10_valid, hits100_valid = metrics[7], metrics[8], metrics[10], metrics[9]

            test_file = m['train_model_file']+'.test'
            with open(test_file) as f:
                for line in f:
                    if re.search('^\s*\.?\d+', line):
                        metrics = line.split()
                        if len(metrics)>10:
                            hits1_test, hits5_test, hits10_test, hits100_test = metrics[7], metrics[8], metrics[10], metrics[9]
        
        print('{0:>45}\t{1:^15}{2:^15}{3:^15}{4:^15}{5:^15}{6:^15}{7:^15}{8:^15}'.\
            format(model_name, hits1_valid, hits5_valid, hits10_valid, hits100_valid, \
            hits1_test, hits5_test, hits10_test, hits100_test))

    print('------------------------------------------------------------------------------------------')


