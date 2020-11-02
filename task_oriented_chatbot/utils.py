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

def get_metrics_TrainModel(file, metric_names):
    metrics = {}
    for n in metric_names:
        metrics[n] = '-'
    with open(file) as f:
        for line in f:
            if (line.strip() not in ['valid:', 'test:']) and (re.search('^\s*[a-z]+', line)):
                names = line.split()
                values = next(f).split()
                for n in metric_names:
                    if n in names:
                        idx = names.index(n)
                        metrics[n] = values[idx]
    return metrics

def get_metrics_EvalModel(file, metric_names):
    metrics = {}
    for n in metric_names:
        metrics[n] = '-'
    with open(file, 'r') as json_file:
        data = json.load(json_file)
        for metric in metric_names:
            if metric in data['report']:
                metrics[metric] = '{:.2f}'.format(data['report'][metric])
    return metrics

def display_results(model_dicts, metric_names):
    print('\n\n------------------------------------------------------------------------------------------')
    strFormat = '{:>50}' + 2*len(metric_names)*'{:^15}'
    title = ['Models'] + [m+'_valid' for m in metric_names] + [m+'_test' for m in metric_names]
    print(strFormat.format(*title))
    for model in model_dicts:
        m = model_dicts[model]
        model_name = m['model_name']
        metrics_valid = {metric:'-' for metric in metric_names}
        metrics_test = {metric:'-' for metric in metric_names}
        if model in ['pretrained_baseline', 'finetuned_ed']:
            valid_file = 'results/'+m['model_name']+'_valid'
            metrics_valid = get_metrics_EvalModel(valid_file, metric_names)

            test_file = 'results/'+m['model_name']+'_test'
            metrics_test = get_metrics_EvalModel(test_file, metric_names)
        else:
            valid_file = m['train_model_file']+'.valid'
            metrics_valid = get_metrics_TrainModel(valid_file, metric_names)

            test_file = m['train_model_file']+'.test'
            metrics_test = get_metrics_TrainModel(test_file, metric_names)

        metric_values = [model_name] + [metrics_valid[metric] for metric in metric_names] + [metrics_test[metric] for metric in metric_names]
        print(strFormat.format(*metric_values))

    print('------------------------------------------------------------------------------------------')



