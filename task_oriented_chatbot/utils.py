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

def display_results(model_dicts):
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

