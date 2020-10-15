from parlai.core.teachers import DialogTeacher, MultiTaskTeacher,ParlAIDialogTeacher
from parlai.utils.io import PathManager
from .build import build
import copy
import os

def _path(opt, filtered):
    # build the data if it does not exist
    build(opt)

    # set up path to data (specific to each dataset)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'Customer_Care', dt + '.txt')


class DefaultTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)

        # get datafile
        opt['parlaidialogteacher_datafile'] = _path(opt, '')

        super().__init__(opt, shared)