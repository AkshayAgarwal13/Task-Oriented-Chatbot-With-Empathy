import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile
import copy
import json
import os


"""
File added by Akshay which downloads the Twitter Customer Care dataset and creates a task

"""
RESOURCES = [
    DownloadableFile(
        'https://raw.githubusercontent.com/AkshayAgarwal13/ParlAi_Test/main/train.txt',
        'train.txt',
        '61fc2cd964fb935ea1cd736c149755495075695312ecc60b21eb10419f6b8ad7',
        zipped=False,
    ),

    DownloadableFile(
        'https://raw.githubusercontent.com/AkshayAgarwal13/ParlAi_Test/main/valid.txt',
        'valid.txt',
        '61fc2cd964fb935ea1cd736c149755495075695312ecc60b21eb10419f6b8ad7',
        zipped=False,
    ),

    DownloadableFile(
        'https://raw.githubusercontent.com/AkshayAgarwal13/ParlAi_Test/main/test.txt',
        'test.txt',
        '61fc2cd964fb935ea1cd736c149755495075695312ecc60b21eb10419f6b8ad7',
        zipped=False,
    ),
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'Customer_Care')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES[:3]:
            downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)