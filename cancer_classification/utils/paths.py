import os
import inspect
from inspect import FrameInfo


s3_bucket_name = 'rishabhmalviya---cancer-classification'

_curr_dir = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR = os.path.join(_curr_dir, '../../data/NCT-CRC-HE-100K/')

EXPERIMENT_LOGS_DIR = os.path.join(_curr_dir, '../../experiments/logs')


def get_curr_dir():
    caller_frame_info: FrameInfo = inspect.stack()[1]
    caller_abs_path = caller_frame_info.filename
    
    return os.path.dirname(caller_abs_path).split('/')[-1]


def get_curr_filename():
    caller_frame_info: FrameInfo = inspect.stack()[1]
    caller_abs_path = caller_frame_info.filename
    
    return os.path.basename(caller_abs_path)

