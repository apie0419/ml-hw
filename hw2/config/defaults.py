from yacs.config import CfgNode as CN
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

_C = CN()
_C.PATH = CN()
_C.MODEL = CN()
_C.DEVICE = CN()
_C.DATA = CN()

_C.DEVICE.GPU = 1 # <gpu_id>
_C.DEVICE.CUDA = True #<cuda_or_cpu>

_C.PATH.TRAIN_SET = "" # <path_to_trainset>
_C.PATH.TEST_SET = "" # <path_to_testset>

_C.MODEL.OUTPUT_PATH = "" # <weight_output_path>
_C.MODEL.LR = 0.01 # <learning_rate>
_C.MODEL.EPOCH = 0 # <train_epochs>

# -----------------------------------------------
# 標準化參數，也可以用其他的，這是推薦值

_C.DATA.PIXEL_MEAN = [0.485, 0.456, 0.406] 
_C.DATA.PIXEL_STD = [0.229, 0.224, 0.225]

# -----------------------------------------------

_C.DATA.RESIZE = [224, 224] # RESIZE 圖片大小
_C.DATA.NUM_WORKERS = 8 #用幾個 processors
_C.DATA.TRAIN_BATCH_SIZE = 32 # <train_batch_size>
_C.DATA.TEST_BATCH_SIZE = 16 # <test_batch_size>


_C.merge_from_file(os.path.join(BASE_PATH, "config.yml"))