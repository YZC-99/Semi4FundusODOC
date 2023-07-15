import os

from yacs.config import CfgNode as CN
# training -cfg
_C = CN()
_C.info = CN()
_C.info.setting='od_val_refuge/1_7/100gamma'
_C.info.description=''

_C.aug = CN()
_C.aug.strong = CN()
_C.aug.strong.Not = False
_C.aug.strong.default = True
_C.aug.strong.ColorJitter = False
_C.aug.strong.RandomGrayscale = False
_C.aug.strong.blur = False
_C.aug.strong.cutout = False
# _C.aug.weak.hflip =

# _C = CN()
_C.MODEL = CN()
_C.MODEL.sup = False
_C.MODEL.NUM_CLASSES = 2
_C.MODEL.batch_size = 16

_C.MODEL.epochs = 150
_C.MODEL.crop_size = 150
_C.MODEL.lr = 0.004

_C.MODEL.dataset = 'od_val_refuge'
_C.MODEL.task = 'od'
_C.MODEL.data_root = './data/fundus_datasets/od_oc/WACV/REFUGE_cross_new/'
_C.MODEL.backbone = 'resnet50'
_C.MODEL.model = 'deeplabv3plus'
_C.MODEL.labeled_id_path = ''
_C.MODEL.labeled_id_path_2 = ''
_C.MODEL.unlabeled_id_path = ''
_C.MODEL.pseudo_mask_path = ''
_C.MODEL.logs_path = ''
_C.MODEL.save_path = ''
_C.MODEL.uda = False
_C.MODEL.sup_uda = False
_C.MODEL.stage1 = False #训练教师网络
_C.MODEL.stage1_ckpt_path = None
_C.MODEL.stage2 = False # 计算prototype或者进行全监督域适应训练
_C.MODEL.stage2_prototype = False
_C.MODEL.stage2_prototype_useTeacher = True
_C.MODEL.stage3 = False #打伪标签
_C.MODEL.stage4 = False #进行半监督训练
_C.MODEL.stage4_uda = False #进行半监督训练
_C.MODEL.stage5 = False
_C.MODEL.NAME = "deeplabv3plus_resnet50"
_C.MODEL.DEVICE = "cuda"
_C.MODEL.WEIGHTS = ""
_C.MODEL.FREEZE_BN = False
_C.MODEL.EVAL_BN = False
_C.MODEL.MOMENTUM_ITER = 100
_C.MODEL.THRESHOLD_PERCENT = 0.5

_C.MODEL.CONTRAST = CN()
_C.MODEL.CONTRAST.PROJ_DIM = 256
_C.MODEL.CONTRAST.MEMORY_SIZE = 1000
_C.MODEL.CONTRAST.PIXEL_UPDATE_FREQ = 10
_C.MODEL.CONTRAST.TAU = 50.0
_C.MODEL.CONTRAST.USE_MOMENTUM = False
_C.MODEL.CONTRAST.MOMENTUM = 0.9

_C.dataset = CN()
_C.dataset.target =''
_C.dataset.params = CN()
_C.dataset.params.batch_size = 8
_C.dataset.params.num_workers = 8
#train
_C.dataset.params.train = CN()
_C.dataset.params.train.target =''
_C.dataset.params.train.params = CN()
_C.dataset.params.train.params.task = ''
_C.dataset.params.train.params.name = ''
_C.dataset.params.train.params.root = ''
_C.dataset.params.train.params.mode = ''
_C.dataset.params.train.params.size = 512
_C.dataset.params.train.params.labeled_id_path = ''
_C.dataset.params.train.params.unlabeled_id_path = ''
_C.dataset.params.train.params.pseudo_mask_path = ''
_C.dataset.params.train.params.cfg = CN()
#
_C.dataset.params.train2 = CN()
_C.dataset.params.train2.target =''
_C.dataset.params.train2.params = CN()
_C.dataset.params.train2.params.task = ''
_C.dataset.params.train2.params.name = ''
_C.dataset.params.train2.params.root = ''
_C.dataset.params.train2.params.mode = ''
_C.dataset.params.train2.params.size = 512
_C.dataset.params.train2.params.labeled_id_path = ''
_C.dataset.params.train2.params.unlabeled_id_path = ''
_C.dataset.params.train2.params.pseudo_mask_path = ''
_C.dataset.params.train2.params.cfg = CN()
#val
_C.dataset.params.validation = CN()
_C.dataset.params.validation.target =''
_C.dataset.params.validation.params = CN()
_C.dataset.params.validation.params.task = ''
_C.dataset.params.validation.params.name = ''
_C.dataset.params.validation.params.root = ''
_C.dataset.params.validation.params.mode = ''
_C.dataset.params.validation.params.size = 512
_C.dataset.params.validation.params.labeled_id_path = ''
_C.dataset.params.validation.params.unlabeled_id_path = ''
_C.dataset.params.validation.params.pseudo_mask_path = ''
_C.dataset.params.validation.params.cfg = CN()
# _C = CN()
_C.OUTPUT_DIR = 'experiments_pca/prototype'
# update
_C.INPUT = CN()
_C.INPUT.SOURCE_INPUT_SIZE_TRAIN = (512, 512)
_C.INPUT.TARGET_INPUT_SIZE_TRAIN = (512, 512)
_C.INPUT.INPUT_SIZE_TEST = (512, 512)
_C.INPUT.INPUT_SCALES_TRAIN = (1.0, 1.0)
_C.INPUT.IGNORE_LABEL = 255
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Convert image to BGR format (for Caffe2 models), in range 0-255
_C.INPUT.TO_BGR255 = False

# GaussianBlur
_C.INPUT.GAUSSIANBLUR = False

# Image ColorJitter
_C.INPUT.BRIGHTNESS = 0.0
_C.INPUT.CONTRAST = 0.0
_C.INPUT.SATURATION = 0.0
_C.INPUT.HUE = 0.0

# RandomApply Transforms
_C.INPUT.RANDOMAPPLY = 0.0

# RandomGrayscale
_C.INPUT.GRAYSCALE = 0.0

# Flips
_C.INPUT.HORIZONTAL_FLIP_PROB_TRAIN = 0.0

_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.SOURCE_TRAIN = ""
_C.DATASETS.TARGET_TRAIN = ""
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ""

_C.SOLVER = CN()
_C.SOLVER.NUM_WORKERS = 4
_C.SOLVER.MAX_ITER = 16000
_C.SOLVER.STOP_ITER = 10000
_C.SOLVER.CHECKPOINT_PERIOD = 1000

_C.SOLVER.LR_METHOD = 'poly'
_C.SOLVER.BASE_LR = 0.02
_C.SOLVER.BASE_LR_D = 0.008
_C.SOLVER.LR_POWER = 0.9
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.EMA_DECAY = 0.99
_C.SOLVER.KD_WEIGHT = 1.0
_C.SOLVER.WEIGHT_DECAY = 0.0005

# Number of images per batch, if we have 4 GPUs and BATCH_SIZE = 8, each GPU will
# see 2 images per batch, one for source and one for target
_C.SOLVER.BATCH_SIZE = 8
_C.SOLVER.BATCH_SIZE_VAL = 1


# Hyper-parameter
_C.SOLVER.MULTI_LEVEL = False
# lovasz_softmax loss
_C.SOLVER.LAMBDA_LOV = 0.0
# constant threshold for target mask
_C.SOLVER.DELTA = 0.9
# weight of feature level contrastive loss
_C.SOLVER.LAMBDA_FEAT = 1.0
# weight of output level contrastive loss
_C.SOLVER.LAMBDA_OUT = 1.0


# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 1

_C.resume = ""
_C.PREPARE_DIR = ""
#存放prototype的
_C.prototype_path = ""
