from fvcore.common.config import CfgNode


_C = CfgNode()

# ---------------------------------------------------------------------- #
# Architecture
# ---------------------------------------------------------------------- #
_C.MODEL = CfgNode()

_C.MODEL.META_ARCHITECTURE = CfgNode()

# Choose one of the following
# (1) BasicClassificationModel
# (2) BatchEnsembleClassificationModel
# (3) MIMOClassificationModel
#
_C.MODEL.META_ARCHITECTURE.NAME = "BasicClassificationModel"

# Path to a checkpoint file to be loaded to the model
# it is useful for loading pre-trained weights or evaluation
_C.MODEL.WEIGHTS = ""

# ---------------------------------------------------------------------- #
# BatchEnsemble
# ---------------------------------------------------------------------- #
_C.MODEL.BATCH_ENSEMBLE = CfgNode()
_C.MODEL.BATCH_ENSEMBLE.ENABLED = False

# The size of ensembles
_C.MODEL.BATCH_ENSEMBLE.ENSEMBLE_SIZE = 4

# Apply bias terms independently
_C.MODEL.BATCH_ENSEMBLE.USE_ENSEMBLE_BIAS = True

# Initialization of rank-one factors
_C.MODEL.BATCH_ENSEMBLE.ALPHA_INITIALIZER = CfgNode()
_C.MODEL.BATCH_ENSEMBLE.ALPHA_INITIALIZER.NAME = "normal"
_C.MODEL.BATCH_ENSEMBLE.ALPHA_INITIALIZER.VALUES = [1.0, 0.5,]
_C.MODEL.BATCH_ENSEMBLE.GAMMA_INITIALIZER = CfgNode()
_C.MODEL.BATCH_ENSEMBLE.GAMMA_INITIALIZER.NAME = "normal"
_C.MODEL.BATCH_ENSEMBLE.GAMMA_INITIALIZER.VALUES = [1.0, 0.5,]

# ---------------------------------------------------------------------- #
# MIMO
# ---------------------------------------------------------------------- #
_C.MODEL.MIMO = CfgNode()
_C.MODEL.MIMO.ENABLED = False

# The size of ensembles
_C.MODEL.MIMO.ENSEMBLE_SIZE = 3

# ---------------------------------------------------------------------- #
# Dropout
# ---------------------------------------------------------------------- #
_C.MODEL.DROPOUT = CfgNode()
_C.MODEL.DROPOUT.ENABLED = False
_C.MODEL.DROPOUT.DROP_PROBABILITY = 0.5

# ---------------------------------------------------------------------- #
# SpatialDropout
# ---------------------------------------------------------------------- #
_C.MODEL.SPATIAL_DROPOUT = CfgNode()
_C.MODEL.SPATIAL_DROPOUT.ENABLED = False
_C.MODEL.SPATIAL_DROPOUT.DROP_PROBABILITY = 0.5

# ---------------------------------------------------------------------- #
# DropBlock
# ---------------------------------------------------------------------- #
_C.MODEL.DROP_BLOCK = CfgNode()
_C.MODEL.DROP_BLOCK.ENABLED = False
_C.MODEL.DROP_BLOCK.DROP_PROBABILITY = 0.5
_C.MODEL.DROP_BLOCK.BLOCK_SIZE = 3
_C.MODEL.DROP_BLOCK.USE_SHARED_MASKS = False

# ---------------------------------------------------------------------- #
# Image Normalization
# ---------------------------------------------------------------------- #
# Values to be used for image normalization (RGB order)
# some pre-computed values are as follows:
#
# CIFAR10 (50k)
# >> [0.4914, 0.4822, 0.4465,]
# >> [0.2470, 0.2435, 0.2616,]
#
# CIFAR10 (45k)
# >> [0.4915, 0.4821, 0.4464,]
# >> [0.2472, 0.2437, 0.2617,]
#
# CIFAR100 (50k)
# >> [0.5071, 0.4865, 0.4409,]
# >> [0.2673, 0.2564, 0.2762,]
#
# CIFAR100 (45k)
# >> [0.5072, 0.4866, 0.4410,]
# >> [0.2673, 0.2564, 0.2760,]
#
# TinyImageNet200 (100k)
# >> [0.4802, 0.4481, 0.3975,]
# >> [0.2770, 0.2691, 0.2821,]
#
# TinyImageNet200 (90k)
# >> [0.4802, 0.4481, 0.3976,]
# >> [0.2770, 0.2691, 0.2822,]
#
# ImageNet1k
# >> [0.485, 0.456, 0.406,]
# >> [0.229, 0.224, 0.225,]
#
_C.MODEL.PIXEL_MEAN = [0.0, 0.0, 0.0,]
_C.MODEL.PIXEL_STD = [1.0, 1.0, 1.0,]

# ---------------------------------------------------------------------- #
# Datasets
# ---------------------------------------------------------------------- #
_C.DATASETS = CfgNode()

# The name of the dataset
# choose one of the following:
# [ "MNIST", "FashionMNIST", 
#   "CIFAR10", "CIFAR100",
#   "TinyImageNet200", "ImageNet1k" ]
_C.DATASETS.NAME = "CIFAR10"

# Set seed to shuffle datasets
_C.DATASETS.SEED = 42

# When the name is in ["MNIST", "FashionMNIST"]
_C.DATASETS.MNIST = CfgNode()
_C.DATASETS.MNIST.SHUFFLE_INDICES = False
_C.DATASETS.MNIST.TRAIN_INDICES = [0, 55000,]
_C.DATASETS.MNIST.VALID_INDICES = [55000, 60000,]
_C.DATASETS.MNIST.DATA_AUGMENTATION = "STANDARD_TRAIN_TRANSFORM"

# When the name is in ["CIFAR10", "CIFAR100"]
_C.DATASETS.CIFAR = CfgNode()
_C.DATASETS.CIFAR.SHUFFLE_INDICES = False
_C.DATASETS.CIFAR.TRAIN_INDICES = [0, 45000,]
_C.DATASETS.CIFAR.VALID_INDICES = [45000, 50000,]
_C.DATASETS.CIFAR.DATA_AUGMENTATION = "STANDARD_TRAIN_TRANSFORM"

# When the name is in ["TinyImageNet200"]
_C.DATASETS.TINY = CfgNode()
_C.DATASETS.TINY.SHUFFLE_INDICES = False
_C.DATASETS.TINY.TRAIN_INDICES = [0, 90000,]
_C.DATASETS.TINY.VALID_INDICES = [90000, 100000,]
_C.DATASETS.TINY.DATA_AUGMENTATION = "STANDARD_TRAIN_TRANSFORM"

# When the name is in ["ImageNet1k"]
_C.DATASETS.IMAGENET = CfgNode()
_C.DATASETS.IMAGENET.SHUFFLE_INDICES = True
_C.DATASETS.IMAGENET.TRAIN_INDICES = [0, 1231167,]
_C.DATASETS.IMAGENET.VALID_INDICES = [1231167, 1281167,]
_C.DATASETS.IMAGENET.DATA_AUGMENTATION = "STANDARD_TRAIN_TRANSFORM"

# ---------------------------------------------------------------------- #
# Dataloader
# ---------------------------------------------------------------------- #
_C.DATALOADER = CfgNode()

# The number of data loading threads
_C.DATALOADER.NUM_WORKERS = 0
_C.DATALOADER.PIN_MEMORY = False

# ---------------------------------------------------------------------- #
# Backbone
# ---------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CfgNode()
_C.MODEL.BACKBONE.NAME = "build_resnet_backbone"

# ResNet options
_C.MODEL.BACKBONE.RESNET = CfgNode()
_C.MODEL.BACKBONE.RESNET.CHANNELS = 3
_C.MODEL.BACKBONE.RESNET.IN_PLANES = 64
_C.MODEL.BACKBONE.RESNET.FIRST_BLOCK = CfgNode()
_C.MODEL.BACKBONE.RESNET.FIRST_BLOCK.USE_NORM_LAYER = True
_C.MODEL.BACKBONE.RESNET.FIRST_BLOCK.USE_ACTIVATION = True
_C.MODEL.BACKBONE.RESNET.FIRST_BLOCK.USE_POOL_LAYER = True
_C.MODEL.BACKBONE.RESNET.FIRST_BLOCK.CONV_KSP = [7, 2, 3,]
_C.MODEL.BACKBONE.RESNET.FIRST_BLOCK.POOL_KSP = [3, 2, 1,]
_C.MODEL.BACKBONE.RESNET.BLOCK = "Bottleneck"
_C.MODEL.BACKBONE.RESNET.SHORTCUT = "ProjectionShortcut"
_C.MODEL.BACKBONE.RESNET.NUM_BLOCKS = [3, 4, 6, 3,]
_C.MODEL.BACKBONE.RESNET.WIDEN_FACTOR = 1
_C.MODEL.BACKBONE.RESNET.CONV_LAYERS = "Conv2d"
_C.MODEL.BACKBONE.RESNET.CONV_LAYERS_BIAS = False
_C.MODEL.BACKBONE.RESNET.CONV_LAYERS_SAME_PADDING = False
_C.MODEL.BACKBONE.RESNET.NORM_LAYERS = "BatchNorm2d"
_C.MODEL.BACKBONE.RESNET.ACTIVATIONS = "ReLU"

# ---------------------------------------------------------------------- #
# Classifier
# ---------------------------------------------------------------------- #
_C.MODEL.CLASSIFIER = CfgNode()
_C.MODEL.CLASSIFIER.NAME = "build_softmax_classifier"

# SoftmaxClassifier options
_C.MODEL.CLASSIFIER.SOFTMAX_CLASSIFIER = CfgNode()
_C.MODEL.CLASSIFIER.SOFTMAX_CLASSIFIER.FEATURE_DIM = 64
_C.MODEL.CLASSIFIER.SOFTMAX_CLASSIFIER.NUM_CLASSES = 10
_C.MODEL.CLASSIFIER.SOFTMAX_CLASSIFIER.NUM_HEADS = 1
_C.MODEL.CLASSIFIER.SOFTMAX_CLASSIFIER.USE_BIAS = True
_C.MODEL.CLASSIFIER.SOFTMAX_CLASSIFIER.LINEAR_LAYERS = "Linear"

# DUQClassifier options
_C.MODEL.CLASSIFIER.DUQ_CLASSIFIER = CfgNode()
_C.MODEL.CLASSIFIER.DUQ_CLASSIFIER.FEATURE_DIM = 64
_C.MODEL.CLASSIFIER.DUQ_CLASSIFIER.NUM_CLASSES = 10
_C.MODEL.CLASSIFIER.DUQ_CLASSIFIER.CENTROID_DIM = 64
_C.MODEL.CLASSIFIER.DUQ_CLASSIFIER.LENGTH_SCALE = 0.1

# CentroidClassifier options
_C.MODEL.CLASSIFIER.CENTROID_CLASSIFIER = CfgNode()
_C.MODEL.CLASSIFIER.CENTROID_CLASSIFIER.FEATURE_DIM = 64
_C.MODEL.CLASSIFIER.CENTROID_CLASSIFIER.NUM_CLASSES = 10

# ---------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------- #
_C.SOLVER = CfgNode()
_C.SOLVER.NUM_EPOCHS = 200
_C.SOLVER.BATCH_SIZE = 128
_C.SOLVER.VALID_FREQUENCY = 20
_C.SOLVER.VALID_FINALE = 20

_C.SOLVER.OPTIMIZER = CfgNode()
_C.SOLVER.OPTIMIZER.NAME = "SGD"

_C.SOLVER.SCHEDULER = CfgNode()
_C.SOLVER.SCHEDULER.NAME = "WarmupSimpleCosineLR"

# SGD optimizer options
_C.SOLVER.OPTIMIZER.SGD = CfgNode()
_C.SOLVER.OPTIMIZER.SGD.BASE_LR = 0.1
_C.SOLVER.OPTIMIZER.SGD.WEIGHT_DECAY = 0.0005
_C.SOLVER.OPTIMIZER.SGD.MOMENTUM = 0.9
_C.SOLVER.OPTIMIZER.SGD.NESTEROV = False
_C.SOLVER.OPTIMIZER.SGD.DECOUPLED_WEIGHT_DECAY = False

# SGD optimizer options for BatchEnsemble
_C.SOLVER.OPTIMIZER.SGD.SUFFIX_BE = ["alpha_be", "gamma_be",]
_C.SOLVER.OPTIMIZER.SGD.BASE_LR_BE = 0.1
_C.SOLVER.OPTIMIZER.SGD.WEIGHT_DECAY_BE = 0.0000
_C.SOLVER.OPTIMIZER.SGD.MOMENTUM_BE = 0.9
_C.SOLVER.OPTIMIZER.SGD.NESTEROV_BE = False
_C.SOLVER.OPTIMIZER.SGD.DECOUPLED_WEIGHT_DECAY_BE = False

# SGHMC optimizer options
_C.SOLVER.OPTIMIZER.SGHMC = CfgNode()
_C.SOLVER.OPTIMIZER.SGHMC.BASE_LR = 0.1
_C.SOLVER.OPTIMIZER.SGHMC.BASE_LR_SCALE = 1.0 / 45000
_C.SOLVER.OPTIMIZER.SGHMC.WEIGHT_DECAY = 0.0005
_C.SOLVER.OPTIMIZER.SGHMC.MOMENTUM_DECAY = 0.9
_C.SOLVER.OPTIMIZER.SGHMC.TEMPERATURE = 1.0

# WarmupSimpleCosineLR scheduler options
_C.SOLVER.SCHEDULER.WARMUP_SIMPLE_COSINE_LR = CfgNode()
_C.SOLVER.SCHEDULER.WARMUP_SIMPLE_COSINE_LR.WARMUP_EPOCHS = 5
_C.SOLVER.SCHEDULER.WARMUP_SIMPLE_COSINE_LR.WARMUP_METHOD = "linear"
_C.SOLVER.SCHEDULER.WARMUP_SIMPLE_COSINE_LR.WARMUP_FACTOR = 0.01

# WarmupCyclicalCosineLR scheduler options
_C.SOLVER.SCHEDULER.WARMUP_CYCLICAL_COSINE_LR = CfgNode()
_C.SOLVER.SCHEDULER.WARMUP_CYCLICAL_COSINE_LR.WARMUP_EPOCHS = 5
_C.SOLVER.SCHEDULER.WARMUP_CYCLICAL_COSINE_LR.WARMUP_METHOD = "linear"
_C.SOLVER.SCHEDULER.WARMUP_CYCLICAL_COSINE_LR.WARMUP_FACTOR = 0.01
_C.SOLVER.SCHEDULER.WARMUP_CYCLICAL_COSINE_LR.PRETRAIN_EPOCHS = 160
_C.SOLVER.SCHEDULER.WARMUP_CYCLICAL_COSINE_LR.REPEATED_EPOCHS = [120, 139,]

# WarmupMultiStepLR scheduler options
_C.SOLVER.SCHEDULER.WARMUP_MULTI_STEP_LR = CfgNode()
_C.SOLVER.SCHEDULER.WARMUP_MULTI_STEP_LR.MILESTONES = [5, 100, 150,]
_C.SOLVER.SCHEDULER.WARMUP_MULTI_STEP_LR.WARMUP_METHOD = "linear"
_C.SOLVER.SCHEDULER.WARMUP_MULTI_STEP_LR.WARMUP_FACTOR = 0.01
_C.SOLVER.SCHEDULER.WARMUP_MULTI_STEP_LR.GAMMA = 0.1

# WarmupLinearDecayLR scheduler options
_C.SOLVER.SCHEDULER.WARMUP_LINEAR_DECAY_LR = CfgNode()
_C.SOLVER.SCHEDULER.WARMUP_LINEAR_DECAY_LR.MILESTONES = [5, 100, 180,]
_C.SOLVER.SCHEDULER.WARMUP_LINEAR_DECAY_LR.WARMUP_METHOD = "linear"
_C.SOLVER.SCHEDULER.WARMUP_LINEAR_DECAY_LR.WARMUP_FACTOR = 0.01
_C.SOLVER.SCHEDULER.WARMUP_LINEAR_DECAY_LR.GAMMA = 0.01

# ---------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------- #

# The number of GPUs
_C.NUM_GPUS = 1

# If inputs images have the same sizes, benchmark is often helpful
_C.CUDNN_BENCHMARK = False

# If normally-nondeterministic operations should be deterministic
# Refer to https://pytorch.org/docs/stable/notes/randomness.html
_C.CUDNN_DETERMINISTIC = False

# Set seed to positive to use a fixed seed
_C.SEED = -1

# Frequency of logging during training
_C.LOG_FREQUENCY = 4

# Directory where output files are written
_C.OUTPUT_DIR = "./outputs/debug/"
