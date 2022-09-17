import os
from mim.commands.download import download

os.makedirs('models', exist_ok=True)

checkpoint_name = 'faster_rcnn_r50_fpn_1x_coco'
config_fname = checkpoint_name + '.py'

checkpoint = download(package="mmdet", configs=[checkpoint_name], dest_root="models")[0]
# checkpoint: faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
from mmcv import Config

cfg = Config.fromfile(os.path.join('models', config_fname))

####
## modify model configuration file
####

# modify dataset type and path
cfg.dataset_type = 'VOCDataset' # default: CocoDataset
cfg.data_root = 'data/VOCdevkit/' # default: data/coco

cfg.data.train.type = 'VOCDataset'
cfg.data.train.data_root = 'data/VOCdevkit/'
cfg.data.train.ann_file = 'VOC2007/ImageSets/Main/trainval.txt'
cfg.data.train.img_prefix = 'VOC2007/'

cfg.data.test.type = 'VOCDataset'
cfg.data.test.data_root = 'data/VOCdevkit/'
cfg.data.test.ann_file = 'VOC2007/ImageSets/Main/val.txt'
cfg.data.test.img_prefix = 'VOC2007/'

cfg.data.val.type = 'VOCDataset'
cfg.data.val.data_root = 'data/VOCdevkit/'
cfg.data.val.ann_file = 'VOC2007/ImageSets/Main/val.txt'
cfg.data.val.img_prefix = 'VOC2007/'

# number of classes
cfg.model.roi_head.bbox_head.num_classes = 20 # default: 80

# checkpoint path
cfg.load_from = os.path.join('models', checkpoint)

# learning rate
cfg.optimizer.lr = 0.02 / 8 # default: 0.02

# evaluation metric
cfg.evaluation.metric = 'mAP'

# modify cuda setting
cfg.gpu_ids = range(1)
cfg.device = 'cuda'

# set seed
cfg.seed = 0

# set epochs
cfg.runner.max_epochs = 3 #default: 12

# set output dir
cfg.work_dir = 'output'

from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import os.path as osp
import mmcv

# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(cfg.model)
model.CLASSES = datasets[0].CLASSES

mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)
