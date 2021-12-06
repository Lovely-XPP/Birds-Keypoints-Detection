import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import logging
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.utils.visualizer import Visualizer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.modeling import GeneralizedRCNNWithTTA

# 数据集路径
ROOT_DIR = os.getcwd()
DATASET_ROOT = ROOT_DIR
ANN_ROOT = os.path.join(DATASET_ROOT, 'coco/annotations/')
TRAIN_PATH = os.path.join(DATASET_ROOT, 'coco/train2019/')
VAL_PATH = os.path.join(DATASET_ROOT, 'coco/val2019/')
TRAIN_JSON = os.path.join(ANN_ROOT, 'instances_train2019.json')
VAL_JSON = os.path.join(ANN_ROOT, 'instances_val2019.json')

# 数据集类别元数据
DATASET_CATEGORIES = [
    {"name": "bird", "id": 1, "isthing": 1, "iskeypoint": 0, "color": [0, 255, 255]},
    {"name": "flying_bird", "id": 2, "isthing": 1, "iskeypoint": 0, "color": [0, 255, 255]},
    {"name": "mouth", "id": 1, "isthing": 0, "iskeypoint": 1, "color": [0, 139, 139]},
    {"name": "head", "id": 2, "isthing": 0, "iskeypoint": 1, "color": [25, 25, 112]},
    {"name": "eye_L", "id": 3, "isthing": 0, "iskeypoint": 1, "color": [139, 105, 20]},
    {"name": "eye_R", "id": 4, "isthing": 0, "iskeypoint": 1, "color": [139, 105, 20]},
    {"name": "body", "id": 5, "isthing": 0, "iskeypoint": 1, "color": [0, 139, 0]},
    {"name": "wing_mid_L", "id": 6, "isthing": 0, "iskeypoint": 1, "color": [0, 0, 205]},
    {"name": "wing_L", "id": 7, "isthing": 0, "iskeypoint": 1, "color": [0, 0, 0]},
    {"name": "wing_mid_R", "id": 8, "isthing": 0, "iskeypoint": 1, "color": [0, 0, 205]},
    {"name": "wing_R", "id": 9, "isthing": 0, "iskeypoint": 1, "color": [0, 0, 0]},
    {"name": "tail", "id": 10, "isthing": 0, "iskeypoint": 1, "color": [85, 26, 139]}
]

# 是否开启关键点检测
keypt = True

# 对称的时候左右相反，需要注明
keypoint_flip_map = [["eye_L", "eye_R"],
                     ["eye_R", "eye_L"],
                     ["wing_mid_L", "wing_mid_R"],
                     ["wing_mid_R", "wing_mid_L"],
                     ["wing_L", "wing_R"],
                     ["wing_R", "wing_L"],
                    ]

# 关键点连接
skeleton = [[1,2], # 第 1 组连接线，下同
            [0, 255, 255], # 第 1 组连接线的颜色，下同
            [2,3],
            [85, 26, 139],
            [2,4],
            [85, 26, 139],
            [2,5],
            [0, 0, 0],
            [5,6],
            [0, 139, 139],
            [6,7],
            [139, 105, 20],
            [5,8],
            [0, 139, 139],
            [8,9],
            [139, 105, 20],
            [5,10],
            [0, 139, 0]
            ]

# 声明数据集的子集
PREDEFINED_SPLITS_DATASET = {
    "bird_train": (TRAIN_PATH, TRAIN_JSON),
    "bird_val": (VAL_PATH, VAL_JSON),
}

def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key, 
                                   metadate=get_dataset_instances_meta(), 
                                   json_file=json_file, 
                                   image_root=image_root)


def get_dataset_instances_meta():
    """
    purpose: get metadata of dataset from DATASET_CATEGORIES
    return: dict[metadata]
    """
    thing_ids = [k["id"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in DATASET_CATEGORIES if k["iskeypoint"] == 1]
    # assert len(thing_ids) == 2, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    if keypt:
        if len(skeleton) % 2 != 0:
            print("请检查关键点输入是否符合格式（注释有格式说明）！")
            exit(0)
        keypoint_names = [k["name"] for k in DATASET_CATEGORIES if k["iskeypoint"] == 1]
        keypoint_connection_rules = []
        for i in range(0, len(skeleton), 2):
            keypoint_connection_rules.append([DATASET_CATEGORIES[skeleton[i][0] + 1]["name"], 
                DATASET_CATEGORIES[skeleton[i][1] + 1]["name"], skeleton[1 + i]])
        ret = {
            "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
            "thing_classes": thing_classes,
            "thing_colors": thing_colors,
            "keypoint_names": keypoint_names,
            "keypoint_connection_rules": keypoint_connection_rules,
            "keypoint_flip_map":keypoint_flip_map,
            }
    else:
        ret = {
            "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
            "thing_classes": thing_classes,
            "thing_colors": thing_colors,
            }
    return ret

# 注册数据集和元数据
def register_dataset_instances(name, metadate, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file, 
                                  image_root=image_root, 
                                  evaluator_type="coco", 
                                  **metadate)




class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, distributed=False, output_dir=output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg() # 拷贝default config副本
    args.config_file = "../detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(args.config_file)   # 从config file 覆盖配置
    cfg.merge_from_list(args.opts)          # 从CLI参数 覆盖配置

    # 更改配置参数
    cfg.DATASETS.TRAIN = ("bird_train",)
    cfg.DATASETS.TEST = ("bird_val",)
    cfg.DATALOADER.NUM_WORKERS = 4  # 单线程
    cfg.INPUT.MAX_SIZE_TRAIN = 400
    cfg.INPUT.MAX_SIZE_TEST = 400
    cfg.INPUT.MIN_SIZE_TRAIN = (160,)
    cfg.INPUT.MIN_SIZE_TEST = 160
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 # 类别数
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 10  # 关键点数量
    cfg.TEST.KEYPOINT_OKS_SIGMAS = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    # cfg.MODEL.WEIGHTS = "./R50-FPN-3x.pkl"  # 预训练模型权重
    # cfg.MODEL.WEIGHTS = "./output/model_final.pth"   # 最终权重
    cfg.SOLVER.IMS_PER_BATCH = 5  # batch_size=2; iters_in_one_epoch = dataset_imgs/batch_size
    ITERS_IN_ONE_EPOCH = 40
    cfg.SOLVER.MAX_ITER = 200# 12 epochs
    cfg.SOLVER.BASE_LR = 0.002
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WEIGHT_DECAY = 0.001
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (100,)
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WARMUP_ITERS = 10
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    print(cfg)

    # 注册数据集
    register_dataset()

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        print(cfg.TEST.KEYPOINT_OKS_SIGMAS)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=True)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    ) 
