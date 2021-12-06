from genericpath import exists
from posixpath import basename
import numpy as np
from detectron2.engine import DefaultPredictor
import multiprocessing as mp
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode

# constants
WINDOW_NAME = "detections"
ROOT_DIR = os.getcwd()

# inference
INPUT_IMG_PATH = os.path.join(ROOT_DIR, 'input_img/')
OUTPUT_DATA_PATH = os.path.join(ROOT_DIR, 'out_data/')

# 数据集路径
DATASET_ROOT = ROOT_DIR
ANN_ROOT = os.path.join(DATASET_ROOT, 'coco/annotations/')
TRAIN_PATH = os.path.join(DATASET_ROOT, 'coco/train2019/')
VAL_PATH = os.path.join(DATASET_ROOT, 'coco/val2019/')
TRAIN_JSON = os.path.join(ANN_ROOT, 'instances_train2019.json')
VAL_JSON = os.path.join(ANN_ROOT, 'instances_val2019.json')

# 数据集类别元数据
DATASET_CATEGORIES = [
    {"name": "bird", "id": 1, "isthing": 1,
        "iskeypoint": 0, "color": [0, 255, 255]},
    {"name": "flying_bird", "id": 2, "isthing": 1,
        "iskeypoint": 0, "color": [0, 255, 255]},
    {"name": "mouth", "id": 1, "isthing": 0,
        "iskeypoint": 1, "color": [0, 139, 139]},
    {"name": "head", "id": 2, "isthing": 0,
        "iskeypoint": 1, "color": [25, 25, 112]},
    {"name": "eye_L", "id": 3, "isthing": 0,
        "iskeypoint": 1, "color": [139, 105, 20]},
    {"name": "eye_R", "id": 4, "isthing": 0,
        "iskeypoint": 1, "color": [139, 105, 20]},
    {"name": "body", "id": 5, "isthing": 0,
        "iskeypoint": 1, "color": [0, 139, 0]},
    {"name": "wing_mid_L", "id": 6, "isthing": 0,
        "iskeypoint": 1, "color": [0, 0, 205]},
    {"name": "wing_L", "id": 7, "isthing": 0,
        "iskeypoint": 1, "color": [0, 0, 0]},
    {"name": "wing_mid_R", "id": 8, "isthing": 0,
        "iskeypoint": 1, "color": [0, 0, 205]},
    {"name": "wing_R", "id": 9, "isthing": 0,
        "iskeypoint": 1, "color": [0, 0, 0]},
    {"name": "tail", "id": 10, "isthing": 0,
        "iskeypoint": 1, "color": [85, 26, 139]}
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
skeleton = [[1, 2],  # 第 1 组连接线，下同
            [0, 255, 255],  # 第 1 组连接线的颜色，下同
            [2, 3],
            [85, 26, 139],
            [2, 4],
            [85, 26, 139],
            [2, 5],
            [0, 0, 0],
            [5, 6],
            [0, 139, 139],
            [6, 7],
            [139, 105, 20],
            [5, 8],
            [0, 139, 139],
            [8, 9],
            [139, 105, 20],
            [5, 10],
            [0, 139, 0]
            ]



def setup_cfg():
    # load config from file and command-line arguments
    cfg = get_cfg()
    config_file = "../detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(config_file)   # 从config file 覆盖配置
    
    # 更改配置参数
    cfg.DATASETS.TRAIN = ("bird_train",)
    cfg.DATASETS.TEST = ("bird_val",)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.DATALOADER.NUM_WORKERS = 4  # 单线程
    cfg.INPUT.MAX_SIZE_TRAIN = 400
    cfg.INPUT.MAX_SIZE_TEST = 400
    cfg.INPUT.MIN_SIZE_TRAIN = (160,)
    cfg.INPUT.MIN_SIZE_TEST = 160
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 # 类别数
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 10  # 关键点数量
    cfg.TEST.KEYPOINT_OKS_SIGMAS = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    cfg.MODEL.WEIGHTS = "./output/model_final.pth"  # 预训练模型权重
    # cfg.MODEL.WEIGHTS = "./output/model_final.pth"   # 最终权重
    # batch_size=2; iters_in_one_epoch = dataset_imgs/batch_size
    cfg.SOLVER.IMS_PER_BATCH = 10
    ITERS_IN_ONE_EPOCH = 500
    cfg.SOLVER.MAX_ITER = 1500  # 12 epochs
    cfg.SOLVER.BASE_LR = 0.002
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WEIGHT_DECAY = 0.001
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (100,)
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WARMUP_ITERS = 20
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5

    cfg.freeze()
    return cfg


def get_Predictions_Info(predictions):
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
    keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

    a_boxes = []
    a_scores = []
    a_classes = []
    a_keypoints = []

    for box in boxes:
        a_boxes.append(box.tolist())

    for score in scores:
        a_scores.append(score.item())

    for cls in classes:
        a_classes.append(cls)

    for points in keypoints:
        a_keypoints.append(points.tolist())
    
    dest = a_scores.index(max(a_scores))
    a_boxes = np.array(a_boxes)
    a_scores = np.array(a_scores)
    a_keypoints = np.array(a_keypoints)
    a_classes = np.array(a_classes)
    a_boxes = a_boxes[dest,:]
    a_classes = a_classes[dest]
    a_keypoints = a_keypoints[dest][:,:]
    a_scores = max(a_scores)
    print(a_boxes)
    print(a_scores)
    print(a_classes)
    print(a_keypoints)
    return a_boxes, a_scores, a_classes, a_keypoints


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    logger = setup_logger()

    cfg = setup_cfg()
    if not exists(OUTPUT_DATA_PATH):
        os.mkdir(OUTPUT_DATA_PATH)

    for imgfile in os.listdir(INPUT_IMG_PATH):
        img_fullName = os.path.join(INPUT_IMG_PATH, imgfile)
        img = read_image(img_fullName, format="BGR")
        predictor = DefaultPredictor(cfg)
        outputs = predictor(img)
        if len(outputs["instances"].scores) == 0:
            dect = 0
            boxes = []
            scores = []
            classes = []
            keypoints = []
            print("无检测结果")
        else:
            dect = 1
            boxes, scores, classes, keypoints = get_Predictions_Info(outputs["instances"])
        
        if '.' in imgfile:
            imgfile = imgfile.split('.')[0]
        data_name = OUTPUT_DATA_PATH + imgfile + '.npz'
        np.savez(data_name, dect=dect,boxes=boxes,scores=scores,classes=classes,keypoints=keypoints)
        print(imgfile)
        print(img_fullName)
