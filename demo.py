from Visualizer import Visualizer
from detectron2.engine import DefaultPredictor
import argparse
import glob
import multiprocessing as mp
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import cv2
import tqdm
import time

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode
from predictor import VisualizationDemo

# constants
WINDOW_NAME = "detections"
ROOT_DIR = os.getcwd()

# inference
INPUT_IMG_PATH = os.path.join(ROOT_DIR, 'input_img/')
INPUT_VIDEO_PATH = os.path.join(ROOT_DIR, 'input_video/')
OUTPUT_IMG_PATH = os.path.join(ROOT_DIR, 'out_img/')
OUTPUT_VIDEO_PATH = os.path.join(ROOT_DIR, 'out_video/')

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
    thing_colors = [k["color"]
                    for k in DATASET_CATEGORIES if k["isthing"] == 1]
    # assert len(thing_ids) == 2, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"]
                     for k in DATASET_CATEGORIES if k["isthing"] == 1]
    if keypt:
        if len(skeleton) % 2 != 0:
            print("请检查关键点输入是否符合格式（注释有格式说明）！")
            exit(0)
        keypoint_names = [k["name"]
                          for k in DATASET_CATEGORIES if k["iskeypoint"] == 1]
        keypoint_connection_rules = []
        for i in range(0, len(skeleton), 2):
            keypoint_connection_rules.append(
                [DATASET_CATEGORIES[skeleton[i][0] + 1]["name"], DATASET_CATEGORIES[skeleton[i][1] + 1]["name"], skeleton[1 + i]])
        ret = {
            "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
            "thing_classes": thing_classes,
            "thing_colors": thing_colors,
            "keypoint_names": keypoint_names,
            "keypoint_connection_rules": keypoint_connection_rules,
            "keypoint_flip_map": keypoint_flip_map,
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
    DatasetCatalog.register(
        name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file,
                                  image_root=image_root,
                                  evaluator_type="coco",
                                  **metadate)


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    args.config_file = "../detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(args.config_file)   # 从config file 覆盖配置
    cfg.merge_from_list(args.opts)          # 从CLI参数 覆盖配置
    
    # 更改配置参数
    cfg.DATASETS.TRAIN = ("bird_train",)
    cfg.DATASETS.TEST = ("bird_val",)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95
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
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold

    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    # 是否读取摄像头
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    
    # 是否读取视频
    parser.add_argument("--video-input", help="Path to video file.")

    parser.add_argument(
        "--config-file",
        # default="../faster_rcnn_R_101_FPN_3x.yaml",
        default="../detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.95,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def get_Predictions_Info(predictions):
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
    keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None
    return boxes, scores, classes, keypoints


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    
    # 注册数据集
    register_dataset()
    
    demo = VisualizationDemo(cfg, instance_mode=ColorMode.SEGMENTATION)

    if args.webcam:
        assert args.input is None, "不可以同时输入 --input 和 --webcam!"
        time_now = time.strftime("%Y%m%d%H%M", time.localtime()) # 获取当前时间
        cap = cv2.VideoCapture(0)
        fps = cap.get(cv2.CAP_PROP_FPS)  # 返回视频的fps--帧率
        path = OUTPUT_VIDEO_PATH + time_now + "/" #获取当前路径
        vpath =  path + 'output.mp4'#视频目录
        if not(os.path.exists(path)):
            #print('n')  #没有就建一个
            os.makedirs(path)
        output_fname = time_now + '.mp4'# 以当前时间命名文件
        output_video = cv2.VideoWriter(
                filename=vpath,
                fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                fps=10,
                frameSize=(640, 480),
            )
        """
            参数1 即将保存的文件路径
            参数2 VideoWriter_fourcc为视频编解码器
            fourcc意为四字符代码（Four-Character Codes），顾名思义，该编码由四个字符组成,下面是VideoWriter_fourcc对象一些常用的参数,注意：字符顺序不能弄混
            cv2.VideoWriter_fourcc('I', '4', '2', '0'),该参数是YUV编码类型，文件名后缀为.avi 
            cv2.VideoWriter_fourcc('P', 'I', 'M', 'I'),该参数是MPEG-1编码类型，文件名后缀为.avi 
            cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),该参数是MPEG-4编码类型，文件名后缀为.avi 
            cv2.VideoWriter_fourcc('T', 'H', 'E', 'O'),该参数是Ogg Vorbis,文件名后缀为.ogv 
            cv2.VideoWriter_fourcc('F', 'L', 'V', '1'),该参数是Flash视频，文件名后缀为.flv
            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),文件名后缀为.mp4
            参数3 为帧播放速率
            参数4 (width,height)为视频帧大小
        """
        for vis in tqdm.tqdm(demo.run_on_video(cap)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            output_video.write(vis)
            if cv2.waitKey(1) == 27:
                break  # 按Esc键结束
        cap.release()
        output_video.release()
        cv2.destroyAllWindows()

    elif args.video_input:
        for videofile in os.listdir(INPUT_VIDEO_PATH):
            video_fullname = os.path.join(INPUT_VIDEO_PATH, videofile)
            video = cv2.VideoCapture(video_fullname)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames_per_second = video.get(cv2.CAP_PROP_FPS)
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            basename = os.path.basename(videofile)
            output_fname = os.path.join(OUTPUT_VIDEO_PATH, basename)
            output_fname = os.path.splitext(output_fname)[0] + ".mp4"
            output_file = cv2.VideoWriter(
                filename=output_fname,
                fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                fps=10,
                frameSize=(width, height),
                isColor=True,
            )
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
            cv2.imshow(basename, vis_frame)
            output_file.write(vis_frame)
            if cv2.waitKey(1) == 27:
                break  # 按Esc键结束
        video.release()
        output_file.release()
        cv2.destroyAllWindows()
        
    else:
        for imgfile in os.listdir(INPUT_IMG_PATH):

            # use PIL, to be consistent with evaluation
            img_fullName = os.path.join(INPUT_IMG_PATH, imgfile)
            img = read_image(img_fullName, format="BGR")
            start_time = time.time()
            predictor = DefaultPredictor(cfg)
            outputs = predictor(img)
            v = Visualizer(img[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            boxes, scores, classes, keypoints = get_Predictions_Info(outputs["instances"]);
            print(keypoints)

            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            img = out.get_image()[:, :, ::-1]
            cv2.imshow(WINDOW_NAME, img)
            img_name = OUTPUT_IMG_PATH + os.path.basename(imgfile)
            print(img_name)
            cv2.imwrite(img_name , img)
            if cv2.waitKey(0) == 27:
                continue  # 按Esc键继续下一个图片

