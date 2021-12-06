import numpy as np
import os



'''
总体设置区域
'''
# 关键点阈值和颜色
_KEYPOINT_THRESHOLD = 0.5
keypoint_color = [255, 0, 0]


# 数据标注部分（无需改动）



def load_pred_label(img_name):
    # constants
    ROOT_DIR = os.getcwd()
    OUTPUT_DATA_PATH = os.path.join(ROOT_DIR, 'out_data/')
    data_name = OUTPUT_DATA_PATH + img_name + '.npz'

    # dect, boxes, scores, classes, keypoints
    datas = np.load(data_name)
    dect = datas['dect']
    boxes = datas['boxes']
    scores = datas['scores']
    classes = datas['classes']
    keypoints = datas['keypoints']
    print(dect)
    print(boxes)
    print(scores)
    print(classes)
    print(keypoints)


def draw_and_connect_keypoints(keypoints):
    visible = {}
    keypoint_names = self.metadata.get("keypoint_names")
    for idx, keypoint in enumerate(keypoints):
        # draw keypoint
        x, y, prob = keypoint
        if prob > _KEYPOINT_THRESHOLD:
            self.draw_circle((x, y), color=keypoint_color)
            if keypoint_names:
                keypoint_name = keypoint_names[idx]
                visible[keypoint_name] = (x, y)

        if self.metadata.get("keypoint_connection_rules"):
            for kp0, kp1, color in self.metadata.keypoint_connection_rules:
                if kp0 in visible and kp1 in visible:
                    x0, y0 = visible[kp0]
                    x1, y1 = visible[kp1]
                    color = tuple(x / 255.0 for x in color)
                    self.draw_line([x0, x1], [y0, y1], color=color)
        return self.output

ROOT_DIR = os.getcwd()
INPUT_IMG_PATH = os.path.join(ROOT_DIR, 'input_img/')
for imgfile in os.listdir(INPUT_IMG_PATH):
    if '.' in imgfile:
        imgfile = imgfile.split('.')[0]
    load_pred_label(imgfile)
