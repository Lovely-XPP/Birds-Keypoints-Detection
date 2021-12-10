import numpy as np
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2
import os

'''
总体设置区域
'''
# 路径设置
ROOT_DIR = os.getcwd()

# 关键点阈值
_KEYPOINT_THRESHOLD = 0.004


'''
数据标注部分（无需改动）
'''
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
keypoint_names = [k["name"] 
                  for k in DATASET_CATEGORIES if k["iskeypoint"] == 1]
keypoint_color = [k["color"]
                  for k in DATASET_CATEGORIES if k["iskeypoint"] == 1]
keypoint_color = np.array(keypoint_color)
keypoint_connection_rules = []
for i in range(0, int(len(skeleton)), 2):
    keypoint_connection_rules.append(
        [DATASET_CATEGORIES[skeleton[i][0]+1]["name"], DATASET_CATEGORIES[skeleton[i][1]+1]["name"], skeleton[1 + i]])

# 读取数据函数
def load_pred_label(img_name):
    # constants
    OUTPUT_DATA_PATH = os.path.join(ROOT_DIR, 'out_data/')
    data_name = OUTPUT_DATA_PATH + img_name + '.npz'

    # dect, boxes, scores, classes, keypoints
    datas = np.load(data_name)
    dect = datas['dect']
    boxes = datas['boxes']
    scores = datas['scores']
    classes = datas['classes']
    keypoints = datas['keypoints']
    # print(dect)
    # print(boxes)
    # print(scores)
    # print(classes)
    # print(keypoints)
    return dect, boxes, scores, classes, keypoints


'''
绘图子函数
'''
# 初始化图片
class Image:
    def __init__(self, img, scale=1):
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._default_font_size = max(
            np.sqrt(self.height * self.width) // 90, 10 // scale
        )
        self.setup_figure(img)

    def setup_figure(self, img):
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        # Need to imshow this first so that other patches can be drawn on top
        ax.imshow(img, extent=(0, self.width, self.height, 0),
                  interpolation="nearest")

        self.fig = fig
        self.ax = ax

    def save(self, filepath):
        self.fig.savefig(filepath)

    def get_image(self):
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")


def draw_box(output, box_coord, alpha=0.5, edge_color="g", line_style="-"):
    x0, y0, x1, y1 = box_coord
    width = x1 - x0
    height = y1 - y0

    linewidth = max(output._default_font_size / 4, 1)

    output.ax.add_patch(
        mpl.patches.Rectangle(
            (x0, y0),
            width,
            height,
            fill=False,
            edgecolor=edge_color,
            linewidth=linewidth * output.scale,
            alpha=alpha,
            linestyle=line_style,
        )
    )
    return output


# 绘制圆点
def draw_circle(output, circle_coord, color, radius=3):
        x, y = circle_coord
        radius = output._default_font_size / 2
        radius = max(radius, 1)
        output.ax.add_patch(
            mpl.patches.Circle(circle_coord, radius=radius,
                               fill=True, color=color)
        )
        return output

# 绘制连接线
def draw_line(output, x_data, y_data, color, linestyle="-", linewidth=1):
    linewidth = output._default_font_size / 3
    linewidth = max(linewidth, 1)
    output.ax.add_line(
            mpl.lines.Line2D(
                x_data,
                y_data,
                linewidth=linewidth * output.scale,
                color=color,
                linestyle=linestyle,
            )
        )
    return output

# 绘制关键点和及其连接线
def draw_and_connect_keypoints(output, keypoints, keypoint_names, keypoint_color, keypoint_connection_rules, classes):
    visible = {}
    for idx, keypoint in enumerate(keypoints.tolist()):
        x, y, prob = keypoint
        if prob > _KEYPOINT_THRESHOLD:
            if classes == 0 and (idx == 5 or idx == 7 or idx == 6 or idx == 8):
                continue
            output = draw_circle(output, (x, y), color=keypoint_color[idx]/255.0)
            if keypoint_names:
                keypoint_name = keypoint_names[idx]
                visible[keypoint_name] = (x, y)

    for kp0, kp1, color in keypoint_connection_rules:
        if kp0 in visible and kp1 in visible:
            x0, y0 = visible[kp0]
            x1, y1 = visible[kp1]
            color = tuple(x / 255.0 for x in color)
            output = draw_line(output, [x0, x1], [y0, y1], color=color)
        
    return output


def fig_instances(img, classes, boxes, keypoints, keypoint_names, keypoint_color, keypoint_connection_rules):
    img = np.asarray(img).clip(0, 255).astype(np.uint8)
    output = Image(img)
    output = draw_box(output, boxes, alpha=0.5, edge_color="g", line_style="-")
    output = draw_and_connect_keypoints(output, keypoints, keypoint_names, keypoint_color, keypoint_connection_rules, classes)
    cv2.imshow('detection', output.get_image())
    if cv2.waitKey(0) == 27:
        return 

'''
主函数
'''
if __name__ == '__main__':
    staus = ["不在飞行", "正在飞行"]
    INPUT_IMG_PATH = os.path.join(ROOT_DIR, 'input_img/')
    for imgfile in os.listdir(INPUT_IMG_PATH):
        if '.' in imgfile:
            imgfile_base = imgfile.split('.')[0]
        dect, boxes, scores, classes, keypoints = load_pred_label(imgfile_base)
        str_ = '---------------------------------------'
        nums_str = len(str_)*2 + 4*2
        print("\n" + str_ + " 检测结果 " + str_ + "\n")
        print("图片名称：" + imgfile)
        print("图片位置：" + INPUT_IMG_PATH)
        if dect == 0:
            print("无检测结果！\n")
            print("-"*nums_str)
            continue
        print("鸟类名称：\n当前状态：" + staus[classes] + "\n")
        print("-"*nums_str)
        img_fullName = os.path.join(INPUT_IMG_PATH, imgfile)
        img = cv2.imread(img_fullName)
        
        fig_instances(img, classes, boxes, keypoints, keypoint_names, keypoint_color, keypoint_connection_rules)
