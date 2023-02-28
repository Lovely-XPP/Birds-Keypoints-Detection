import os, sys
import json
import numpy as np
import glob
import shutil
from sklearn.model_selection import train_test_split

#---------------------- 可修改选项 ----------------------#
# 测试集所占比例
test_rate = 0.5

# 设置图像的标记类别,0为背景(一定要根据labelme的标记改,否则会报错)
classname_to_id = {"astronaut": 1}

# 图像关键点标签（留空代表无keypoint）
keypoints = []
for i in range(1,18):
    keypoints.append(str(i))

# 对称的时候左右相反，需要注明
keypoint_flip_map = [["2", "3"],
                     ["6", "9"],
                     ["7", "11"],
                     ["8", "11"],
                     ["12", "15"],
                     ["13", "16"],
                     ["14", "17"]
                    ]

# 检测关键点的物体名称，如："person"，但要与标注的bbox标签对应
obj_name = ["astronaut"]

# 图像关键点的连接，注意序号对应
skeleton = [["1", "2"],
            ["1", "3"],
            ["2", "4"],
            ["3", "4"],
            ["4", "5"],
            ["5", "6"],
            ["6", "7"],
            ["7", "8"],
            ["5", "9"],
            ["9", "10"],
            ["10", "11"],
            ["6", "12"],
            ["12", "13"],
            ["13", "14"],
            ["9", "15"],
            ["15", "16"],
            ["16", "17"],
            ["12", "15"],
            ]
# print(len(skeleton))

#------------------------------------------------#

# - 若无问题请勿更改以下的代码 - #
print("--------------- 标签概况 -----------------")
print("物体识别类型：")
print(classname_to_id)
print()

ROOT_DIR = sys.path[0]
labelme_path = os.path.join(ROOT_DIR + 'pic')
saved_coco_path = sys.path[0]

# 通过集合是否为0来判断是否开启keypoints写入，
if len(keypoints) == 0:
    keypt = 0
    print("是否开启关键点标注：否\n")
else:
    keypt = 1
    print("是否开启关键点标注：是\n")
    print("关键点标签：" )
    print(keypoints)
    print()
    print("关键点连接：")
    print(skeleton)

class Lableme2CoCo:

    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)  # indent=2 更加美观显示

    # 由json文件构建COCO
    def to_coco(self, json_path_list):
        for json_path in json_path_list:
            # 读取 labelme 的标注文件
            obj = self.read_jsonfile(json_path)
            # 添加 image 字段（图片长宽以及文件名）
            self.images.append(self._image(obj, json_path))
            # 获取单个文件所有标注信息
            shapes = obj['shapes']
            rectangles_shape = []
            # 获取bbox信息，如果有多个，则分开存放
            for shape in shapes:
                if shape["shape_type"] == "rectangle":
                    rectangles_shape.append(shape)
            # 对每个bbox信息进行遍历，判断关键点是否在bbox里面，是的话就认为是同一个notation内
            for rectangle_shape in rectangles_shape:
                if keypt:
                    rectangle_points = rectangle_shape['points']
                    annotation = self._annotation_keypoint(shapes, rectangle_points)
                annotation = self._annotation(annotation, rectangle_shape)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        self._init_categories()
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 构建类别
    def _init_categories(self):
        if keypt:
            category = {}
            category['supercategory'] = obj_name[0]
            category['id'] = 1
            category['name'] = obj_name[0]
            category['keypoint'] = keypoints
            category['skeleton'] = skeleton
            category['keypoint_flip_map'] = keypoint_flip_map
            self.categories.append(category)
        else:
            for k, v in classname_to_id.items():
                category = {}
                category['id'] = v
                category['name'] = k
                self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, obj, path):
        image = {}
        # img_x = utils.img_b64_to_arr(obj['imageData'])
        # h, w = img_x.shape[:-1]
        image['height'] = obj['imageHeight']
        image['width'] = obj['imageWidth']
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path).replace(".json", ".jpg")
        return image

    # 构建COCO的annotation字段（仅包含bbox）
    def _annotation(self, annotation, shape):
        points = shape['points']
        label = shape['label']
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(classname_to_id[label])
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        if shape["shape_type"] == "rectangle":
            temp = annotation['segmentation']
            temp[0].append(temp[0][2]) 
            temp[0].append(temp[0][3])
            temp[0].append(temp[0][2]) 
            temp[0].append(temp[0][1])
            temp[0][2] = temp[0][0]
            annotation['segmentation'] = temp
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        [x1, y1, w, h] = self._get_box(points)
        annotation['area'] = h * w
        return annotation

    # 构建属于keypoint的COCO的annotation字段
    def _annotation_keypoint(self, shapes, rectangle_points):
        annotation = {}
        temp_keypoints = [ ]
        num_keypoints = 0
        p1 = rectangle_points[0]
        p2 = rectangle_points[1]
        x_range = [p1[0], p2[0]]
        y_range = [p1[1], p2[1]]
        x_range.sort()
        y_range.sort()
        # 对已知关键点进行遍历
        for keypoint in keypoints:
            temp = []
            temp.clear
            for shape in shapes:
                # 如果找到标注关键点则存放
                if keypoint == shape["label"]:
                    point = shape["points"][0]
                    # 判断关键点是否在bbox范围内
                    if point[0] > x_range[0] and point[0] < x_range[1]:
                        if point[1] > y_range[0] and point[1] < y_range[1]:
                            temp = shape["points"]
            # 如果遍历都没有找到对应的关键点，则认为不存在
            if len(temp) == 0:
                temp_keypoints.append(0)
                temp_keypoints.append(0)
                temp_keypoints.append(0)
            else:
                temp_keypoints.append(temp[0][0])
                temp_keypoints.append(temp[0][1])
                temp_keypoints.append(2)
                num_keypoints = num_keypoints + 1
        annotation['keypoints'] = temp_keypoints
        annotation['num_keypoints'] = num_keypoints
        # num_keypoints
        return annotation

    # 读取json文件，返回一个json对象
    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]


if __name__ == '__main__':
    rootdir = os.path.join(sys.path[0], "COCO")
    if ((os.path.exists(rootdir))):
        filelist=[]
        filelist=os.listdir(rootdir)
        for f in filelist:
            filepath = os.path.join( rootdir, f )
            if os.path.isfile(filepath):
                os.remove(filepath)
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath,True)
        shutil.rmtree(rootdir,True)
    # 创建文件
    annotations_coco_path = os.path.join(rootdir, "annotations")
    train_coco_path = os.path.join(rootdir, "train")
    val_coco_path = os.path.join(rootdir, "val")
    if not os.path.exists(annotations_coco_path):
        os.makedirs(annotations_coco_path)
    if not os.path.exists(train_coco_path):
        os.makedirs(train_coco_path)
    if not os.path.exists(val_coco_path):
        os.makedirs(val_coco_path)
    # 获取images目录下所有的json文件列表
    json_list_path = glob.glob(labelme_path + "/*.json")
    # 数据划分,这里没有区分val2017和tran2017目录，所有图片都放在images目录下
    train_path, val_path = train_test_split(json_list_path, test_size=test_rate)
    print()
    print("训练集数目：", len(train_path))
    print('验证集数目：', len(val_path))

    # 把训练集转化为COCO的json格式
    l2c_train = Lableme2CoCo()
    train_instance = l2c_train.to_coco(train_path)
    l2c_train.save_coco_json(train_instance, os.path.join(annotations_coco_path, "instances_train.json"))
    for file in train_path:
        shutil.copy(file.replace("json","jpg"), train_coco_path)

    # 把验证集转化为COCO的json格式
    l2c_val = Lableme2CoCo()
    val_instance = l2c_val.to_coco(val_path)
    l2c_val.save_coco_json(val_instance, os.path.join(annotations_coco_path, "instances_val.json"))
    for file in val_path:
        shutil.copy(file.replace("json","jpg"), val_coco_path)
