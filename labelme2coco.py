import os
import json
import numpy as np
import glob
import shutil
from sklearn.model_selection import train_test_split
from labelme import utils
np.random.seed(41)

#---------------------- 可修改选项 ----------------------#

# 设置图像的标记类别,0为背景(一定要根据labelme的标记改,否则会报错)
classname_to_id = {"bird": 1, "flying_bird": 2}

# 图像关键点标签（留空代表无keypoint）
keypoints = ["mouth",           # 1
             "head",            # 2
             "eye_L",           # 3
             "eye_R",           # 4
             "body",            # 5
             "wing_L",          # 7
             "wing_R",          # 9
             "tail"             # 10
            ]

            # 图像关键点标签（留空代表无keypoint）
keypoints_f = ["mouth",           # 1
             "head",            # 2
             "eye_L",           # 3
             "eye_R",           # 4
             "body",            # 5
             "wing_mid_L",      # 6
             "wing_L",          # 7
             "wing_mid_R",      # 8
             "wing_R",          # 9
             "tail"             # 10
            ]

# 对称的时候左右相反，需要注明
keypoint_flip_map = [["eye_L", "eye_R"],
                     ["wing_L", "wing_R"],
                     ["eye_R", "eye_L"],
                     ["wing_R", "wing_l"],
                    ]

keypoint_flip_map_f = [["eye_L", "eye_R"],
                     ["eye_R", "eye_L"],
                     ["wing_mid_L", "wing_mid_R"],
                     ["wing_mid_R", "wing_mid_L"],
                     ["wing_L", "wing_R"],
                     ["wing_R", "wing_L"],
                    ]

# 检测关键点的物体名称，如："person"，但要与标注的bbox标签对应
obj_name = ["bird", "flying_bird"]

# 图像关键点的连接，注意序号对应
skeleton = [[1,2], 
            [2,3],
            [2,4],
            [2,5],
            [5,7],
            [5,9],
            [5,10]
            ]

skeleton_f = [[1,2], 
            [2,3],
            [2,4],
            [2,5],
            [5,6],
            [6,7],
            [5,8],
            [8,9],
            [5,10]
            ]

# 测试集所占比例
test_rate = 0.1

#------------------------------------------------#

# - 若无问题请勿更改以下的代码 - #
print("--------------- 标签概况 -----------------")
print("物体识别类型：")
print(classname_to_id)
print()

ROOT_DIR = os.getcwd()
labelme_path = ROOT_DIR + '/pic/'
saved_coco_path = "./"

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
            obj = self.read_jsonfile(json_path)
            self.images.append(self._image(obj, json_path))
            shapes = obj['shapes']
            annotation = {}
            if keypt:
                annotation, F_bird = self._annotation_keypoint(shapes)
            for shape in shapes:
                if shape["shape_type"] != "point":
                    annotation = self._annotation(annotation, shape, F_bird)
                    self.annotations.append(annotation)
                    self.ann_id += 1
                    break
            self.img_id += 1
        self._init_categories(F_bird)
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 构建类别
    def _init_categories(self, F_bird):
        if keypt:
            category = {}
            category['supercategory'] = obj_name[0]
            category['id'] = 1
            category['name'] = obj_name[0]
            category['keypoint'] = keypoints
            category['skeleton'] = skeleton
            category['keypoint_flip_map'] = keypoint_flip_map
            self.categories.append(category)
            category = {}
            category['supercategory'] = obj_name[1]
            category['id'] = 2
            category['name'] = obj_name[1]
            category['keypoint'] = keypoints_f
            category['skeleton'] = skeleton_f
            category['keypoint_flip_map'] = keypoint_flip_map_f
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
        img_x = utils.img_b64_to_arr(obj['imageData'])
        h, w = img_x.shape[:-1]
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path).replace(".json", ".jpg")
        return image

    # 构建COCO的annotation字段
    def _annotation(self, annotation, shape, F_bird):
        points = shape['points']
        label = shape['label']
        if F_bird == 1:
            label = obj_name[1]
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
    def _annotation_keypoint(self, shapes):
        F_bird = 0
        annotation = {}
        temp_keypoints = [ ]
        num_keypoints = 0
        for keypoint in keypoints_f:
            temp = []
            temp.clear
            for shape in shapes:
                if keypoint == shape["label"]:
                    temp = shape["points"]
                    if keypoint == "wing_mid_L" or keypoint == "wing_mid_R":
                        F_bird = 1
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
        return annotation, F_bird

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
    rootdir="./coco/"
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
    if not os.path.exists("%scoco/annotations/"%saved_coco_path):
        os.makedirs("%scoco/annotations/"%saved_coco_path)
    if not os.path.exists("%scoco/train2019/"%saved_coco_path):
        os.makedirs("%scoco/train2019"%saved_coco_path)
    if not os.path.exists("%scoco/val2019/"%saved_coco_path):
        os.makedirs("%scoco/val2019"%saved_coco_path)
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
    l2c_train.save_coco_json(train_instance, '%scoco/annotations/instances_train2019.json'%saved_coco_path)
    for file in train_path:
        shutil.copy(file.replace("json","jpg"),"%scoco/train2019/"%saved_coco_path)
    for file in val_path:
        shutil.copy(file.replace("json","jpg"),"%scoco/val2019/"%saved_coco_path)

    # 把验证集转化为COCO的json格式
    l2c_val = Lableme2CoCo()
    val_instance = l2c_val.to_coco(val_path)
    l2c_val.save_coco_json(val_instance, '%scoco/annotations/instances_val2019.json'%saved_coco_path)
