import json
import os, sys, cv2
import numpy as np

# 图片设置: 和coco注释的image_id对应
image_id = 1
# 路径设置
root = sys.path[0]
train_json = os.path.join(root, 'COCO/annotations/instances_train.json')
train_path = os.path.join(root, 'COCO/train')

def visualization_bbox1(num_image, json_path,img_path):# 需要画的第num副图片， 对应的json路径和图片路径
    with open(json_path) as annos:
        annotation_json = json.load(annos)

    print('the annotation_json num_key is:',len(annotation_json))  # 统计json文件的关键字长度
    print('the annotation_json key is:', annotation_json.keys()) # 读出json文件的关键字
    print('the annotation_json num_images is:', len(annotation_json['images'])) # json文件中包含的图片数量

    image_name = annotation_json['images'][num_image - 1]['file_name']  # 读取图片名
    id = annotation_json['images'][num_image - 1]['id']  # 读取图片id

    image_path = os.path.join(img_path, str(image_name).zfill(5)) # 拼接图像路径
    image = cv2.imread(image_path, 1)  # 保持原始格式的方式读取图像
    num_bbox = 0  # 统计一幅图片中bbox的数量

    for i in range(len(annotation_json['annotations'][::])):
        if  annotation_json['annotations'][i-1]['image_id'] == id:
            bgr=np.random.randint(0,255,3,dtype=np.int32)
            num_bbox = num_bbox + 1
            x, y, w, h = annotation_json['annotations'][i-1]['bbox']  # 读取边框
            keypoints = annotation_json['annotations'][i-1]['keypoints']
            for i in range(0, len(keypoints), 3):
                if keypoints[i+2] == 0:
                    continue
                image = cv2.circle(image, (int(keypoints[i]), int(keypoints[i+1])), 8, (np.int(bgr[0]),np.int(bgr[1]),np.int(bgr[2])), thickness=-1)
            image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (np.int(bgr[0]),np.int(bgr[1]),np.int(bgr[2])), 2)

    print('The unm_bbox of the display image is:', num_bbox)

    # 显示方式1：用plt.imshow()显示
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) #绘制图像，将CV的BGR换成RGB
    # plt.show() #显示图像

    # 显示方式2：用cv2.imshow()显示
    cv2.namedWindow(image_name, 0)  # 创建窗口
    cv2.resizeWindow(image_name, 1000, 1000) # 创建500*500的窗口
    cv2.imshow(image_name, image)
    cv2.waitKey(0)

if __name__ == "__main__":
   visualization_bbox1(image_id, train_json, train_path)
