# -- 仅作数据增强用,已经针对性优化源代码 -- #

import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import sys
bag_path = "../cocoapi-master/PythonAPI/"
if not bag_path in sys.path:
    sys.path.append(bag_path)

from pycocotools.coco import COCO

rootdir="./data/"
if ((os.path.exists(rootdir))):
    filelist=[]
    filelist=os.listdir(rootdir)
    for f in filelist:
        filepath = os.path.join( rootdir, f )
        if os.path.isfile(filepath):
            os.remove(filepath)
        shutil.rmtree(rootdir,True)
os.mkdir("./data/")
os.mkdir("./data/img/")

# the path you want to save your results for coco to voc
save_cvs_path = "./data"#保存csv路径
csv_name = "labels.csv"#csv名称
# datasets_list=['train2014', 'val2014']
datasets_list = ['train2019']#coco数据集里面图片

#path = os.path.abspath(dataDir)


def id2name(coco):
    classes = dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']] = cls['name']
    return classes

def copy_all_files(sourcedir, outputdir):
    sourcedir = os.path.abspath(sourcedir)
    sourcedir = sourcedir + '/'
    outputdir = os.path.abspath(outputdir)
    outputdir = outputdir + '/'
    list = os.listdir(sourcedir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        path = os.path.join(sourcedir, list[i])
        #outputdir = os.path.join(outputdir, list[i])
        shutil.copy(path, outputdir)

# 检测是否已经生成coco数据
if ( os.path.exists('./coco/') ):
    dataDir = './coco'
    img_path = './coco/' + datasets_list[0] + '/'
    copy_all_files(img_path, './data/img')
else:
    print("没有检测到coco数据,请检查!")
    exit(1)

def showimg(coco, dataset, img, classes, cls_id, show=True):
    global dataDir
    I = Image.open('%s/%s/%s' % (dataDir, dataset, img['file_name']))
    # 通过id，得到注释的信息
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
    # print(annIds)
    anns = coco.loadAnns(annIds)
    # print(anns)
    # coco.showAnns(anns)
    objs = []
    for ann in anns:
        class_name = classes[ann['category_id']]
        if 'bbox' in ann:
            bbox = ann['bbox']
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2] + bbox[0])
            ymax = int(bbox[3] + bbox[1])
            obj = [class_name, xmin, ymin, xmax, ymax]
            objs.append(obj)
            draw = ImageDraw.Draw(I)
            draw.rectangle([xmin, ymin, xmax, ymax])
    if show:
        plt.figure()
        plt.axis('off')
        plt.imshow(I)
        plt.show()

    return objs

csv_labels = open(save_cvs_path +"/"+csv_name, "w")

for dataset in datasets_list:
    # ./COCO/annotations/instances_train2014.json
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataset)
    #images = '{}/train{}'.format(dataDir, dataset)
    images = './data/img'#存放图片的文件夹路径,必须是纯图片
    # COCO API for initializing annotated data
    coco = COCO(annFile)
    '''
    COCO 对象创建完毕后会输出如下信息:
    loading annotations into memory...
    Done (t=0.81s)
    creating index...
    index created!
    至此, json 脚本解析完毕, 并且将图片和对应的标注数据关联起来.
    '''
    # show all classes in coco
    classes = id2name(coco)

    classes_names = []
    for key, value in classes.items():
        classes_names.append(value)

    classes_ids = coco.getCatIds(catNms=classes_names)
    img_ids_totoal =[]
    for cls in classes_names:
        # Get ID number of this class
        cls_id = coco.getCatIds(catNms=[cls])
        img_ids = coco.getImgIds(catIds=cls_id)
        for img_id in img_ids:
            img_ids_totoal.append(img_id)
    print(len(img_ids_totoal))
    tem=set(img_ids_totoal)
    temp = list(tem)
    temp.sort()
    print(len(tem))
    print(len(temp))
    for imgId in tqdm(temp):
        img = coco.loadImgs(imgId)[0]
        filename = img['file_name']
        # print(filename)
        abspath_img = os.path.abspath(images+"/"+filename)
        objs = showimg(coco, dataset, img, classes, classes_ids, show=False)
        #print(objs)
        for ack in objs:
            objectname = ack[0]
            x1 = ack[1]
            y1 = ack[2]
            x2 = ack[3]
            y2 = ack[4]
            #print(x1,y1,x2,y2)
            #print(objectname)
            csv_labels.write(
                abspath_img + "," + str(x1) + "," + str(y1) + "," + str(x2) + "," + str(y2) + "," + objectname + "\n")
csv_labels.close() 
