import time as time
import json
import numpy as np
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import sys
import cv2


dataDir = './coco'
dataType = 'train2019'
annFile = './coco/annotations/instances_train2019.json'.format(dataDir,dataType)
catalog = 'bird'

num = 0o0015
point_size = 1
point_color = (0, 0, 255) # BGR
line_color = (255, 0, 0)
thickness = 2 #  0 、4、8
lineThickness = 1


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')
class COCO:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()
    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats
    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']

        else:
            cats = self.dataset['categories']
            print(' ')
            print('keypoints的cat就只有人1种')
            print(cats)
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name'] in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id'] in catIds]
            print(cats)
        ids = [cat['id'] for cat in cats]
        return ids
    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]
    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)
    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]
    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            #根据imgIds找到所有的ann
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            #通过各类条件如catIds对anns进行筛选
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:

            ids = [ann['id'] for ann in anns]
        return ids
    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]
    def showAnns(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
            datasetType = 'instances'
        elif 'caption' in anns[0]:
            datasetType = 'captions'
        else:
            raise Exception('datasetType not supported')
        if datasetType == 'instances':
            ax = plt.gca()
            ax.set_autoscale_on(False)
            polygons = []
            color = []
            for ann in anns:
                c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
                if 'segmentation' in ann:
                    if type(ann['segmentation']) == list:
                        # polygon
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape((int(len(seg)/2), 2))
                            polygons.append(Polygon(poly))
                            color.append(c)
                    else:
                        # mask
                        t = self.imgs[ann['image_id']]
                        if type(ann['segmentation']['counts']) == list:
                            rle = maskUtils.frPyObjects([ann['segmentation']], t['height'], t['width'])
                        else:
                            rle = [ann['segmentation']]
                        m = maskUtils.decode(rle)
                        img = np.ones( (m.shape[0], m.shape[1], 3) )
                        if ann['iscrowd'] == 1:
                            color_mask = np.array([2.0,166.0,101.0])/255
                        if ann['iscrowd'] == 0:
                            color_mask = np.random.random((1, 3)).tolist()[0]
                        for i in range(3):
                            img[:,:,i] = color_mask[i]
                        ax.imshow(np.dstack( (img, m*0.5) ))
                if 'keypoints' in ann and type(ann['keypoints']) == list:
                    # turn skeleton into zero-based index
                    sks = np.array(self.loadCats(ann['category_id'])[0]['skeleton'])-1
                    kp = np.array(ann['keypoints'])
                    x = kp[0::3]
                    y = kp[1::3]
                    v = kp[2::3]
                    for sk in sks:
                        if np.all(v[sk]>0):
                            plt.plot(x[sk],y[sk], linewidth=3, color=c)
                    plt.plot(x[v>0], y[v>0],'o',markersize=8, markerfacecolor=c, markeredgecolor='k',markeredgewidth=2)
                    plt.plot(x[v>1], y[v>1],'o',markersize=8, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2)
            p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
            ax.add_collection(p)
        elif datasetType == 'captions':
            for ann in anns:
                print(ann['caption'])

coco_kps=COCO(annFile)
catIds = coco_kps.getCatIds(catNms=[catalog])
print('得到种类的Id')
print(catIds)
print(' ')
imgIds = coco_kps.getImgIds(imgIds = [num])
print('指定图片Id后得到图片的Id')
print(imgIds)
print(' ')

img = coco_kps.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
print('指定图片Id后得到图片信息')
print(img)
print(' ')

annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
print('根据图片Id后得到标注Id')
print(annIds)
print(' ')

anns = coco_kps.loadAnns(annIds)
print('根据标注Id后得到标注信息')
print(anns)
print(' ')

# coco_kps.showAnns(anns)

fig, ax = plt.subplots(figsize=(img['width']/100,img['height']/100))
ax = plt.gca()
ax.set_autoscale_on(False)
polygons = []
color = []
ann = anns[0]
c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
if 'segmentation' in ann:
    if type(ann['segmentation']) == list:
        # polygon
        for seg in ann['segmentation']:
            poly = np.array(seg).reshape((int(len(seg) / 2), 2))
            polygons.append(Polygon(poly))
            color.append(c)

sks = np.array(coco_kps.loadCats(ann['category_id'])[0]['skeleton']) - 1
print('sks')
print(sks)
print(' ')

kp = np.array(ann['keypoints'])
x = kp[0::3]
y = kp[1::3]
v = kp[2::3]
print(kp)
print('x,y,v')
print(x)
print(y)
print(v)
print(' ')
# y = img['height']-y #coco的图片坐标原点在左上。plt的原点在左下。


print(x[[5,6]])
print(v[[0,1]]>0)
print(np.all(v[[5,6]]>0))
image = cv2.imread(dataDir + '/' + dataType + '/' + img['file_name'])

# cv2.imshow('1', image)
# cv2.waitKey()


for sk in sks:
    if np.all(v[sk] > 0):#相互连接的两个点的可见性都是否大于0,都大于0的话就画出连线
        cv2.circle(image, [int(x[sk][0]), int(y[sk][0])], point_size, point_color, thickness)
        cv2.circle(image, [int(x[sk][1]), int(y[sk][1])], point_size, point_color, thickness)
        cv2.line(image, [int(x[sk][0]), int(y[sk][0])], [int(x[sk][1]), int(y[sk][1])], line_color, lineThickness)


# plot(x[v == 2], y[v == 2])
cv2.imshow('2', image)
cv2.waitKey()
cv2.imwrite('1.jpg', image)