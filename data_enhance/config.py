# -- 仅作数据增强用,已经针对性优化源代码 -- #

import os

class DefaultConfigs(object):
    # 原始图片路径
    raw_images = "./data/img/"
    raw_images = os.path.abspath(raw_images) + '/'
    # 原始csv格式标签
    raw_csv_files = "./data/labels.csv"
    raw_csv_files = os.path.abspath(raw_csv_files)
    # 增强后的图片保存路径
    augmented_images = "./data/img/"
    augmented_images = os.path.abspath(augmented_images) + '/'
    # 增强后的csv格式的标注文件
    augmented_csv_file = "./data"
    augmented_csv_file = os.path.abspath(augmented_csv_file) + '/labels.csv'
    # 默认图片格式
    image_format = "jpg"                                                    
config = DefaultConfigs() 
