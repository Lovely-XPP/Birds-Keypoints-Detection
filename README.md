# Birds-Keypoints-Detection

Origin Picture            |  Detected Picture
:-------------------------:|:-------------------------:
<img src="https://user-images.githubusercontent.com/66028151/164048173-b14e89c5-26ad-4d44-afd3-a56f75a7f5d1.jpg" width="600"> |  <img src="https://user-images.githubusercontent.com/66028151/164048201-51a65d57-0e24-4f64-a3a3-92afb075d26a.jpg" width="600"> 



## Introduction

The repo is a bird-keypoints-detection based on Detectron 2. I use `labelme` as the tool to annotate pictures, which generates `json` files. Then, translate the `json` files to `coco` dataset by `labelme2coco.py`. Therefore, we can register the dataset to Detectron 2 and train the model.

## Downloads
Just clone the repo.

Moreover, if you want the annotated dataset or pre-trained model, you can download them in release.


## Requirement

### **Install Detectron 2**

Follow the official Tutorials : https://detectron2.readthedocs.io/en/latest/tutorials/install.html

### **Other Modules**

````sh
pip3 install labelme cv2 tqdm argparse
````

## Code Files Description

At this part, I will introduce the function of main code files. For how to use the code, you can read the comment on the head of these code files.

### coco_visualize.py

Visualize COCO format data.

### labelme2coco_universal.py

Transform the labelme annotation format to the coco format (suit for any case).

### labelme2coco.py

Transform the labelme annotation format to the coco format (only suit for this repo).

###  train.py

Train models.

### demo.py

Demonstrate the result of input (pics or videos).

### output_data.py

Output the infomation of the detection results, including boxes bounder, scores, classes, keypoints (if existed).

### fig.py

Visualize the Loss - Iter curve by reading the log file generated by Detectron 2. Here is an example:
![Figure_1](https://user-images.githubusercontent.com/66028151/161725566-c061b44e-b91d-4f71-a4e6-fc36e3ccff38.png)


### data_enhance/main.py

Enhance annotation datas, such as scaling, adding noise and so on.

## Credits

* [Detectron 2](https://github.com/facebookresearch/detectron2) from [Facebookresearch](https://github.com/facebookresearch).  
* Wah C., Branson S., Welinder P., Perona P., Belongie S. “The Caltech-UCSD Birds-200-2011 Dataset.” Computation & Neural Systems Technical Report, CNS-TR-2011-001.
