# coding=utf-8
# -- 仅作数据增强用,已经针对性优化源代码 -- #

import os
str=('python3 labelme2coco.py')
print()
print("\033[1;36;40mINFO \033[0m \033[1;33;40m[Step1: recreate coco data] \033[0m")
p=os.system(str)
if (p == 0):
    print()
    print("\033[1;36;40mINFO \033[0m \033[0;32;40m[Step1: recreate coco data success!]\033[0m")
    print()
    print("\033[1;36;40mINFO \033[0m \033[1;33;40m[Step2: translate coco data to csv]\033[0m")
    str=('python3 coco2csv.py')
    p=os.system(str)
    if (p == 0):
        print()
        print("\033[1;36;40mINFO \033[0m \033[0;32;40m[Step2: translate coco data to csv success!]\033[0m")
        print()
        print("\033[1;36;40mINFO \033[0m \033[1;33;40m[Step3: dataset enhanced]\033[0m")
        str=('python3 dataset_enhance.py')
        p=os.system(str)
        if (p == 0):
            print()
            print("\033[1;36;40mINFO \033[0m \033[0;32;40m[Step3: dataset enhanced success!]\033[0m")
            print()
            print("\033[1;36;40mINFO \033[0m \033[1;33;40m[Step4: csv retranslate to coco data]\033[0m")
            str=('python3 csv2coco.py')
            p=os.system(str)
            if (p == 0):
                print()
                print("\033[1;36;40mINFO \033[0m \033[0;32;40m[Step4: csv retranslate to coco data success!]\033[0m")
                print()
                print("\033[1;36;40mINFO \033[0m \033[0;32;40m[All Steps finish successfully!]\033[0m")
            else:
                print()
                print("\033[5;31;40mERROR \033[0m \033[0m \033[1;31;40m[Step4: csv retranslate to coco data fail!]\033[0m")
                print()
                exit(1)
        else:
            print()
            print("\033[5;31;40mERROR \033[0m \033[0m \033[1;31;40m[Step3: dataset enhanced fail!]\033[0m")
            print()
            exit(1)
    else:
        print()
        print("\033[5;31;40mERROR \033[0m \033[1;31;40m[Step2: translate from coco data to csv fail!]\033[0m")
        print()
        exit(1)
else:
    print()
    print("\033[5;31;40mERROR \033[0m \033[1;31;40m[Step1: recreate coco data fail!]\033[0m")
    print()
    exit(1)
