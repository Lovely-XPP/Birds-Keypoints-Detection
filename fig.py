'''
本脚本专门用于读取 detectron2 生成的log文件，并生成可视化的Loss - Iter图
by Lovely_XPP
'''
import numpy as np
import matplotlib.pyplot as plt


# 设定log路径
log_file = "./output/model/log1.txt"

# 设定绘图模式
# mode = 0 代表读取所有loss项，mode = 1 代表读取自定义loss项
mode = 0
# fig_mode = 0代表画所有loss，fig_mode = n 代表绘制第 n 个loss项
fig_mode = 0
# 动图模式，gif = 1 代表开启动图， gif = 0 代表正常画图
gif = 1

# --------------- 若代码运行正常，以下代码无需更改 ---------------- #

# 需要提取的数据，按照提取顺序排列，mode = 0自定义无效
names = ["iter", "total_loss"]

# 读取log文件
file = open(log_file, 'r')
lines = file.readlines()
strs = []
for line in lines:
	tmp = line.split('  ')
	items = []
	if len(tmp) >= 2:
		if "eta" in tmp[1]:
			del tmp[0]
			del tmp[0]
			strs.append(tmp)
lines.clear()

# 将字符读取为数组
all_names = []
tmp = strs[0].copy()
for i in range(len(tmp)):
	del tmp[-1]
	if 'time' in tmp[-1] and not('data_time' in tmp[-1]):
		del tmp[-1]
		break
for item in tmp:
	all_names.append(item.split(':')[0])
if mode == 0:
	names = all_names.copy()

datas = []
for str in strs:
	data_names = names.copy()
	k = len(data_names)
	items = []
	for item in str:
		if data_names[0] in item:
			items.append(float(item.split(' ')[1]))
			del data_names[0]
			k = len(data_names)
			if k == 0:
				break
	datas.append(items)

# plt绘图
datas = np.array(datas)
for k in range(len(strs)*(1 - gif) + gif - 1, len(strs)):
    fig1 = plt.figure(figsize=(12, 6))
    if fig_mode == 0:
        for i in range(1, len(names)):
            plt.plot(datas[0:k, 0], datas[0:k, i], '-o', markersize=2.5)
        plt.xlabel('Iter')
        plt.ylabel('Loss')
        plt.legend(names[1:-1])
    else:
        plt.plot(datas[0:k, 0], datas[0:k, fig_mode], '-o', markersize=2.5)
        plt.xlabel('Iter')
        plt.ylabel('Loss')
        plt.legend(names[fig_mode])
    if (k + 1) >= len(strs):
        plt.show(fig1)
        break
    plt.pause(0.01)
    plt.close(fig1)
