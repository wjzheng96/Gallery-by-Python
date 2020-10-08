#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np

# 创建一个点数为 8 x 6 的窗口, 并设置分辨率为 80像素/每英寸
plt.figure(figsize=(8, 6), dpi=100)

# 再创建一个规格为 1 x 1 的子图
plt.subplot(1, 1, 1)

# 柱子总数
N = 4
# 包含每个柱子对应值的序列
values = (48,94,74,112 )

# 包含每个柱子下标的序列
index = np.arange(N)

# 柱子的宽度
width = 0.36

error_par=dict(elinewidth=2, ecolor='#0000CD', capsize=10)

# 绘制柱状图, 每根柱子的颜色为紫罗兰色
p2 = plt.bar(index, values, width, color="#00FF7F", yerr=[1,2,3,2], error_kw = error_par)

# 设置横轴标签
#plt.xlabel('Months')
# 设置纵轴标签
plt.ylabel('Accuracy (%)', fontdict={'family' : 'Times New Roman', 'size': 20})

# 添加标题
plt.title('(d) NELL', fontdict={'family' : 'SimHei', 'size': 20, 'weight':'light'})

# 添加纵横轴的刻度
plt.xticks(index, ('M-GWNN_WCR', 'M-GWNN_SEM', 'M-GWNN_LM', 'M-GWNN'),fontproperties = 'Times New Roman', size = 15)
y_val = [70.0,72.0,74.0,76.0,78.0,80.0, 82.0]
plt.yticks(np.linspace(0,120 ,7),y_val,fontproperties = 'Times New Roman', size = 17)

# 添加图例
#plt.legend(loc="upper right")

plt.savefig('test.png')
plt.show()

