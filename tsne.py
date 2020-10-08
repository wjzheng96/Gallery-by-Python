# -*- coding: UTF-8 -*
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import gensim
import matplotlib as mpl

# 若要显示中文字体则取消注释下面两行
# mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定中文字体
# mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题


def plot_with_labels(low_dim_embs, labels, filename):   # 绘制词向量图
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    print('绘制词向量中......')
    plt.figure(figsize=(10, 10))  # in inches
    mark = ['orange', 'yellow', 'lightskyblue', 'violet', 'purple', 'black', 'green']
    color = []
    l = defaultdict(list)
    for i, label in enumerate(labels):
        # color.append(mark[label])
        l[label].append(i)
        # x, y = low_dim_embs[i, :]
        # plt.scatter(x, y, mark[label])	# 画点，对应low_dim_embs中每个词向量
        #plt.annotate(label,	# 显示每个点对应哪个单词
                     #xy=(x, y),
                     #xytext=(5, 2),
                     #textcoords='offset points',
                     #ha='right',
                     #va='bottom')
    label_len = len(l)
    for i in range(label_len):
        plt.scatter(low_dim_embs[l[i], 0], low_dim_embs[l[i], 1], c=mark[i])
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.savefig(filename)
    plt.show()

def load_vec(ndarr, threshold=0, dtype='float'):	# 读取txt文件格式的word2vec词向量
    print('读取词向量文件中......')
    #header = file.readline().split(' ')
    # header=[1044, 16]
    # count = int(header[0]) if threshold <= 0 else min(threshold, int(header[0]))
    # dim = int(header[1])
    count = ndarr.shape[0]
    dim = ndarr.shape[1]
    words = []
    matrix = np.empty((count, dim), dtype=dtype)
    for i in range(count):
        words.append(i)
        matrix[i] = ndarr[i]  # np.fromstring(vec, sep=' ', dtype=dtype)
    print("#####")
    return (words, matrix)

def read_label(input):

    # f = open(inputFileName, 'r')
    # lines = f.readlines()
    # f.close()
    # N = len(lines)
    # y = np.zeros(N, dtype=int)
    # for line in lines:
    #     l = line.strip('\n\r').split(' ')
    #     y[int(l[0])] = int(l[1])
    input = np.load(input)
    N = input.shape[0]
    y = np.zeros(N,dtype=int)
    i = 0
    for i in range(N):
        y[i] = int(input[i])

    return input

if __name__ == '__main__':
    try:
        # 若要载入txt文件格式的word2vec词向量，则执行下面这两行
        # method = 'dblp_HANE.txt'
        # method = 'DBLP_mile.txt'
        input = 'embeddings_cora_nc.npy'
        # w2v_txt_file = open(method, 'r')	# txt格式的词向量文件
        # words, vectors = load_txt_vec(w2v_txt_file)	# 载入txt文件格式的word2vec词向量
        input = np.load(input)
        words, vectors = load_vec(input)
        # tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=500, method='exact')
        tsne = TSNE(n_components=2, init='pca')
        # pca = PCA(n_components=2)  # 加载PCA算法，设置降维后主成分数目为2
        # low_dim_embs = pca.fit_transform(vectors)
        #plot_only = 2707	# 限定画点（词向量）的个数，只画词向量文件的前plot_only个词向量
        low_dim_embs = tsne.fit_transform(vectors) # 需要显示的词向量，一般比原有词向量文件中的词向量个数少，不然点太多，显示效果不好
        np.save('fig_stne/' + 'cora_tsne.npy', low_dim_embs)
        labels=read_label("/home/devine/Documents/hgwnn/cora_label.npy")

        # low_dim_embs = np.load('node_ax.npy')
        #labels = [words[i] for i in range(plot_only)] # 要显示的点对应的单词列表
        print('paint picture')
        plot_with_labels(low_dim_embs, labels, 'fig_stne/' + 'cora_nc_HGWNN_1' + '.png')

    except ImportError as ex:
        print('Please install gensim, sklearn, numpy, matplotlib, and scipy to show embeddings.')
        print(ex)

