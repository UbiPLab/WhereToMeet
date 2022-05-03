import matplotlib.pyplot as plt
import random


def random_color():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color


def show(clusters):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()
    fig.suptitle('轨迹聚类')
    for cluster_id in clusters:
        cluster = clusters[cluster_id]
        color = random_color()
        index = -1
        for line in cluster:
            index += 1
            x1 = line['line'][0][0]
            x2 = line['line'][1][0]
            y1 = line['line'][0][1]
            y2 = line['line'][1][1]
            if index == len(cluster) - 1:
                plt.plot([x1, x2], [y1, y2], color=color, label='簇' + str(cluster_id))
            else:
                plt.plot([x1, x2], [y1, y2], color=color)
    plt.legend(loc=3)
    plt.show()
