"""
测定聚类质量，计算轮廓系数
"""
from lab2 import calc_line_distance


def calc_representative_trajectory(O):
    """
    :param O: 簇集合
    :return: 每个簇的代表轨迹
    1.计算代表轨迹应该是先求平均向量
    2.把整个簇内的向量按平均向量旋转
    3.使用垂直于x轴的sweep line延x轴平扫，如果与这条直线相交的向量大于等于设置的最小值（MinLns），
      则计算这些相交点的y坐标的平均值，形成点(xi,yi)，重复此过程，直到sweep line的右边再无向量的起始或结束点。
    4.把3步骤生成的点旋转回原来的角度，连接成一条轨迹，这条轨迹就是这个簇的代表轨迹。
    此出只是求了平均坐标作为代表轨迹
    """
    result = []
    for cluster_id in O:
        x1 = 0
        y1 = 0
        x2 = 0
        y2 = 0
        for line in O[cluster_id]:
            x1 += line['line'][0][0]
            x2 += line['line'][1][0]
            y1 += line['line'][0][1]
            y2 += line['line'][1][1]
        result.append(
            [[x1 / len(O[cluster_id]), y1 / len(O[cluster_id])], [x2 / len(O[cluster_id]), y2 / len(O[cluster_id])]])
    return result


def calc_intra_cluster_dissimilarity(cluster):
    """计算簇内不相似度"""
    result = 0
    length = len(cluster)
    for line in cluster:
        distance = 0
        for item in cluster:
            distance += calc_line_distance(line['line'][0], line['line'][1], item['line'][0], item['line'][1])
        result += distance / length
    return result / length


def calc_dissimilarity_between_clusters(sample, clusters):
    """计算簇间不相似度"""
    result = 0
    for line in clusters:
        result += calc_line_distance(sample['line'][0], sample['line'][1], line[0], line[1])
    return result / len(clusters)


def calc_silhouette_coefficient(clusters):
    """计算轮廓系数"""
    ai = 0
    bi = float('inf')
    representative_trajectory = calc_representative_trajectory(clusters)
    for cluster_id in clusters:
        ai += calc_intra_cluster_dissimilarity(clusters[cluster_id])
        for i in range(3):
            tem = calc_dissimilarity_between_clusters(clusters[cluster_id][i], representative_trajectory)
            if bi > tem:
                bi = tem
    ai = ai / len(clusters)
    si = (bi - ai) / max(ai, bi)
    return si
