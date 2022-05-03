import math
import pickle
import numpy as np
import pandas as pd
import silhouette_coefficient as SC
from draw import show

"""论文：http://hanj.cs.illinois.edu/pdf/sigmod07_jglee.pdf"""

UNCLASSIFIED = -1  # 未分类的cluster_id
NOISE = 9999  # noise的cluster_id


def calc_distance(p1, p2):
    """计算p1,p2的直线距离"""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calc_straight_line(p1, p2):
    """计算p1 p2的直线方程"""
    try:
        k = (p1[1] - p2[1]) / (p1[0] - p2[0])
    except ZeroDivisionError:
        k = 0
    b = p1[1] - k * p1[0]
    return k, b


def calc_projection(p1, p2, p3):
    """计算 p1 在 线段（p2,p3）上的投影点"""
    k, b = calc_straight_line(p2, p3)
    x = (k * (p1[1] - b) + p1[0]) / (k ** 2 + 1)
    y = k * x + b
    return [x, y]


def calc_point_line_distance(p1, p2, p3):
    """计算点 p1 到 线段（p2,p3）的距离"""
    a = p3[1] - p2[1]
    b = p2[1] - p3[1]
    c = p3[0] * p2[1] - p2[0] * p2[1]
    try:
        distance = (math.fabs(a * p1[0] + b * p1[1] + c)) / (math.pow(a * a + b * b, 0.5))
    except ZeroDivisionError:
        distance = 0
    return distance


def calc_angel_sin(p1, p2, p3, p4):
    """计算两条线段之间夹角的sin"""
    dx1 = p1[0] - p2[0]
    dy1 = p1[1] - p2[1]
    dx2 = p3[0] - p4[0]
    dy2 = p3[1] - p4[1]
    angle1 = math.atan2(dy1, dx1)
    angle2 = math.atan2(dy2, dx2)
    if angle1 * angle2 >= 0:
        angle = abs(angle1 - angle2)
    else:
        angle = abs(angle1) + abs(angle2)
    sin = abs(math.sin(angle))
    return sin


def calc_li_and_lj(p1, p2, p3, p4):
    """
    短的为Lj,长的为Li
    :return: Lj,Li
    """
    if calc_distance(p1, p2) < calc_distance(p3, p4):
        return [p1, p2], [p3, p4]
    else:
        return [p3, p4], [p1, p2]


def calc_d_vertical(p1, p2, p3, p4):
    """计算 d垂直"""
    Lj, Li = calc_li_and_lj(p1, p2, p3, p4)
    L1 = calc_point_line_distance(Lj[0], Li[0], Li[1])
    L2 = calc_point_line_distance(Lj[1], Li[0], Li[1])
    try:
        d = (L1 ** 2 + L2 ** 2) / (L1 + L2)
    except ZeroDivisionError:
        d = 0
    return d


def calc_d_parallel(p1, p2, p3, p4):
    """计算d平行，取Lj在Li上的投影点到两端的距离中短的一部分"""
    Lj, Li = calc_li_and_lj(p1, p2, p3, p4)
    point1 = calc_projection(Lj[0], Li[0], Li[1])
    point2 = calc_projection(Lj[1], Li[0], Li[1])
    d1 = calc_distance(point1, Li[0])
    d2 = calc_distance(point1, Li[1])
    d3 = calc_distance(point2, Li[0])
    d4 = calc_distance(point2, Li[1])
    return min(d1, d2, d3, d4)


def calc_d_sin(p1, p2, p3, p4):
    """Lj * sin"""
    Lj, Li = calc_li_and_lj(p1, p2, p3, p4)
    sin = calc_angel_sin(p1, p2, p3, p4)
    return calc_distance(Lj[0], Lj[1]) * sin


def calc_line_distance(p1, p2, p3, p4):
    """计算两条线段的距离"""
    return calc_d_parallel(p1, p2, p3, p4) + calc_d_vertical(p1, p2, p3, p4) + calc_d_sin(p1, p2, p3, p4)


def calc_mdl_pair(points):
    """计算某个节点的MDL_pair = L(H) + L(D|H)"""
    distance = calc_distance(points[0], points[-1])
    if distance == 0:
        return -float('inf')
    LH = math.log(distance, 2)
    d_vertical = 0  # d垂直
    d_sin = 0
    for i in range(len(points) - 1):
        d_vertical += calc_d_vertical(points[0], points[-1], points[i], points[i + 1])
        d_sin += calc_d_sin(points[0], points[-1], points[i], points[i + 1])
    if d_vertical == 0 or d_sin == 0:
        return -float('inf')
    LDH = math.log(d_vertical, 2) + math.log(d_sin, 2)
    return LDH + LH


def calc_mdl_no_pair(points):
    """计算某个节点的MDL_no_pair = L(H) (轨迹总长度)"""
    total_length = 0
    for i in range(len(points) - 1):
        total_length += calc_distance(points[i], points[i + 1])
    if total_length == 0:
        return -float('inf')
    return math.log(total_length, 2)


def trajectory_division(points):
    """
    :param points: 轨迹点的数组
    :return: 轨迹的特征点
    """
    result = [points[0]]
    start = 0
    length = 1
    while start + length <= len(points):
        current = start + length
        cost_pair = calc_mdl_pair(points[start:current + 1])
        cost_no_pair = calc_mdl_no_pair(points[start:current + 1])
        if cost_pair > cost_no_pair:
            result.append(points[current])
            start = current
            length = 1
        else:
            length += 1
    result.append(points[-1])
    return result


def compute_area(L, line_segment, e):
    """
    :param L: 线段
    :param line_segment: 所有线段集合
    :param e: 半径
    :return: 邻域
    计算线段邻域
    """
    area_L = [L]
    for i in range(len(line_segment)):
        line = line_segment[i].copy()
        distance = calc_line_distance(L['line'][0], L['line'][1], line['line'][0], line['line'][1])
        # print(distance)
        if distance <= e:
            line['index'] = i
            area_L.append(line)
    return area_L


def expand_cluster(Q, cluster_id, e, min_lns, line_segment):
    """
    while len(Q) > 0:
        M = Q.pop(0)
        计算 M 的邻域 area_M
        if len(area_M) >= min_lns:
            for X in area_M:
                if X is unclassified or noise:
                    给 X 分配一个 cluster_id
                if X is unclassified:
                    Q.append(X)
    """
    Q.pop(0)  # 删除L
    while len(Q) > 0:
        M = Q.pop(0)
        # line_segment[M['index']]['cluster_id'] = cluster_id
        area_M = compute_area(M, line_segment, e)
        if len(area_M) >= min_lns:
            for X in area_M:
                if X['cluster_id'] == UNCLASSIFIED or X['cluster_id'] == NOISE:
                    line_segment[X['index']]['cluster_id'] == cluster_id
                # if X['cluster_id'] == UNCLASSIFIED:
                #     Q.append(X)


def assign_cluster_id(area_L, line_segment, cluster_id):
    for line in area_L:
        cur_id = line_segment[line['index']]['cluster_id']
        if cur_id == UNCLASSIFIED or cur_id == NOISE:
            line_segment[line['index']]['cluster_id'] = cluster_id


def line_segment_clustering(line_segments, e, min_lns):
    """
    :param line_segments: 线段集合
    :param e: 邻域半径
    :param min_lns: 邻域需包含的最小线段数
    :return: 簇集合 O
    初始化cluster_id为0
    标记线段集合中的所有线段为未分类
    for L in line_segments:
        if L is unclassified:
            计算 L 的邻域 area_L
            if len(area_L) >= min_lns:
                给 area_L 中的所有线段分配cluster_id
                将 area_L - L 插入队列 Q
                expand_cluster(Q, cluster_id, e, min_lns)
                cluster_id++
            else:
                将 L 标记为 noise
    分配line_segments中所有线段，得到簇集合O
    for C in O:
        if |PTR(C)| < min_lns:
            将C从O中移除
    """
    cluster_id = 0
    O = {}
    for i in range(len(line_segments)):
        print('簇：', cluster_id)
        L = line_segments[i].copy()
        L['index'] = i
        if L['cluster_id'] == UNCLASSIFIED:
            area_L = compute_area(L, line_segments, e)
            if len(area_L) >= min_lns:
                assign_cluster_id(area_L, line_segments, cluster_id)
                expand_cluster(area_L, cluster_id, e, min_lns, line_segments)
                cluster_id += 1
            else:
                line_segments[i]['cluster_id'] = NOISE
    for line in line_segments:
        if line['cluster_id'] not in O:
            O[line['cluster_id']] = [line]
        else:
            O[line['cluster_id']].append(line)
    for cluster_id in O.copy():
        if len(O[cluster_id]) < min_lns:
            del O[cluster_id]
    return O


def preprocessed_data():
    """将数据去重，处理成 {'track_id':[轨迹点]}，并存储"""
    frame = pd.read_csv('666.csv', engine='python')
    data = frame.drop_duplicates(keep='first', subset=['latitude', 'longitude', 'track_id', 'time'])  # 去重
    rows = np.array(data)
    data_grouped = {}
    for line in rows:
        if line[3] not in data_grouped:
            data_grouped[line[3]] = [[line[1], line[2]]]
        else:
            data_grouped[line[3]].append([line[1], line[2]])
    fw = open('data_preprocessed.txt', 'wb')
    pickle.dump(data_grouped, fw, 0)
    fw.close()


def read_data():
    fr = open('data_preprocessed.txt', 'rb')
    return pickle.load(fr)


def calc_line_segment(data):
    """
    :param data: 预处理后的数据
    :return: line_segment [{'track_id':1,line:[point1,point2],'cluster_id':UNCLASSIFIED}]
    将轨迹提取特征点处理成线段集合line_segment
    """
    line_segment = []
    for track_id in data:
        feature_points = trajectory_division(data[track_id])
        for i in range(0, len(feature_points) - 1):
            line = {'track_id': track_id, 'cluster_id': UNCLASSIFIED,
                    'line': [feature_points[i], feature_points[i + 1]]}
            line_segment.append(line)
        # if len(line_segment) > 100:
        #     break
    return line_segment


def main():
    preprocessed_data()
    E = 10
    min_lns = 3
    data = read_data()
    line_segment = calc_line_segment(data)
    #line_segment=data
    print(line_segment)
    clusters = line_segment_clustering(line_segment, E, min_lns)
    if NOISE in clusters:
        del clusters[NOISE]
    si = SC.calc_silhouette_coefficient(clusters)
    print('E: ', E, ' min_lns: ', min_lns)
    print('轮廓系数：', si)
    show(clusters)
    print(clusters)


if __name__ == '__main__':
    main()
