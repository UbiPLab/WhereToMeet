import random
import webbrowser
from math import sin,radians,cos,asin,sqrt
import skmob
import pandas as pd
import folium
import numpy as np
from sklearn.cluster import DBSCAN
import csv
from dateutil.parser import parse

class latlng:
    def __init__(self,lat,lng):
        self.lat=lat
        self.lng=lng

class simple_tra:
    def __init__(self,origin,destination,tid,o_time,d_time):
        self.origin=origin
        self.destination=destination
        self.tid=tid
        self.o_time=o_time
        self.d_time=d_time

    def __str__(self):
        return "Origin:%s,%s Des:%s,%s Tid:%s" % (self.origin.lat,self.origin.lng,self.destination.lat,self.destination.lng,self.tid)

def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

def judge_distance(tra1,tra2):
    return 0

def haversine(lonlat1, lonlat2):
    #print(lonlat1,lonlat2)
    lat1, lon1 = lonlat1
    lat2, lon2 = lonlat2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r#根据经纬度的差值计算两点间距离，返回单位为公里

if __name__=="__main__":
    df = pd.read_csv('000.csv', encoding='utf-8', header=0)
    ls=df.iloc[:, -1]
    max_tid=int(ls.iloc[-1])
    trlist=[]
    incidents = folium.map.FeatureGroup()
    df.rename(columns={'lat':'latitude'}, inplace = True)
    df.rename(columns={'lng': 'longitude'}, inplace=True)
    df.rename(columns={'tid': 'track_id'}, inplace=True)
    df.rename(columns={'timestamp': 'time'}, inplace=True)
    df['id'] = range(len(df))
    df=df.drop(['uid'],axis=1)
    order=["id","latitude","longitude","track_id","time"]
    df=df[order]
    print(df)
    ##get data from dataframe
    start_list = []
    end_list = []
    time_dict={}
    tdf_list=[]
    for i in range(0,max_tid+1):
        temp_df=df[df['track_id']==i]#split dataframe by tid
        tdf_list.append(temp_df)
        o_line=temp_df.iloc[0,:]
        d_line=temp_df.iloc[-1,:]
        # o_latlng=latlng(o_line['lat'],o_line['lng'])
        # d_latlng=latlng(d_line['lat'],d_line['lng'])
        tid=o_line['track_id']
        start_list.append([o_line['latitude'],o_line['longitude']])
        end_list.append([d_line['latitude'], d_line['longitude']])
        time_dict[str(o_line['latitude'])+str(o_line['longitude'])]=o_line['time']
    incidents = folium.map.FeatureGroup()

    ##DBSCAN
    X = np.array(start_list)
    dbscan = DBSCAN(eps=0.05, min_samples=3, metric=haversine).fit(start_list)
    labels = dbscan.labels_  # 和X同一个维度，labels对应索引序号的值 为她所在簇的序号。若簇编号为-1，表示为噪声
    print('每个样本的簇标号:')
    print(labels)
    raito = len(labels[labels[:] == -1]) / len(labels)  # 计算噪声点个数占总数的比例
    print('噪声比:', format(raito, '.2%'))
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目
    print('分簇的数目: %d' % n_clusters_)
    for i in range(n_clusters_):
        print('簇 ', i, '的样本数:')
        one_cluster = X[labels == i]
        print(len(one_cluster))
        rcloor=randomcolor()
        for point in one_cluster:
            lat=point[0]
            long=point[1]
            #ptext=dict[str(lat)+","+str(long)]
            incidents.add_child(
                folium.CircleMarker(
                    (float(lat), float(long)),
                    radius=7, # define how big you want the circle markers to be
                    color=rcloor,
                    fill=True,
                    fill_opacity=0.4,
                    popup=time_dict[str(lat)+str(long)]
                )
            )

    #show
    map = folium.Map(zoom_start=12,tiles='https://mt.google.com/vt/lyrs=h&x={x}&y={y}&z={z}',attr='default')
    map.add_child(incidents)
    file_path = "map.html"
    map.save(file_path)
    webbrowser.open(file_path)