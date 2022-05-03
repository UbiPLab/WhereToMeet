import pandas
import folium
import os
import linecache
import re
from sklearn.cluster import DBSCAN
from math import sin,radians,cos,asin,sqrt
import numpy as np
import random
import webbrowser

def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

def haversine(lonlat1, lonlat2):
    lat1, lon1 = lonlat1
    lat2, lon2 = lonlat2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r#根据经纬度的差值计算两点间距离，返回单位为公里


def judge_distance(tra_dict):
    start_list=[]
    end_list=[]
    for date in tra_dict:
        print(date)
        print(tra_dict[date])
        start_lat_lng=tra_dict[date][0]
        lat=start_lat_lng.split(",")[0]
        lng=start_lat_lng.split(",")[1]
        start_list.append([float(lat),float(lng)])
        end_lat_lng=tra_dict[date][1]
        lat=end_lat_lng.split(",")[0]
        lng=end_lat_lng.split(",")[1]
        end_list.append([float(lat),float(lng)])
    #do cluster at start points
    X = np.array(start_list)
    dbscan = DBSCAN(eps=0.05, min_samples=3, metric=haversine).fit(start_list)
    labels = dbscan.labels_  #和X同一个维度，labels对应索引序号的值 为她所在簇的序号。若簇编号为-1，表示为噪声
    print('每个样本的簇标号:')
    print(labels)
    raito = len(labels[labels[:] == -1]) / len(labels)  #计算噪声点个数占总数的比例
    print('噪声比:', format(raito, '.2%'))
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目
    print('分簇的数目: %d' % n_clusters_)
    incidents = folium.map.FeatureGroup()
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
                    fill_opacity=0.4
                )
            )
    #print(one_cluster)
    #plt.plot(one_cluster[:,0],one_cluster[:,1],'o')


#plt.show()
    latitude = lat
    longitude = long
    map = folium.Map(location=[latitude, longitude], zoom_start=12)
    map.add_child(incidents)
    file_path = "map.html"
    map.save(file_path)
    webbrowser.open(file_path)


def read_plts(path):
    users=os.listdir(path)
    dates_list=[]
    for user in users:
        user_path=path+"\\"+user+"\\Trajectory"
        date_files=os.listdir(user_path)
        user_record_dict={}
        for date_file in date_files:
            date=date_file.replace(".plt","")#file name
            firstrecord=linecache.getline(user_path+"\\"+date_file,7)
            lastrecord=linecache.getline(user_path+"\\"+date_file,len(linecache.getlines(user_path+"\\"+date_file)))
            try:
                first_lat_lng=re.search("^-?[1-9][0-9]*\.?[0-9]+,-?[1-9][0-9]*\.?[0-9]+",firstrecord).group()
                last_lat_lng=re.search("^-?[1-9][0-9]*\.?[0-9]+,-?[1-9][0-9]*\.?[0-9]+",lastrecord).group()
                #print(first_lat_lng,last_lat_lng)
            except:
                print("\033[31mfuck"+user,date+"\033[0m")

            user_record_dict[date]=[first_lat_lng,last_lat_lng]
            #print(user_record_dict)
        dates_list.append(user_record_dict)
    return dates_list

if __name__=="__main__":
    path="H:\\Geolife Trajectories 1.3\\Geolife Trajectories 1.3\\Data"
    data_list=read_plts(path)
    for item in data_list:#item is a dict
        #print(item)
        #print("###")
        judge_distance(item)
        break