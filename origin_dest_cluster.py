import time
from math import sin,radians,cos,asin,sqrt
import random
import folium
import skmob
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
import pandas as pd
import matplotlib
import webbrowser
from skmob.preprocessing import detection
import os
DataPath="D:\\Geolife"
def haversine(lonlat1, lonlat2):
    lat1, lon1 = lonlat1
    lat2, lon2 = lonlat2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

def dist(v1,v2):
    v1lat1=v1[0]
    v1lon1=v1[1]
    v1lat2 = v1[2]
    v1lon2 = v1[3]

    v2lat1 = v2[0]
    v2lon1 = v2[1]
    v2lat2 = v2[2]
    v2lon2 = v2[3]
    dist=haversine([v1lat1,v1lon1],[v2lat1,v2lon1])+haversine([v1lat2,v1lon2],[v2lat2,v2lon2])
    dist=dist/2
    #dist=sqrt((v1x1-v2x1)**2+(v1y1-v2y1)**2)+sqrt((v1x2-v2x2)**2+(v1y2-v2y2)**2)
    #print("V1:",v1,"V2:",v2)
    #print(dist)
    return dist

def spdist(sp1,sp2):
    lat1=sp1[1]
    lng1=sp1[2]
    lat2=sp2[1]
    lng2=sp2[2]
    dist=haversine([lat1,lng1],[lat2,lng2])
    return dist

def read_origin_dest_coor(uid,coorlist):
    csvfile=DataPath+"\\"+str(uid).zfill(3)+".csv"
    print("Reading",csvfile)
    df = pd.read_csv(csvfile, encoding='utf-8', header=0)
    #print(df)
    IDset = set(df["tid"].tolist())
    #print(IDset)
    #coorlist=[]
    for tid in IDset:
        temp_df = df[df['tid'] == tid]
        vec=[temp_df.iloc[0]['lat'],temp_df.iloc[0]['lng'],temp_df.iloc[-1]['lat'],temp_df.iloc[-1]['lng'],uid,tid]
        #print(vec)
        coorlist.append(vec)
    return coorlist


def calculate_usernumber():
    path="cluster"
    count=0
    allfile=os.listdir(path)
    for file in allfile:
        one_cluster = np.loadtxt(open(path+"\\"+file, "rb"), delimiter=",", skiprows=0)
        u=np.unique(one_cluster[:,4])
        if len(u)>=10:
            count+=1
            print(file,len(u),u)
        #sumup here


def showclustertraj():
    csvs = os.listdir("cluster")
    for csvf in csvs:
        print(csvf)
        one_cluster = np.loadtxt(open("cluster\\"+csvf, "rb"), delimiter=",", skiprows=0)
        u = np.unique(one_cluster[:, 4])
        uidlist=[]
        user_tids={}
        user_color={}
        map_f = folium.Map(zoom_start=12, tiles='https://mt.google.com/vt/lyrs=h&x={x}&y={y}&z={z}', attr='default')
        for uid in u:
            tids = one_cluster[one_cluster[:, 4] == uid][:,5]
            tids=tids.tolist()
            tids=list(map(int,tids))
            uid=int(uid)
            user_tids[uid]=tids
            uidlist.append(uid)
            user_color[uid]=randomcolor()
        # #get source data for every user and tid
        count=0
        for uid in uidlist:
            csvfile = DataPath + "\\" + str(uid).zfill(3) + ".csv"
            print(csvfile)
            df = pd.read_csv(csvfile, encoding='utf-8', header=0)
        #     print("\033[1;32m ***************\033[0m",uid)
            for tid in user_tids[uid]:
                print(tid)
                traj=df[df['tid'] == tid]
                tdf = skmob.TrajDataFrame(traj, latitude='lat', longitude='lng', user_id='uid', datetime='timestamp')
                stdf = detection.stops(tdf, stop_radius_factor=0.5, minutes_for_a_stop=20.0, spatial_radius_km=0.2,leaving_time=True)
                tdf.plot_trajectory(map_f=map_f, max_users=1, start_end_markers=True,hex_color=user_color[uid])
                stdf.plot_stops(max_users=1, map_f=map_f,hex_color=user_color[uid])
            count+=1
            # if count>=3:
            #     break
        map_f.save("clustermap\\"+"cmap"+csvf.replace(".csv","")+".html")
    webbrowser.open("clustertraj.html")


##detect stop points and clustering
def detectSP():
    one_cluster = np.loadtxt(open("cluster\\000.csv", "rb"), delimiter=",", skiprows=0)
    u = np.unique(one_cluster[:, 4])
    uidlist = []
    user_tids = {}
    for uid in u:
        tids = one_cluster[one_cluster[:, 4] == uid][:, 5]
        tids = tids.tolist()
        tids = list(map(int, tids))
        uid = int(uid)
        user_tids[uid] = tids
        uidlist.append(uid)

    flag=True
    print(uidlist)
    for uid in uidlist:
        #read source file
        csvfile = DataPath + "\\" + str(uid).zfill(3) + ".csv"
        print(csvfile)
        df = pd.read_csv(csvfile, encoding='utf-8', header=0)
        for tid in user_tids[uid]:
            traj = df[df['tid'] == tid]
            tdf = skmob.TrajDataFrame(traj, latitude='lat', longitude='lng', user_id='uid', datetime='timestamp')
            stdf = detection.stops(tdf, stop_radius_factor=0.5, minutes_for_a_stop=20.0, spatial_radius_km=0.2,
                                   leaving_time=True)
            #one traj
            print(uid, tid)
            #stdf['datetime']=pd.to_datetime(stdf['datetime'])
            #stdf['datetime'] = pd.to_datetime(stdf['datetime'])
            #stdf['leaving_datetime'] = pd.to_datetime(stdf['leaving_datetime'])
            #stdf['leaving_datetime'] = pd.to_numeric(stdf['leaving_datetime'])
            ######
            #print(stdf)
            if flag:
                array=stdf
                flag=False
            else:
                array=array.append(stdf)
    ##get all SP positions and perform dbscan
    ##array=merge(stdfs)
    array['datetime'] = pd.to_numeric(array['datetime'])
    array['leaving_datetime'] = pd.to_datetime(array['leaving_datetime']).astype(np.int64)
    print(array)
    data=np.array(array)
    dbscan = DBSCAN(eps=0.1, min_samples=1, metric=spdist).fit(data)
    labels=dbscan.labels_
    print(labels)
    print(type(labels))
    raito = len(labels[labels[:] == -1]) / len(labels)
    print('噪声比:', format(raito, '.2%'))
    ###give a label to each location
    ##create a new dataframe based on array , add column 'geoid'
    array['geoid']=labels
    array['leaving_datetime'] = pd.to_datetime(array['leaving_datetime'])
    array['datetime']=pd.to_datetime(array['datetime'])

    print(array)
    array.to_csv("SPID.csv",index=False)

def clusterresult():
    color=randomcolor()
    incidents = folium.map.FeatureGroup()
    csvs=os.listdir("cluster")
    for csvf in csvs:
        print(csvf)
        one_cluster = np.loadtxt(open("cluster\\"+csvf, "rb"), delimiter=",", skiprows=0)
        for vec in one_cluster:
            color = color
            incidents.add_child(
                folium.CircleMarker(
                        (float(vec[0]), float(vec[1])),
                        radius=7,  # define how big you want the circle markers to be
                        color=color,
                        fill=True,
                        fill_opacity=0.3,
                    )
            )
            incidents.add_child(
                        folium.CircleMarker(
                            (float(vec[2]), float(vec[3])),
                            radius=7,  # define how big you want the circle markers to be
                            color=color,
                            fill=True,
                            fill_opacity=1,
                        )
                    )
        map = folium.Map(zoom_start=12, tiles='https://mt.google.com/vt/lyrs=h&x={x}&y={y}&z={z}', attr='default')
        map.add_child(incidents)
        file_path = "cmap"+str(csvf.replace(".csv",""))+".html"
        map.save(file_path)
    #webbrowser.open(file_path)
if __name__=="__main__":

    #showclustertrajshowclustertraj()
    detectSP()
    #clusterresult()
    # calculate_usernumber()

    # uid_list=[i for i in range(0,182)]
    # uid_color={}
    # data=[]
    # for u in uid_list:
    #     data=read_origin_dest_coor(u,data)
    #     uid_color[u]=randomcolor()
    # #print(data)
    # starttime=time.time()
    # data = np.array(data)
    # #kmeans = KMeans(n_clusters=2, random_state=0).fit(data_for_kmeans)
    # dbscan = DBSCAN(eps=0.1, min_samples=3,metric=dist).fit(data)
    # #print(kmeans.labels_)
    # labels=dbscan.labels_
    # print(labels)
    # raito = len(labels[labels[:] == -1]) / len(labels)
    # print('噪声比:', format(raito, '.2%'))
    # incidents = folium.map.FeatureGroup()
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目
    # endtime = time.time()
    # print("Cluster sum time:",endtime-starttime)
    #
    # ##save all clusater data
    # for i in range(n_clusters_):
    #     one_cluster = data[labels == i]
    #     #print(type(one_cluster))
    #     np.savetxt("cluster/"+str(i).zfill(3)+".csv", one_cluster, delimiter=',')
    #     rcoloor = randomcolor()
    #     for vec in one_cluster:
    #         #color = uid_color[vec[4]]
    #         color=rcoloor
    #         incidents.add_child(
    #             folium.CircleMarker(
    #                 (float(vec[0]), float(vec[1])),
    #                 radius=7,  # define how big you want the circle markers to be
    #                 color=color,
    #                 fill=True,
    #                 fill_opacity=0.2,
    #             )
    #         )
    #         incidents.add_child(
    #             folium.CircleMarker(
    #                 (float(vec[2]), float(vec[3])),
    #                 radius=7,  # define how big you want the circle markers to be
    #                 color=color,
    #                 fill=True,
    #                 fill_opacity=1,
    #             )
    #         )
    #         # incidents.add_child(
    #         #     folium.PolyLine([[vec[0],vec[1]],[vec[2],vec[3]]],color=color)
    #         # )
    # map = folium.Map(zoom_start=12, tiles='https://mt.google.com/vt/lyrs=h&x={x}&y={y}&z={z}', attr='default')
    # map.add_child(incidents)
    # file_path = "ori_dest_map.html"
    # map.save(file_path)
    # webbrowser.open(file_path)