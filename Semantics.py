import json
import math
import time

import requests
import os
import numpy as np
import numpy.matlib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from scipy.linalg import fractional_matrix_power
"""数据读取，预处理，遇到了哪些问题
    空值、空字段、时间字符串
"""

access_key="1df8ce474c275564a23a5ab179722b19"
path="GaoDeData"

def getWalkingDistance(locations,length):
    #distance=np.zeros((length,length))
    distance=np.load("WalkingDistance.npy")
    print(distance)
    url="https://restapi.amap.com/v3/direction/walking"
    for i in range(0,length):
        for j in range(0,length):
            try:
                if i!=j and distance[i][j]<=0.1:
                    params = {"key": access_key, "origin": locations[i], "destination": locations[j]}
                    r = requests.get(url, params=params)
                    js = r.json()
                    route=js.get("route")
                    path=route.get("paths")[0]
                    dis=float(path.get("distance"))


                    print(dis)
                    distance[i][j]=dis
                    time.sleep(0.1)
            except:
                np.save('WalkingDistance', distance)
                print(i,j)
    np.save('WalkingDistance', distance)

def prorating(ratings):
    veclist=[]
    sum=0
    count=0
    valid=0
    for r in ratings:
        r=round(float(r))
        r=int(r)
        if not r==-1:
            sum+=r
            valid+=1
        else:
            r=3
        ratings[count] = r
        count+=1

    for r in ratings:
        vec=np.zeros(5)
        for i in range(0,r):
            vec[i]=1
        veclist.append(vec)
    return np.array(veclist)


def cos_sim(a,b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a, b) / (a_norm * b_norm)
    return cos


def prepro_timelist(timeDlist,typelist,namelist):
    #24h? 12am,pm?
    """[None None None None None list([]) list([])
 '06:30-09:30 11:00-13:00 17:00-19:00' '11:00-13:30 17:00-23:00' None None
 None None None None None None None None None '08:00-21:00' None None None
 None None None None None None None None None None None None '06:30-23:00'
 '10:00-21:00' None None '10:00-19:30' None None list([]) None None None
 None None None None None None None None None None
 '06:30-09:30 11:00-13:00 17:00-19:00' None None None None None None None
 None None '10:00-21:00' None None '11:00-14:00 17:00-20:30' '09:00-21:00'
 '09:30-20:00' None None None '09:30-20:30'
 '06:30-09:30 10:45-13:00 16:45-19:00' None
 '06:30-08:30 11:00-13:00 17:00-19:00' list([]) '10:00-22:00' None]"""
    #split by space
    #re
    list=[]
    count=0
    #此处进行处理，若类型为商务住宅且time值不存在，修改为0:00-7:00 19:00-24:00

    for td in timeDlist:
        if typelist[count]=="商务住宅" and not type(td).__name__ == 'str':
            td="0:00-7:00 19:00-24:00"

        if typelist[count]=="生活服务" and not type(td).__name__ == 'str':
            td="8:30-22:00"

        if typelist[count]=="餐饮服务" and not type(td).__name__ == 'str':
            td="11:00-13:30 17:00-22:00"
        # print(typelist[count])
        # print(namelist[count])
        # print(td)
        init_array=np.zeros(48)
        if type(td).__name__ == 'str':
            groups=td.split(" ")
            for g in groups:
                # print(g)
                pair=g.split("-")
                starttime=pair[0]
                hours, minutes = starttime.split(":")
                hours=int(hours)
                minutes=int(minutes)
                #判断分钟数离00近还是30近

                if minutes-30>0:
                    minutes=30
                #startloca=hours*2+minutes:30?00:1:0
                seg=1 if minutes==30 else 0
                startloc=hours*2+seg
                # print(startloc)
                endtime=pair[1]
                hours, minutes = endtime.split(":")
                hours = int(hours)
                minutes = int(minutes)
                if minutes-30>0:
                    minutes=30
                seg=1 if minutes==30 else 0
                endloc=hours*2+seg-1
                # print(endloc)
                for i in range(startloc,endloc+1):
                    init_array[i]=1

        # print(init_array)
        count+=1
        list.append(init_array)

    list=np.array(list)
    # print(len(list))
    return list


def readingdata():
    #reading from json files,store each location as a dict in a list
    pois=[]
    for jfilename in os.listdir(path):
        with open(path+"\\"+jfilename,encoding="UTF-8") as jfile:
            j=json.load(jfile)
            for poi in j["pois"]:
                pois.append(poi)
    # print(len(pois))
    names=[]
    times=[]
    times2=[]
    ratings=[]
    types=[]
    locations=[]
    for poi in pois:
        names.append(poi["name"])
        biz_ext=poi.get("biz_ext")
        times.append(biz_ext.get("open_time"))
        times2.append(biz_ext.get("open_time2"))
        ratings.append(biz_ext.get("rating"))
        types.append(poi.get("type").split(";")[0])
        locations.append(poi.get("location"))
    names=np.array(names)
    times=np.array(times)
    times2=np.array(times2)
    ratings=np.array(ratings)
    types=np.array(types)
    #print(names)
    # print(times)
    # print(times2)

    # print(types)
    for i in range(0,len(ratings)):
        if type(ratings[i]).__name__ == 'list':
            ratings[i]='-1'


    ratings=prorating(ratings)
    #print(ratings)

    ##Times
    times=prepro_timelist(times,types,names)


    #applying One-hot encoding to types
    typelabel=np.unique(types)
    #One-hot for types
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(types)
    # print(integer_encoded)
    onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    #print(onehot_encoded)
    return (names,ratings,types,onehot_encoded,times,locations)

def downloaddata(gcjcoor):
    #http Get method
    url="https://restapi.amap.com/v3/place/around"
    location=gcjcoor
    radius=500
    params={"key":access_key,"location":location,"radius":radius}
    r=requests.get(url,params=params)
    # 1:key 2.position 3.radius 4.page
    j=r.json()
    # define new path format

    with open("GaoDeData/1.json", 'w') as file_obj:
        json.dump(j, file_obj)
    # Total items count here, 20 items per page
    total=int(j["count"])
    rem=total%20
    access_count=int(total/20)
    if not rem==0:
        access_count+=1
    print(access_count)
    for i in range(2,access_count+1):
        params['page']=i
        r = requests.get(url, params=params)
        j = r.json()
        with open("GaoDeData/"+str(i)+".json", 'w') as file_obj:
            json.dump(j, file_obj)


def GpsToGCJ(gps):
    url="https://restapi.amap.com/v3/assistant/coordinate/convert"
    params = {"locations": gps,"key":access_key,"coordsys":"gps"}
    r=requests.get(url,params=params)
    print(r.url)
    return r.json()["locations"]


if __name__=="__main__":
    # coor=(116.31987600000001, 40.008303999999995)
    # gps=str(coor[0])+","+str(coor[1])
    # gcjcoor=GpsToGCJ(gps)
    # # print(gcjcoor,type(gcjcoor))
    # downloaddata(gcjcoor)
    names,ratings,types,onehot_encoded,times,locations=readingdata()
    # print(names)
    # print(ratings)
    # print(types)
    # print(times)
    length=len(names)
    venues=np.arange(length)
    homes=list(np.where(types=='商务住宅'))
    # print(venues)
    channel={}
    "Ratings(R),Types(T),Open_times(O)"
    channel['R']=  np.zeros((length,length))
    channel['T'] = np.zeros((length, length))
    channel['O'] = np.zeros((length, length))
    # """Sim of Ratings"""
    # print("""Sim of Ratings:""")
    # for v in venues:
    #     if not v==target:
    #         sim=cos_sim(ratings[target],ratings[v])
    #         print(sim)
    #
    # """Sim of Times"""
    # print("""Sim of Times:""")
    # for v in venues:
    #     if not v==target:
    #         sim=cos_sim(times[target],times[v])
    #         print(sim)
    #
    # """Sim of Types"""
    # print("""Sim of Types:""")
    # for v in venues:
    #     if not v == target:
    #         sim = cos_sim(onehot_encoded[target], onehot_encoded[v])
    #         print(sim)
    #
    """Show"""
    for i in range(0,len(locations)):
        print(names[i],types[i],locations[i])



    """Distance Matrix"""
    DR=channel['R']
    for i in venues:
        for j in venues:
            dis=1-cos_sim(ratings[i],ratings[j])
            DR[i][j]=0 if dis<0 else dis
            #浮点数造成数据1-1<0，强制修改为0
    channel['R']=DR
    DRh=DR.mean()
    # print(DR)
    DT=channel['T']
    for i in venues:
        for j in venues:
            dis = 1 - cos_sim(onehot_encoded[i], onehot_encoded[j])
            DT[i][j] = 0 if dis < 0 else dis
    channel['T'] = DT
    DTh=DT.mean()
    # print(DT)
    DO = channel['O']
    for i in venues:
        for j in venues:
            dis = 1 - cos_sim(times[i], times[j])
            DO[i][j] = 0 if dis < 0 else dis
    channel['O'] = DO
    DOh=DO.mean()

    #print(DRh,DTh,DOh)
    #Calculating Distance Matrix D & mean value Dh
    #for two venues Li and Lj
    """Consider that there are N different venues, Distance Matrix should be N*N """
    """目前看到的统一超图框架(Unified Hypergraph Framework)方式有两种：(1)Affinity矩阵 (2)列级联扩展矩阵"""
    """1.融合三个信道的信息生成Affinity Matrix(A)，大小为N*N
       2.基于A以每一个地点为中心计算KNN并生成一条超边
       3.根据2的结果生成超图矩阵H，大小为N*N，权重矩阵W
       4.计算对角节点度矩阵Dv(具有节点v的超边数量)与对角超边度矩阵De(超边e上节点的数量)
       5.拉普拉斯算子
       6.特征分解，选取前k个最小非零特征值与对应的特征向量
       问题：6中特征分解然后取最小非零值对应特征向量一般是为了和另一个拉普拉斯矩阵计算相似度，维度为n的拉普拉斯矩阵进行分解可以获得n-1个特征值与对应的特征向量，然而实际上只有一个拉普拉斯矩阵，
       如何处理？
            (1)建立两个矩阵(Laplace(Venues-A),Laplace(Venues-B))以计算相似度？
            (2)Rank方法归一化？
       See [36] 5.3 Algorithm 2(Page 84)
    """
    """Affinity Matrix"""

    beta=1/3
    A=np.zeros((length,length))
    for i in range(0,length):
        for j in range(0,length):
                A[i][j]=math.exp(-beta*(DR[i][j]/DRh+DT[i][j]/DTh+DO[i][j]/DOh))


    """Hypergraph Matrix H, H[v][e]=1 if v in e else 0&& Weight Matrix W"""
    """超边度矩阵De，knn过程中选择了k个节点与自身节点生成一条超边，故超边度矩阵应该为 np.eye(len)*(k+1)"""
    """要计算节点度矩阵，可先生成全1向量，因为在生成超边过程中，每个节点都成为过中心，针对以其他节点为中心的过程中生成的idx增加节点度"""
    H = np.eye(length)  # 生成单位对角阵，因为以v为中心计算KNN生成的超边中必包含v
    Wdiag=np.zeros(length)#统计每条超边的权重
    Dvdiag=np.ones(length)
    #Compute KNN
    k=10
    for i in range(0,length):
        rowforV=A[i]
        idx = np.argpartition(rowforV, -k)[-k:]
        if i in idx:
            idx = np.argpartition(rowforV, -k-1)[-k-1:]
            #如果选到了自身，那么再添加一个，以确保选择了k个其他点
            sumA = np.sum(A[i][idx])# 第i条超边的权重
        else:
            sumA = np.sum(A[i][idx]) + A[i][i]  # 第i条超边的权重
        #print(idx)
        #idx = rowforV.argsort()[::-1][0:k]
        #k个最大值的索引，因为A是相似度矩阵

        H[i][idx]=1
        Wdiag[i]=sumA
        Dvdiag[idx]+=1
    W=np.diag(Wdiag)
    Dv=np.diag(Dvdiag)
    De=np.eye(length)*(k+1)
    # np.set_printoptions(threshold=np.inf)
    # print(H)
    # print(W)

    """计算拉普拉斯矩阵"""
    # print(H)
    # print(W)
    # print(De)
    # print(Dv)
    I=np.eye(length)



    mid=np.matmul(fractional_matrix_power(Dv,-0.5), H)#(Dv^-0.5)H
    mid=np.matmul(mid,W)#(Dv^-0.5)HW
    mid=np.matmul(mid,fractional_matrix_power(De,-1))#(Dv^-0.5)HW(De^-1)
    mid=np.matmul(mid,H.T)#(Dv^-0.5)HW(De^-1)(HT)
    mid=np.matmul(mid,fractional_matrix_power(Dv,-0.5))#(Dv^-0.5)HW(De^-1)(HT)((Dv^-0.5))
    Lap=I-mid

    np.set_printoptions(threshold=np.inf)
    #print(Lap)

    """Rank vector"""
    mu = 1200 # 1/(mu+1)=0.1
    fac = 1 / (1 + mu)
    #print(homes)
    timecost=[]
    target=1
    homes[0]=[target]
    for home in homes[0]:
        start=time.clock()
        #print(home)
        y=np.zeros(length)
        y[home]=1
        left=fractional_matrix_power(I-fac*mid,-1)
        f=np.matmul(left,y.T)
        end=time.clock()
        timecost.append((end-start)*1000)

    # print(timecost)
    # print(list(homes[0]))
    # print(f)


    searchnum=5
    # ids = np.argpartition(f, -searchnum)[-searchnum:]#5个最近的
    # print(ids)

    # for id in ids:
    #     print(f[id])
    #     print(types[id])
    #     print(names[id])
    #     print(times[id])
    #     print(ratings[id])


    """Utility"""
    """Generate Walking Distance Matrix By Calling GaoDeMap API"""
    print("******************Utility****************")
    #getWalkingDistance(locations,len(locations))
    distance = np.load("WalkingDistance.npy")

    utilitytime=[]
    wDis=500
    for K in range(1,11):
        start = time.clock()
        hostlist = []
        sorted = np.argsort(f)  # 相似度从低到高
        for host in sorted:
            #print(host)
            if distance[target][host] <= wDis:
                #print(host)
                hostlist.append(host)
            if len(hostlist)==K:
                 break
        end=time.clock()
        print(hostlist)
        utilitytime.append((end-start)*1000)
    print(utilitytime)

    for host in hostlist:
        print(locations[host].split(",")[1]+","+locations[host].split(",")[0])
