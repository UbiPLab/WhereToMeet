import numpy as np
import pandas as pd
import time
from math import sin,radians,cos,asin,sqrt
from pyemd import emd
from pyemd.emd import euclidean_pairwise_distance_matrix, emd_samples


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

def distmatrix():
    df = pd.read_csv("SPID.csv", encoding='utf-8', header=0)
    GeoIDset = set(df['geoid'].tolist())
    # print(GeoIDset)
    idpos = {}
    for geoid in GeoIDset:
        pdf = df[df['geoid'] == geoid]
        lats = pdf['lat']
        lons = pdf['lng']
        point = (lats.mean(), lons.mean())
        idpos[geoid] = point
        # centre point
    matrix = pd.DataFrame(columns=GeoIDset, index=GeoIDset, dtype=np.float)
    matrix = matrix.fillna(0)
    for i in GeoIDset:
        for j in GeoIDset:
            dist = haversine(idpos[i], idpos[j])
            matrix[i][j] = dist
            # print(dist)
    return matrix


def GeoProbVec(user_data,length):
    geoid=user_data['geoid']
    totalcounts=len(geoid)
    vec=np.zeros(length)
    freq=geoid.value_counts()/totalcounts
    for i in freq.index:
        # print(i)
        vec[i]=freq[freq.index==i]
    # print(vec)
    return vec

def Prob(user_data,GeoIDset,allGeoID=True):
    #for each tid
    #print(GeoIDset)
    if not allGeoID:
        GeoIDset=set(user_data['geoid'].tolist())
    #print(GeoIDset)
    matrix=pd.DataFrame(columns=GeoIDset,index=GeoIDset)
    matrix=matrix.fillna(0)
    TIDset=set(user_data["tid"].tolist())
    #print(TIDset)
    for tid in TIDset:
        traj=user_data[user_data["tid"]==tid]
        #print(traj)
        geoids=traj['geoid'].tolist()
        for i in range(0,len(geoids)-1):
            #print((geoids[i],geoids[i+1]))
            currentid=geoids[i]
            nextid=geoids[i+1]
            matrix[currentid][nextid]+=1
    #print(matrix)
    #calculate probability
    if not len(GeoIDset)==1:
        for geoid in GeoIDset:
            row=matrix[matrix.index == geoid].squeeze()
            rsum=row.sum()
            row=row/rsum
            ###div 0?
            row=row.to_frame().T
            matrix[matrix.index==geoid]=row

    matrix=matrix.fillna(0)
    # print(matrix)
    return matrix

# def getMarkovSigTest():
#     # 96 locations for 006.csv
#     N=96
#     features1=np.random.dirichlet(np.ones(N-60),size=1)[0]
#     features1=np.append(np.zeros(60),features1)
#     np.random.shuffle(features1)
#     features2=np.random.dirichlet(np.ones(N-60),size=1)[0]
#     features2=np.append(np.zeros(60), features2)
#     np.random.shuffle(features2)
#     print(features1)
#     weights1 = (1.0 / N) * np.ones((N))
#     weights2= (1.0 / N) * np.ones((N))
#     signature1 = (features1, weights1)
#     signature2 = (features2, weights2)
#     P=signature1
#     Q=signature2


if __name__=="__main__":
    # N=96
    # features1 = np.random.dirichlet(np.ones(N - 60), size=1)[0]
    # features1 = np.append(np.zeros(60), features1)
    # np.random.shuffle(features1)
    # features2 = np.random.dirichlet(np.ones(N - 60), size=1)[0]
    # features2 = np.append(np.zeros(60), features2)
    # np.random.shuffle(features2)
    # distance_matrix=np.array(distmatrix())
    # distance_matrix=distance_matrix.copy(order='C')
    # print(features1.flags)
    # print(features2.flags)
    # print(distance_matrix.flags)
    # # print(type(features1),type(features2),type(distance_matrix))
    # s=emd(features1, features2, distance_matrix)
    # print(s)


    df = pd.read_csv("SPID.csv", encoding='utf-8', header=0)
    UIDset = list(set(df["uid"].tolist()))
    UIDset.sort()
    print(len(UIDset))
    print(UIDset)
    GeoIDset = set(df['geoid'].tolist())
    distance_matrix = np.array(distmatrix())
    distance_matrix = distance_matrix.copy(order='C')
    print(distmatrix())
    # user 0
    targetid=0
    user_data0 = df[df['uid'] == targetid]
    m0 = Prob(user_data0, GeoIDset, allGeoID=True)
    vec0=GeoProbVec(user_data0,96)
    # user 0
    Expec={}
    maxe = 0
    #
    print("Calculating")
    UIDset.remove(targetid)
    start=time.time()
    for j in UIDset:

        user_data1 = df[df['uid'] == j]
        m1=Prob(user_data1,GeoIDset,allGeoID=True)
        vec1 = GeoProbVec(user_data1, 96)
        # print(m0)
        # print(m0)
        # print(m1)
        sum=0
        for k in range(0,96):
            features0= np.array(m0[m0.index==k])[0]
            # print(features0)
            features1 = np.array(m1[m1.index==k])[0]
            # print(features1)
            s = emd(features0, features1,distance_matrix)
            sum+=s*vec0[k]*vec1[k]
            # print(m0[m0.index==i])
        maxe=max(maxe,sum)
        Expec[j]=sum


    for j in UIDset:
         sim = 1-Expec[j]/maxe
         print("User 0 and","User",j,"simG:",sim)
         # print(sim)
    end=time.time()
    print("total time:",end-start)