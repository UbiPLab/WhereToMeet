import numpy as np
import pandas as pd
import time
"""
uid,lat,lng,datetime,tid,leaving_datetime,geoid
0,39.9951995,116.3211215,2009-04-13 05:20:41,55,2009-04-13 05:47:21,0"""

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
    print(matrix)
    return matrix

if __name__=="__main__":
    df = pd.read_csv("SPID.csv", encoding='utf-8', header=0)
    #print(df)
    #for each user
    UIDset = set(df["uid"].tolist())
    print(len(UIDset))
    print(UIDset)
    start=time.time()
    GeoIDset = set(df['geoid'].tolist())

    for uid in UIDset:
        user_data=df[df['uid']==uid]
        #print(uid,user_data)
        Prob(user_data,GeoIDset,allGeoID=False)
    end=time.time()
    print(end-start)