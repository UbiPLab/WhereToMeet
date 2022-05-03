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
from skmob.preprocessing import detection

path="000.csv"

if __name__=="__main__":
    df = pd.read_csv('000.csv', encoding='utf-8', header=0)
    df = df[df['tid'] == 0]
    map_f = folium.Map(zoom_start=12, tiles='https://mt.google.com/vt/lyrs=h&x={x}&y={y}&z={z}', attr='default')
    #print(df)
    IDset = set(df["tid"].tolist())
    print(IDset)
    for tid in IDset:
        #split tra
        tempdf=df[df["tid"]==tid]
        tdf = skmob.TrajDataFrame(tempdf, latitude='lat', longitude='lng', user_id='uid', datetime='timestamp')
        stdf = detection.stops(tdf, stop_radius_factor=0.5, minutes_for_a_stop=20.0, spatial_radius_km=0.2,
                               leaving_time=True)
        print(stdf)
        m = stdf.plot_trajectory(map_f=map_f,max_users=1, start_end_markers=True)
        stdf.plot_stops(max_users=1, map_f=m)
        m.save("stop.html")

    webbrowser.open("stop.html")
    # df=df[df['tid']==0]
    # #print(df)
    # tdf = skmob.TrajDataFrame(df,latitude='lat', longitude='lng', user_id='uid', datetime='timestamp')
    # print(tdf)
    # stdf = detection.stops(tdf, stop_radius_factor=0.5, minutes_for_a_stop=20.0, spatial_radius_km=0.2,
    #                        leaving_time=True)
    # print(stdf)
    # map_f = folium.Map(zoom_start=12, tiles='https://mt.google.com/vt/lyrs=h&x={x}&y={y}&z={z}', attr='default')
