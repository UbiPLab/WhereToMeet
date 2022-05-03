import requests
import time
if __name__=="__main__":
    url="https://restapi.amap.com/v3/direction/walking"
    access_key="1df8ce474c275564a23a5ab179722b19"
    params = {"key": access_key, "origin": "116.324514, 39.998875", "destination": "116.32453899999999, 39.998784"}
    start=time.time()
    r = requests.get(url, params=params)
    end=time.time()
    print((end-start)*1000)