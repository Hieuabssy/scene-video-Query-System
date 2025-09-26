import os
import json
import requests
#PATH = "imgPath/image_paths.txt"
def FileToList(path):
  T = open(path,'r')
  return [i.replace("\n","") for i in T.readlines()]
#A = FileToList(PATH)
#INDEX = "index/folder_index.json"
def FileToJson(path):
  T = open(path,'r')
  return json.loads(T.read())
#Index_Json = FileToJson(INDEX)
#print(Index_Json["L21_V001/037237.jpg"])
def GetFileFromID(ID,filename):
    url = f"https://drive.google.com/uc?export=download&id={ID}"
    r = requests.get(url)
    with open(f"Result/{filename}", "wb") as f:
        f.write(r.content)