import faiss
import numpy as np
import torch
import open_clip
from fastapi import FastAPI, File, UploadFile,Form
from pydantic import BaseModel
from PIL import Image
from googletrans import Translator
from fastapi.middleware.cors import CORSMiddleware

import os
from FILE import *
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14',
    pretrained='openai'
)
tokenizer = open_clip.get_tokenizer('ViT-L-14')
model = model.to(device)
IMAGE_INDEX_PATH = "imgPath/folder_index.json"
IMAGE_PATH = "imgPath/image_paths.npy"
INDEX_PATH = "index/faiss_index-001.idx"
index = faiss.read_index(f"{INDEX_PATH}")
image_index = FileToJson(IMAGE_INDEX_PATH)
image_paths = np.load(IMAGE_PATH, allow_pickle=True)
image_paths = [
    os.path.join(os.path.basename(os.path.dirname(p)), os.path.basename(p))
    for p in image_paths
]
print(index.ntotal)
print(len(image_paths))
sorted_path = sorted(image_paths)
translator = Translator()
class TextQuery(BaseModel):
    query: str
    top_k: int

def translate_if_vietnamese(text):
    # dịch sang tiếng Anh nếu query là tiếng Việt
    det = translator.detect(text)
    if det.lang == "vi":
        translated = translator.translate(text, src="vi", dest="en").text
        print(f"🌐 Query dich từ VI → EN: {translated}")
        return translated
    else:
        print(f"🌐 Query giu nguyen (EN): {text}")
        return text

    
# ======================
# 5. Hàm tìm kiếm
# ======================
def search_by_text(query, top_k=10):
    # dịch nếu cần
    query_en = translate_if_vietnamese(query)

    text_tokens = tokenizer([query_en]).to(device)
    with torch.no_grad():
      text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # search trên FAISS
    query_vec = text_features.cpu().numpy().astype("float32")
    D, I = index.search(query_vec, top_k)

    # trả kết quả
    #results = [(image_paths[i], float(D[0][k])) for k,i in enumerate(I[0])]
    results = [
    (image_paths[i], float(D[0][k]))
    for k, i in enumerate(I[0])
    if i < len(image_paths)
    ]
    results = [
    (image_paths[i].replace("\\", "/"), float(D[0][k]))
    for k, i in enumerate(I[0])
    if i < len(image_paths)
    ]
    return results

def search_by_image(image_path, top_k=10):
    # mở ảnh query
    img = Image.open(image_path).convert("RGB")

    # encode bằng CLIP
    image_input = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    # search trên FAISS
    query_vec = image_features.cpu().numpy().astype("float32")
    D, I = index.search(query_vec, top_k)

    # trả kết quả
    #results = [(image_paths[i], float(D[0][k])) for k,i in enumerate(I[0])]
    results = [
    (image_paths[i], float(D[0][k]))
    for k, i in enumerate(I[0])
    if i < len(image_paths)
    ]
    results = [
    (image_paths[i].replace("\\", "/"), float(D[0][k]))
    for k, i in enumerate(I[0])
    if i < len(image_paths)
    ]
    return results

from typing import List

def binary_search(arr: List[str], target: str) -> int:
    """Tìm kiếm nhị phân một chuỗi trong list đã sắp xếp.
    Trả về index nếu tìm thấy, -1 nếu không có.
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:  # so sánh theo thứ tự từ điển
            left = mid + 1
        else:
            right = mid - 1

    return -1
def search_neighbors(query_image, window=15):
    if(True):
        i = binary_search(sorted_path,query_image)
        if(i==-1):
            return []
        start_index = max(0, i - window)
        end_index = min(len(image_paths), i + window + 1)

        return sorted_path[start_index:end_index]

#query = input("Input Query ")
#results = search_by_text(query)
#for i, (path, _) in enumerate(results):
    #GetFileFromID(image_index[path],f"image-{str(i)}.jpg")
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # Cho phép tất cả domain (hoặc ghi cụ thể ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],            # Cho phép tất cả phương thức: GET, POST, PUT, DELETE...
    allow_headers=["*"],            # Cho phép tất cả header
)
@app.post("/text")
def GetTextQuery(query:TextQuery):
    results = search_by_text(query.query,query.top_k)
    response = []
    for i, (path, _) in enumerate(results):
        response.append([path,image_index[path]])
    return {"List_Image":response}
@app.post("/Image")
async def upload_file(
    file: UploadFile = File(...),
    top_k: int = Form(...)
):
    contents = await file.read()
    with open("image_tmp/img.jpg", "wb") as f:
        f.write(contents)
    print(top_k)
    results = search_by_image(os.getcwd()+"/image_tmp/img.jpg",top_k)
    response = []
    for i, (path, _) in enumerate(results):
        response.append([path,image_index[path]])
    return {"List_Image":response}
@app.get("/Image/neighbors/{query}")
async def searchneighbors(query:str):
    print("Query is ",query)
    L = query.split("-")
    query = L[0]+"/"+L[1]
    Paths = search_neighbors(query)
    return {"List_Image": [
        [i,image_index[i]] for i in Paths
    ]}

