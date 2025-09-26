from __future__ import annotations
import os
import shutil
import requests
import gradio as gr
import aiohttp
import aiofiles
import asyncio
import pandas as pd
import json
mediapath = "media.json"
with open("media.json","r", encoding='utf-8') as f:
    media = json.load(f)    


async def download_and_save(session, file_id,filename, folder="Result"):
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    async with session.get(url) as resp:
        data = await resp.read()

    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{filename}")
    async with aiofiles.open(path, 'wb') as f:
        await f.write(data)
    return path

async def GetListFile(file_ids,filenames,folder = "Result"):
    async with aiohttp.ClientSession() as session:
        tasks = [download_and_save(session,file_ids[i],filenames[i],folder)for i in range(len(file_ids))]
        saved_paths = await asyncio.gather(*tasks)
    results = [(path, os.path.basename(path)) for path in saved_paths]
    return results
 

def clear_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

def GetFileFromID(ID: str, filename: str) -> str:
    url = f"https://drive.google.com/uc?export=download&id={ID}"
    r = requests.get(url)
    save_path = os.path.join("Result", filename)
    with open(save_path, "wb") as f:
        f.write(r.content)
    return save_path

# --------------------------
# Backend call
# --------------------------
def backend_text_search(query: str, top_k: int = 5):
    url = "http://127.0.0.1:8000/text"
    response = requests.post(url, json={"query": query, "top_k": top_k})
    return response.json()

def backend_image_search(filepath, top_k):
    url = "http://127.0.0.1:8000/Image/"
    with open(filepath, "rb") as f:
        files = {"file": f}
        data = {"top_k": top_k}
        res = requests.post(url, files=files, data=data)
    return res.json()
def getlink(query:str):
    L = query.removesuffix(".jpg")
    L = L.split("-")
    videoname = L[0]
    frame = int(L[1])
    video = media[videoname]
    fps = video['fps']
    second = frame//fps
    link = f"{video['watch_url']}&t={second}s"
    return f"""
    <div style = "display: flex;
  justify-content: center; 
  align-items: center;   
  ">
    <button style = "max-width: fit-content; 
                    margin-left: auto; 
                    margin-right: auto;
                    border : 2px solid black;
                    padding: 5px " 
                    onclick="window.open(\'{link}\', \'_blank\')">
                    {videoname} - FPS={fps}
                    </button>
                    </div>
    """

def image_search(filepath, top_k):
    clear_folder("Result")
    R = backend_image_search(filepath, top_k)
    ids = R["List_Image"]
    filenames, fileID, VideoList, FrameID = [], [], [], []
    for i in range(len(ids)):
        name = ids[i][0]
        L = name.split("/")
        name = L[0]+"-"+L[1]
        VideoList.append(L[0])
        FrameID.append(L[1].split(".")[0])
        filenames.append(name)
        fileID.append(ids[i][1])
    
    df = pd.DataFrame({"video_name":VideoList,"Frame_id":FrameID})

    filepath = filepath.split("/")[-1]
    csv_path = f"Csv/{12345}.csv"
  
    WriteCsv(df, csv_path)

    return asyncio.run(GetListFile(fileID,filenames)),csv_path

def WriteCsv(data,filepath):
    os.makedirs("Csv", exist_ok=True)
    data.to_csv(filepath, index=False, encoding="utf-8",header = False)

def backend_search_neighbor(query):
    url = f"http://127.0.0.1:8000/Image/neighbors/{query}"
    rs = requests.get(url)
    return rs.json()

def search_neightbor(query):
    print("Query is ",query)
    clear_folder("Neighbor")
    R = backend_search_neighbor(query)
    ids = R["List_Image"]
    filenames, fileID, VideoList, FrameID = [], [], [], []
    for i in range(len(ids)):
        name = ids[i][0]
        L = name.split("/")
        VideoList.append(L[0])
        FrameID.append(L[1].split(".")[0])
        name = L[0]+"-"+L[1]
        filenames.append(name)
        fileID.append(ids[i][1])
    csv_path = f"Csv/{12345}.csv"
    df = pd.DataFrame({"video_name":VideoList,"Frame_id":FrameID})

    WriteCsv(df, csv_path)
    return asyncio.run(GetListFile(fileID,filenames,"Neighbor")),csv_path

# --------------------------
# Gradio App Logic
# --------------------------
def text_search(query: str, top_k: int):
    if not query:
        return [], None

    clear_folder("Result")
    R = backend_text_search(query, top_k)
    ids = R["List_Image"]
    filenames, fileID, VideoList, FrameID = [], [], [], []

    for i in range(len(ids)):
        name = ids[i][0]
        L = name.split("/")
        name = L[0]+"-"+L[1]
        VideoList.append(L[0])
        FrameID.append(L[1].split(".")[0])
        filenames.append(name)
        fileID.append(ids[i][1])

    df = pd.DataFrame({"video_name":VideoList,"Frame_id":FrameID})
    csv_path = f"Csv/{12345678}.csv"
    WriteCsv(df, csv_path)

    return asyncio.run(GetListFile(fileID,filenames)), csv_path

def search_backend(query_mode, text_query, topk, image_query):
    if query_mode == "Text" and text_query:
        return text_search(text_query, topk)
    elif query_mode == "Image" and image_query is not None:
        return image_search(filepath=image_query, top_k=topk)
    else:
        return [], None

def on_image_select(evt: gr.SelectData):
    a = evt.value 
    return f"{a['caption']}",getlink(a["caption"])

# ---- UI ----
with gr.Blocks(css=".small-btn {width: 120px !important;}") as demo:
    gr.Markdown("## 🔍 Video Retrieval Demo")

    with gr.Row():
        query_mode = gr.Radio(["Text", "Image"], value="Text", label="Choose Query Type", scale=1)
        text_query = gr.Textbox(label="Text Query", placeholder="e.g. 'white car'", scale=3)
        topk = gr.Slider(1, 200, step=1, value=3, label="Top K", scale=1)
        img_query = gr.Image(label="Image Query", type="filepath")

    with gr.Row():
        search_btn = gr.Button("Search", elem_classes="small-btn")

    gallery = gr.Gallery(label="Results", columns=5, object_fit="contain", height="auto", interactive=True)
    
    select_btn = gr.Button("Chọn ảnh đã chọn")
    output = gr.Textbox(label="Kết quả chọn")
    link_btn = gr.HTML("""
    <div style = "display: flex;
  justify-content: center; 
  align-items: center;   
  ">
    <button style = "max-width: fit-content; 
                    margin-left: auto; 
                    margin-right: auto;
                    border : 2px solid black;
                       padding: 5px" >
                    Click to open link 
                    </button>
                    </div>
    """)
    gallery.select(on_image_select, None, [output,link_btn])

    def toggle_inputs(mode):
        return (
            gr.update(visible=(mode == "Text")),
            gr.update(visible=(mode == "Image"))
        )

    query_mode.change(
        fn=toggle_inputs,
        inputs=query_mode,
        outputs=[text_query, img_query],
    )

    # Trả về cả gallery và file CSV
    search_btn.click(
        fn=search_backend,
        inputs=[query_mode, text_query, topk, img_query],
        outputs=[gallery, gr.File(label="Download CSV", interactive=False)],
    )

    gallery_neightbor = gr.Gallery(label="Neighbors", columns=5, object_fit="contain", height="auto", interactive=True)
    select_btn.click(
        fn=search_neightbor,
        inputs=output,
        outputs=[gallery_neightbor, gr.File(label="Download CSV", interactive=False)]
    )
    tmp = gr.HTML()
    link_btn_neightbor = gr.HTML("""
    <div style = "display: flex;
  justify-content: center; 
  align-items: center;   
  ">
    <button style = "max-width: fit-content; 
                    margin-left: auto; 
                    margin-right: auto;
                    border : 2px solid black;
                       padding: 5px" >
                    Click to open link 
                    </button>
                    </div>
    """)
    gallery_neightbor.select(on_image_select,None,[tmp,link_btn_neightbor])

demo.launch()