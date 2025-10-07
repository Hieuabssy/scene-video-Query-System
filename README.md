# Scene Video Query System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Hieuabssy/scene-video-Query-System.svg)](https://github.com/Hieuabssy/scene-video-Query-System/stargazers)

> **A powerful scene-based video retrieval system for searching through ~1,400 Vietnamese news and current affairs videos using AI-powered semantic understanding.**

## 🎯 Overview

The Scene Video Query System is an end-to-end solution that enables semantic search across a large collection of news videos. By breaking videos into meaningful scenes and using state-of-the-art vision transformers, the system allows users to find specific moments within videos using natural language queries or similar images.

### Key Features

- 🎬 **Automatic Scene Detection** - Intelligent segmentation using TransNetV2
- 🖼️ **Smart Keyframe Extraction** - 3 representative frames per scene
- 🧠 **Deep Learning Embeddings** - ViT-L-14 for semantic understanding
- ⚡ **Fast Retrieval** - FAISS-powered vector search (<50ms)
- 🌐 **Multi-Modal Search** - Support for text and image queries
- 📊 **Large Scale** - Handles ~1,400 videos with thousands of scenes

## 🏗️ System Architecture

```
┌─────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────┐    ┌──────────┐
│  Video  │───▶│  TransNetV2  │───▶│   Extract   │───▶│  ViT-L-14│───▶│  FAISS   │
│  Input  │    │Scene Detection│   │ Keyframes   │    │ Features │    │  Index   │
└─────────┘    └──────────────┘    └─────────────┘    └──────────┘    └──────────┘
                                          │                   │              │
                                          ▼                   ▼              ▼
                                    3 frames/scene      Vector embeddings  Retrieval
```

### Pipeline Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Scene Detection** | TransNetV2 | Identifies scene boundaries in videos |
| **Keyframe Extraction** | Custom Logic | Selects 3 representative frames per scene |
| **Feature Extraction** | ViT-L-14 | Generates semantic vector embeddings |
| **Vector Indexing** | FAISS | Builds efficient similarity search index |
| **Retrieval System** | Cosine Similarity | Returns most relevant scenes |


## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Hieuabssy/scene-video-Query-System.git
cd scene-video-Query-System
```

### 2. Create Virtual Environment for both frontend and backend
```bash
conda create --name frontend python=3.11
cd frontend
pip install -r requirement.txt

conda create --name backend python=3.11
cd backend
pip install -r requirement.txt

```
### 3. Follow these struct below
- Download [**faiss_index-001.idx**](https://drive.google.com/file/d/1maA548j18COtiFc3bixuNR1Ue7X06Csa/view?usp=sharing) 
- Download [**folder_index.json**](https://drive.google.com/file/d/1AR819cau_miEzzxMMMTBQvpu43bpWiyl/view?usp=sharing) and [**image_paths.npy**](https://drive.google.com/file/d/1_wBVG7eShKgycMsLfwq8emsQmXA51R6g/view?usp=drive_link)

You can see these files on my drive. Put these files in the same structure in the Project Structure section.
## 📦 Dataset

This system is designed for **Vietnamese news and current affairs videos on youtube**:

- **Total Videos**: ~1,400
- **Domain**: News, documentaries, current events
- **Format**: MP4
- **Language**: Vietnamese content

You can see metadata in `front-end/media`


## 🗂️ Project Structure

```
scene-video-Query-System/
├── back-end/
│   ├── image_tmp/           # input for img retrivial
│   ├── imgPath/             # path of each pictures on drive
│   │   ├── folder_index.json 
│   │   └── image-paths.npy
│   ├── FILE.py              # connect with goodle drive
│   ├── main.py              # logic on backend
│   └── index/               # include file FAISS
│       └──faiss_index-001.idx
├── front-end/
│   ├── transnetv2/          # Scene detection model
│   ├── media.json/          # Information(metadata) about all 1400 video 
│   ├── Neighbor/            # result keyframes retrivial by neighbor
│   ├── Result/              # result feyframes retrivial by text
│   └── mmain.py             # front end by gradio
│
├── transnetv2_2.ipynb       # extract keyframe
│
├── transnettv2.py          # transition detection
│── transnetv2-weights/     # include saved_model for transnetv2.py
├── README.md
└── LICENSE
```


## 🔬 Technical Details

### TransNetV2 Scene Detection
- Input: Video mp4
- Output: Scene boundary predictions (file.txt)

### ViT-L-14 Feature Extraction
- Architecture: Vision Transformer Large (patch 14)
- Embedding dimension: 768
- Pre-trained: CLIP/OpenCLIP weights

### FAISS Indexing
- Index type: IVFFlat (Inverted File with Flat quantizer)
- Distance metric: Cosine similarity
- Number of clusters: 100
- Search probe: 10


## 🙏 Acknowledgments

- **TransNetV2**: [Souček & Lokoč (2020)](https://arxiv.org/abs/2008.04838)
- **Vision Transformer**: [Dosovitskiy et al. (2020)](https://arxiv.org/abs/2010.11929)
- **CLIP**: [Radford et al. (2021)](https://arxiv.org/abs/2103.00020)
- **FAISS**: [Johnson et al. (2019)](https://github.com/facebookresearch/faiss)

## 📞 Contact

**Hieu Nguyen**
- GitHub: [@Hieuabssy](https://github.com/Hieuabssy)
- Repository: [scene-video-Query-System](https://github.com/Hieuabssy/scene-video-Query-System)

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐



## 🗓️ Changelog

### v1.0.0 (2025-01-XX)
- ✨ Initial release
- ✅ TransNetV2 scene detection
- ✅ ViT-L-14 feature extraction
- ✅ FAISS vector indexing
- ✅ Text and image-based retrieval
- ✅ Support for ~1,400 Vietnamese news videos

---