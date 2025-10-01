Video Scene Retrieval System

This project implements a video scene retrieval system designed to search and retrieve relevant video scenes from a collection of around 1400 news and current affairs videos.

📌 Overview

The system extracts keyframes from videos, generates embeddings using a Vision Transformer (ViT-L-14) model, and stores them in a Faiss-powered vector index for efficient retrieval.

🚀 Pipeline

Video Input

The system takes raw video files as input (news and current affairs).

Scene Detection

A TransNetV2 model is used to detect scene boundaries.

The output is a .txt file containing the list of detected scenes.

Keyframe Extraction

For each scene, 3 keyframes are extracted to represent the visual content.

Feature Extraction

The ViT-L-14 model is applied to each keyframe to extract vector embeddings.

Vector Indexing

Embeddings are stored in a Faiss vector index for similarity search.

Retrieval System

Given a query (image/frame/text-to-image embedding), the system retrieves the most relevant scenes from the indexed database.

🛠️ Technologies

Python 3.x

TransNetV2
 – for scene boundary detection

ViT-L-14 (CLIP or OpenCLIP)
 – for image feature extraction

Faiss
 – for efficient similarity search

NumPy / PyTorch / OpenCV
