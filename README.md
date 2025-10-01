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

## 📋 Requirements

### System Requirements
- **OS**: Linux, macOS, or Windows
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 16GB minimum
- **Storage**: 100GB+ for video data and index

### Software Dependencies
```
Python >= 3.8
CUDA >= 11.0 (for GPU acceleration)
```

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Hieuabssy/scene-video-Query-System.git
cd scene-video-Query-System
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# On Linux/macOS
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained Models
```bash
python scripts/download_models.py
```

## 📦 Dataset

This system is designed for **Vietnamese news and current affairs videos**:

- **Total Videos**: ~1,400
- **Domain**: News, documentaries, current events
- **Format**: MP4, AVI, MKV
- **Language**: Vietnamese content

Place your videos in the `data/videos/` directory before processing.

## 💻 Usage

### Quick Start

```python
from src.pipeline import VideoQuerySystem

# Initialize the system
system = VideoQuerySystem()

# Process videos (one-time setup)
system.process_videos("data/videos/")

# Search with text
results = system.search("biểu tình trên đường phố", top_k=10)

# Search with image
results = system.search_by_image("query.jpg", top_k=10)

# Display results
for rank, result in enumerate(results, 1):
    print(f"{rank}. Video: {result['video_name']}")
    print(f"   Scene: {result['scene_id']} at {result['timestamp']}")
    print(f"   Similarity: {result['score']:.3f}\n")
```

### Step-by-Step Processing

#### 1. Scene Detection
```python
from src.scene_detection import SceneDetector

detector = SceneDetector(model_path="models/transnetv2")
scenes = detector.detect("data/videos/news_video.mp4")
print(f"Found {len(scenes)} scenes")
```

#### 2. Extract Keyframes
```python
from src.keyframe_extraction import KeyframeExtractor

extractor = KeyframeExtractor(frames_per_scene=3)
keyframes = extractor.extract("data/videos/news_video.mp4", scenes)
```

#### 3. Generate Embeddings
```python
from src.feature_extraction import FeatureExtractor

feature_extractor = FeatureExtractor(model="ViT-L-14")
embeddings = feature_extractor.encode(keyframes)
```

#### 4. Build Index
```python
from src.indexing import IndexBuilder

builder = IndexBuilder()
builder.add_embeddings(embeddings, metadata)
builder.save("index/video_index.faiss")
```

#### 5. Query System
```python
from src.retrieval import Retriever

retriever = Retriever(index_path="index/video_index.faiss")
results = retriever.search("cháy nhà cao tầng", top_k=10)
```

## 🗂️ Project Structure

```
scene-video-Query-System/
├── data/
│   ├── videos/              # Input video files
│   ├── scenes/              # Detected scene metadata
│   ├── keyframes/           # Extracted keyframes
│   └── embeddings/          # Generated feature vectors
│
├── models/
│   ├── transnetv2/          # Scene detection model
│   └── vit_l_14/            # Vision transformer weights
│
├── index/
│   └── video_index.faiss    # FAISS vector index
│
├── src/
│   ├── __init__.py
│   ├── scene_detection.py   # TransNetV2 implementation
│   ├── keyframe_extraction.py
│   ├── feature_extraction.py # ViT-L-14 encoder
│   ├── indexing.py          # FAISS index builder
│   ├── retrieval.py         # Search functionality
│   └── pipeline.py          # End-to-end pipeline
│
├── scripts/
│   ├── download_models.py   # Download pre-trained weights
│   ├── process_batch.py     # Batch video processing
│   └── evaluate.py          # Evaluation metrics
│
├── notebooks/
│   └── demo.ipynb           # Interactive demo
│
├── tests/
│   └── test_pipeline.py
│
├── requirements.txt
├── config.yaml
├── README.md
└── LICENSE
```

## ⚙️ Configuration

Edit `config.yaml` to customize system behavior:

```yaml
scene_detection:
  model: transnetv2
  threshold: 0.5
  min_scene_length: 15  # frames

keyframe_extraction:
  frames_per_scene: 3
  method: uniform  # uniform, diverse, or adaptive

feature_extraction:
  model: ViT-L-14
  device: cuda  # cuda or cpu
  batch_size: 32
  image_size: 224

indexing:
  index_type: IVFFlat  # Flat, IVFFlat, HNSW
  nlist: 100
  metric: cosine  # cosine or l2

retrieval:
  top_k: 10
  score_threshold: 0.5
```

## 📊 Performance

Benchmarked on NVIDIA RTX 3090:

| Operation | Speed | Notes |
|-----------|-------|-------|
| Scene Detection | ~30 FPS | TransNetV2 inference |
| Feature Extraction | ~200 imgs/sec | ViT-L-14 batch processing |
| Index Building | ~2 min | For ~50,000 scenes |
| Query Time | <50ms | Top-10 retrieval |
| Index Size | ~2.5GB | 1,400 videos processed |

## 🔬 Technical Details

### TransNetV2 Scene Detection
- Input: Video frames at 12 FPS
- Output: Scene boundary predictions
- Threshold: 0.5 (configurable)

### ViT-L-14 Feature Extraction
- Architecture: Vision Transformer Large (patch 14)
- Embedding dimension: 768
- Pre-trained: CLIP/OpenCLIP weights

### FAISS Indexing
- Index type: IVFFlat (Inverted File with Flat quantizer)
- Distance metric: Cosine similarity
- Number of clusters: 100
- Search probe: 10

## 📈 Example Queries

### Text Queries (Vietnamese)
```python
# Search for events
results = system.search("tai nạn giao thông")

# Search for people
results = system.search("phát biểu của chính trị gia")

# Search for locations
results = system.search("chợ đông người")

# Search for activities
results = system.search("người biểu tình xuống đường")
```

### Image-Based Queries
```python
# Find similar scenes
results = system.search_by_image("reference_scene.jpg")
```

## 🧪 Evaluation

Run evaluation on test set:

```bash
python scripts/evaluate.py --test_data data/test_queries.json --output results.json
```

### Metrics
- **Precision@K**: Accuracy of top-K results
- **Recall@K**: Coverage of relevant results
- **mAP**: Mean Average Precision
- **MRR**: Mean Reciprocal Rank

## 🛠️ Troubleshooting

### Common Issues

**Problem**: CUDA out of memory
```bash
# Solution: Reduce batch size in config.yaml
feature_extraction:
  batch_size: 16  # Reduce from 32
```

**Problem**: Scene detection too slow
```bash
# Solution: Use GPU acceleration
scene_detection:
  device: cuda
```

**Problem**: Index file too large
```bash
# Solution: Use compressed index type
indexing:
  index_type: IVFPQ  # Product Quantization
  m: 8               # Number of sub-quantizers
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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

## 📝 Citation

```bibtex
@software{scene_video_query_system,
  author = {Nguyen, Hieu},
  title = {Scene Video Query System},
  year = {2025},
  url = {https://github.com/Hieuabssy/scene-video-Query-System}
}
```

## 🗓️ Changelog

### v1.0.0 (2025-01-XX)
- ✨ Initial release
- ✅ TransNetV2 scene detection
- ✅ ViT-L-14 feature extraction
- ✅ FAISS vector indexing
- ✅ Text and image-based retrieval
- ✅ Support for ~1,400 Vietnamese news videos

---

**Made with ❤️ by Hieu Nguyen**
