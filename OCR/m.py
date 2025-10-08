from elasticsearch import Elasticsearch
import json

# Kết nối tới Elasticsearch (giống như trong file helpers.py)
client = Elasticsearch("http://localhost:9200")

def print_results(response):
    """Hàm tiện ích để in kết quả tìm kiếm cho đẹp."""
    print(f"Tìm thấy tổng cộng: {response['hits']['total']['value']} kết quả.")
    if not response['hits']['hits']:
        print("Không có kết quả nào phù hợp.")
        return

    for hit in response['hits']['hits']:
        score = hit['_score']
        source = hit['_source']
        vid_name = source['vid_name']
        keyframe_id = source['keyframe_id']
        text = source['text']
        
        print("-" * 50)
        print(f"Score: {score:.2f}")
        print(f"Video: {vid_name}")
        print(f"Keyframe ID: {keyframe_id}")
        # In một đoạn trích ngắn của văn bản
        print(f"Text Snippet: {text[:200]}...")

def save_results(response, output_file="results.json"):
    """Lưu top 100 kết quả tìm kiếm thành JSON theo cấu trúc video/keyframe: text."""

    results = {}

    # Lấy tối đa 100 kết quả (nếu có)
    hits = response['hits']['hits'][:100]

    for hit in hits:
        source = hit['_source']
        vid_name = source['vid_name']
        keyframe_id = source['keyframe_id']
        text = source['text']

        # Nếu video chưa có trong dict thì thêm vào
        if vid_name not in results:
            results[vid_name] = {}

        # Gán keyframe: text
        results[vid_name][keyframe_id] = text

    # Ghi ra file JSON (UTF-8, có format đẹp)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Đã lưu {len(hits)} kết quả vào {output_file}")

if __name__ == "__main__":
    query_text = "đồng xoài bình phước"
    print(f"Tìm kiếm các câu có chứa {query_text}")
    query_body_1 = {
        "query": {
            "match": {
                "text": query_text
            }
        }
    }
    
    response = client.search(index="ocr", body=query_body_1)
    save_results(response)
    print_results(response)