# helpers.py

from elasticsearch import Elasticsearch

# --- KẾT NỐI ELASTICSEARCH ---
# Thay đổi host và port nếu Elasticsearch của bạn chạy ở địa chỉ khác
elastic_client = Elasticsearch("http://localhost:9200")


# --- CẤU HÌNH CHO INDEX TIẾNG VIỆT ---
# Đây là phần rất quan trọng để tìm kiếm tiếng Việt hiệu quả.
# Nó sẽ xử lý việc bỏ dấu, chuyển thành chữ thường, v.v.
vietnamese_index_settings = {
    "settings": {
        "analysis": {
            "analyzer": {
                "vietnamese_analyzer": {
                    "tokenizer": "icu_tokenizer",
                    "filter": [
                        "icu_folding", # Xóa dấu (ví dụ: "việt" -> "viet")
                        "lowercase"
                    ]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "vid_name": {
                "type": "keyword" # Tìm kiếm chính xác tên video
            },
            "keyframe_id": {
                "type": "integer"
            },
            "text": {
                "type": "text",
                "analyzer": "vietnamese_analyzer" # Áp dụng bộ phân tích tiếng Việt
            }
        }
    }
}