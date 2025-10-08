# utils.py

import os
from typing import List

def list_file_recursively(base_path):
    file_paths = []
    for root, _, files in os.walk(base_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths