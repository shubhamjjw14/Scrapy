#!/bin/bash

# Step 1: Install dependencies
pip install requests beautifulsoup4 mysql-connector-python configparser sentence-transformers pymilvus

# Step 2: Run the Python script for crawling and storing data
python3 extract.py

# Step 3: Output success message
echo "Crawling, storing in MySQL, and Milvus indexing completed successfully."
