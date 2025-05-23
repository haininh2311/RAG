import os
import json
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import nltk

nltk.download("punkt")
from dotenv import load_dotenv
import re

load_dotenv()

MIN_CHUNK_LEN = 200
MAX_CHUNK_LEN = 500
INPUT_DIR = "data_clean_verified"
OUTPUT_FILE = "chunks/chunks.jsonl"


def chunk_text_by_char(text, min_len=200, max_len=500):
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) <= max_len:
            current += " " + sentence
        else:
            if len(current.strip()) >= min_len:
                chunks.append(current.strip())
            current = sentence
    if len(current.strip()) >= min_len:
        chunks.append(current.strip())
    return chunks


def process_single_file(file_path, category):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text_by_char(text)
    file_name = os.path.basename(file_path)

    result = []
    for i, chunk in enumerate(chunks):
        result.append(
            {
                "source_file": file_name,
                "category": category,
                "chunk_id": i,
                "text": chunk,
            }
        )
    return result


#  XỬ LÝ TOÀN BỘ THƯ MỤC
def process_all_files(input_root, output_file):
    with open(output_file, "w", encoding="utf-8") as out:
        for category in os.listdir(input_root):
            folder_path = os.path.join(input_root, category)
            if not os.path.isdir(folder_path):
                continue

            for fname in tqdm(os.listdir(folder_path), desc=f"📁 {category}"):
                if not fname.endswith(".txt"):
                    continue

                file_path = os.path.join(folder_path, fname)
                chunks = process_single_file(file_path, category)
                for chunk in chunks:
                    out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nĐã tạo chunk từ {input_root} vào: {output_file}")


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    process_all_files(INPUT_DIR, OUTPUT_FILE)
