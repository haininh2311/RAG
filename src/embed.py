#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import logging
from typing import List, Dict, Any, Tuple

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Embedder:
    def __init__(
        self,
        chunk_file: str = "../data/chunks/chunks.jsonl",
        embedding_model: str = "bkai-foundation-models/vietnamese-bi-encoder",
        output_dir: str = "../embeddings",
    ):
        """
        Khởi tạo lớp Embedder để tạo và lưu trữ embeddings.

        Args:
            chunk_file: Đường dẫn đến file JSONL chứa các chunks văn bản
            embedding_model: Mô hình embedding được sử dụng
            output_dir: Thư mục lưu trữ kết quả embedding
        """
        self.chunk_file = chunk_file
        self.embedding_model = embedding_model
        self.output_dir = output_dir
        self.texts = []
        self.metas = []
        self.embeddings = None
        self.index = None

        # Tạo thư mục output nếu chưa tồn tại
        os.makedirs(output_dir, exist_ok=True)

    def load_chunks(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Đọc dữ liệu từ file chunks JSONL

        Returns:
            Tuple[List[str], List[Dict]]: Danh sách văn bản và metadata
        """
        logger.info(f"Đang đọc chunks từ {self.chunk_file}")
        texts = []
        metas = []

        try:
            with open(self.chunk_file, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    texts.append(item["text"])
                    metas.append(
                        {
                            "source_file": item.get("source_file"),
                            "category": item.get("category"),
                            "chunk_id": item.get("chunk_id"),
                        }
                    )

            logger.info(f"Đã đọc {len(texts)} chunks")
            self.texts = texts
            self.metas = metas
            return texts, metas
        except Exception as e:
            logger.error(f"Lỗi khi đọc file chunks: {e}")
            raise

    def create_embeddings(self) -> np.ndarray:
        """
        Tạo embeddings từ các văn bản đã load

        Returns:
            np.ndarray: Ma trận embeddings
        """
        if not self.texts:
            self.load_chunks()

        logger.info(f"Tạo embeddings với mô hình {self.embedding_model}")
        try:
            model = SentenceTransformer(self.embedding_model)
            embeddings = model.encode(
                self.texts, show_progress_bar=True, convert_to_numpy=True
            )
            logger.info(f"Embedding shape: {embeddings.shape}")
            self.embeddings = embeddings
            return embeddings
        except Exception as e:
            logger.error(f"Lỗi khi tạo embeddings: {e}")
            raise

    def build_faiss_index(self) -> faiss.Index:
        """
        Tạo chỉ mục FAISS từ embeddings

        Returns:
            faiss.Index: Chỉ mục FAISS đã tạo
        """
        if self.embeddings is None:
            self.create_embeddings()

        logger.info("Đang tạo chỉ mục FAISS")
        try:
            dimension = self.embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(self.embeddings)
            self.index = index
            logger.info(f"Đã tạo chỉ mục FAISS với {index.ntotal} vectors")
            return index
        except Exception as e:
            logger.error(f"Lỗi khi tạo chỉ mục FAISS: {e}")
            raise

    def save_artifacts(self) -> None:
        """
        Lưu embeddings và metadata vào thư mục output
        """
        if self.embeddings is None:
            self.create_embeddings()

        emb_path = os.path.join(self.output_dir, "embeddings.npy")
        meta_path = os.path.join(self.output_dir, "metadata.json")
        faiss_path = os.path.join(self.output_dir, "index.faiss")

        logger.info(f"Lưu embeddings và metadata vào {self.output_dir}")
        try:
            # Lưu embeddings
            np.save(emb_path, self.embeddings)

            # Lưu metadata
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self.metas, f, ensure_ascii=False, indent=2)

            # Lưu FAISS index nếu đã tạo
            if self.index is not None:
                faiss.write_index(self.index, faiss_path)

            logger.info(f"Đã lưu embeddings và metadata thành công")
        except Exception as e:
            logger.error(f"Lỗi khi lưu dữ liệu: {e}")
            raise

    def run_pipeline(self) -> None:
        """
        Chạy toàn bộ quy trình embedding từ đầu đến cuối
        """
        logger.info("Bắt đầu quy trình embedding")
        self.load_chunks()
        self.create_embeddings()
        self.build_faiss_index()
        self.save_artifacts()
        logger.info("Hoàn thành quy trình embedding")


def main():
    # Các tham số có thể được đọc từ config hoặc command line arguments
    embedder = Embedder(
        chunk_file="../chunks/chunks.jsonl",
        embedding_model="bkai-foundation-models/vietnamese-bi-encoder",
        output_dir="../embeddings",
    )
    embedder.run_pipeline()


if __name__ == "__main__":
    main()
