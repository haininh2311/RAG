import os
import json
import faiss
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer


# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        index_path: str = "../embeddings/index.faiss",
        embedding_model: str = "bkai-foundation-models/vietnamese-bi-encoder",
        metadata_path: str = "../embeddings/metadata.json",
        top_k: int = 5,
    ):
        """
        Khởi tạo Retriever để tìm kiếm các đoạn văn bản liên quan.

        Args:
            index_path: Đường dẫn đến file FAISS index
            embedding_model: Mô hình embedding sử dụng để encode query
            metadata_path: Đường dẫn đến file metadata
            top_k: Số lượng kết quả trả về
        """
        self.index_path = index_path
        self.embedding_model = embedding_model
        self.metadata_path = metadata_path
        self.top_k = top_k

        self.index = None
        self.metadata = None
        self.model = None

        self._load_resources()

    def _load_resources(self):
        """
        Tải các tài nguyên cần thiết như FAISS index, metadata và mô hình embedding.
        """
        try:
            # Load FAISS index
            logger.info(f"Đang đọc FAISS index từ {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            logger.info(f"Đã đọc FAISS index với {self.index.ntotal} vectors")

            # Load metadata
            logger.info(f"Đang đọc metadata từ {self.metadata_path}")
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            logger.info(f"Đã đọc metadata cho {len(self.metadata)} chunks")

            # Load model
            logger.info(f"Đang tải mô hình embedding {self.embedding_model}")
            self.model = SentenceTransformer(self.embedding_model)
            logger.info("Đã tải mô hình embedding thành công")

        except Exception as e:
            logger.error(f"Lỗi khi tải tài nguyên: {e}")
            raise

    def encode_query(self, query: str) -> np.ndarray:
        """
        Mã hóa truy vấn thành vector.

        Args:
            query: Truy vấn văn bản

        Returns:
            np.ndarray: Vector mã hóa của truy vấn
        """
        if self.model is None:
            raise ValueError("Mô hình embedding chưa được tải")

        try:
            query_vector = self.model.encode([query], convert_to_numpy=True)
            return query_vector
        except Exception as e:
            logger.error(f"Lỗi khi tạo embedding cho query: {e}")
            raise

    def search(
        self, query: str, top_k: Optional[int] = None, return_texts: bool = True
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Tìm kiếm các đoạn văn bản liên quan đến câu truy vấn

        Args:
            query: Câu truy vấn
            top_k: Số lượng kết quả trả về, nếu None sẽ sử dụng giá trị mặc định
            return_texts: Có trả về nội dung văn bản hay không

        Returns:
            Tuple[List[Dict], List[float], List[str]]: Danh sách metadata và độ tương đồng và các đoạn văn bản
        """
        if top_k is None:
            top_k = self.top_k

        try:
            # Tạo embedding cho câu truy vấn
            query_vector = self.encode_query(query)

            # Tìm kiếm với FAISS
            distances, indices = self.index.search(query_vector, top_k)

            # Distances là một mảng 2D, lấy hàng đầu tiên
            distances = distances[0]
            indices = indices[0]

            # Chuẩn bị kết quả
            results = []
            for i, idx in enumerate(indices):
                if idx < 0 or idx >= len(self.metadata):
                    continue  # Bỏ qua nếu index không hợp lệ

                # Lấy metadata của đoạn văn tương ứng
                doc_metadata = self.metadata[idx].copy()

                # Thêm nội dung văn bản nếu cần
                if return_texts:
                    # Để đọc nội dung văn bản, cần phải biết đường dẫn đến file chunks
                    # Giả sử rằng text được lưu trong files theo cấu trúc
                    chunk_id = doc_metadata.get("chunk_id")
                    category = doc_metadata.get("category")
                    source_file = doc_metadata.get("source_file")

                    # Đọc từ file chunks để lấy nội dung
                    chunk_file = "../chunks/chunks.jsonl"
                    if os.path.exists(chunk_file):
                        with open(chunk_file, "r", encoding="utf-8") as f:
                            for line in f:
                                item = json.loads(line)
                                if (
                                    item.get("chunk_id") == chunk_id
                                    and item.get("category") == category
                                    and item.get("source_file") == source_file
                                ):
                                    doc_metadata["text"] = item["text"]
                                    break

                results.append(doc_metadata)

            return results, distances.tolist()

        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm: {e}")
            raise

    def retrieve_with_context(
        self, query: str, top_k: Optional[int] = None, max_context_length: int = 2500
    ) -> Dict[str, Any]:
        """
        Truy xuất các đoạn văn bản liên quan và kết hợp chúng thành một ngữ cảnh

        Args:
            query: Câu truy vấn
            top_k: Số lượng kết quả trả về
            max_context_length: Độ dài tối đa của ngữ cảnh

        Returns:
            Dict: Chứa query, context và các thông tin liên quan
        """
        results, scores = self.search(query, top_k=top_k, return_texts=True)

        # Kết hợp các đoạn văn tìm được thành một ngữ cảnh
        context_parts = []
        sources = []
        current_length = 0

        for i, result in enumerate(results):
            text = result.get("text", "")
            if not text:
                continue

            # Kiểm tra độ dài
            if current_length + len(text) > max_context_length:
                # Chỉ thêm một phần nếu vượt quá giới hạn
                remaining = max_context_length - current_length
                if remaining > 100:  # Chỉ thêm nếu còn đủ chỗ trống
                    text = text[:remaining]
                    context_parts.append(text)
                    current_length += len(text)
                break

            context_parts.append(text)
            current_length += len(text)

            # Thêm nguồn
            source = {
                "source_file": result.get("source_file", ""),
                "category": result.get("category", ""),
                "chunk_id": result.get("chunk_id", ""),
                "similarity": scores[i],
            }
            sources.append(source)

        # Tạo ngữ cảnh hoàn chỉnh
        context = "\n\n".join(context_parts)

        return {"query": query, "context": context, "sources": sources}


def main():
    # Ví dụ sử dụng
    retriever = Retriever(
        index_path="../embeddings/index.faiss",
        metadata_path="../embeddings/metadata.json",
        embedding_model="bkai-foundation-models/vietnamese-bi-encoder",
        top_k=8,
    )

    query = "Ai là hiệu trưởng trường Đại học Công nghệ Đại học Quốc gia Hà Nội?"
    results, scores = retriever.search(query)

    print("Kết quả tìm kiếm:", results)
    print("Độ tương đồng:", scores)

    context_result = retriever.retrieve_with_context(query)
    print("Ngữ cảnh:", context_result["context"])
    print("Nguồn:", context_result["sources"])


if __name__ == "__main__":
    main()
