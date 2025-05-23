import os
import json
import logging
from typing import List, Dict, Any, Optional
import time
from dotenv import load_dotenv
from retrieve import Retriever
from groq import Groq

load_dotenv()

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("API_KEY")

client = Groq(api_key=API_KEY)


class QASystem:
    def __init__(
        self,
        retriever: Retriever,
        model_name: str = "llama3-8b-8192",
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 5000,
        logging_enabled: bool = True,
        results_dir: str = "../system_outputs",
    ):
        """
        Khởi tạo hệ thống Hỏi Đáp

        Args:
            retriever: Đối tượng Retriever để truy xuất thông tin
            model_name: Tên của mô hình ngôn ngữ lớn để trả lời
            api_url: URL của API cho mô hình (nếu sử dụng API)
            api_key: API key (nếu cần)
            temperature: Độ đa dạng của câu trả lời (0-1)
            max_tokens: Số tokens tối đa trong câu trả lời
            logging_enabled: Bật/tắt ghi log
            results_dir: Thư mục để lưu kết quả
        """
        self.retriever = retriever
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logging_enabled = logging_enabled
        self.results_dir = results_dir

        # Tạo thư mục kết quả nếu chưa tồn tại
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

    def _construct_prompt(self, query: str, context: str) -> str:
        """
        Xây dựng prompt cho mô hình

        Args:
            query: Câu hỏi
            context: Danh sách các đoạn văn bản liên quan

        Returns:
            Prompt cho mô hình
        """

        # context_text = "\n".join([item["text"] for item in context])

        prompt = f"""Dưới đây là thông tin liên quan:

{context}

Dựa vào thông tin trên, hãy trả lời câu hỏi sau một cách ngắn gọn và chính xác nhất có thể, không giải thích:
{query}
"""
        return prompt

    def _call_model(self, prompt: str) -> str:
        """
        Gọi mô hình để lấy câu trả lời

        Args:
            prompt: Prompt cho mô hình

        Returns:
            Câu trả lời từ mô hình
        """
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là trợ lý AI chuyên trả lời tiếng Việt.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=512,
                top_p=1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Lỗi gọi API: {e}")
            return "Lỗi khi gọi mô hình."

    def answer_question(
        self,
        query: str,
        top_k: int = 5,
        max_context_length: int = 2000,
        log_result: bool = True,
    ) -> Dict[str, Any]:

        # Bắt đầu tính thời gian xử lý
        start_time = time.time()

        # Lấy ngữ cảnh liên quan từ retriever
        logger.info(f"Đang truy xuất thông tin cho câu hỏi: {query}")
        retrieval_result = self.retriever.retrieve_with_context(
            query, top_k=top_k, max_context_length=max_context_length
        )
        context = retrieval_result["context"]

        # Tạo prompt
        prompt = self._construct_prompt(query, context)

        logger.info("Gửi truy vấn tới API...")

        answer = self._call_model(prompt)

        return {
            "query": query,
            "answer": answer,
            "processing_time": time.time() - start_time,
        }

    def _log_result(self, result: Dict[str, Any]) -> None:
        """
        Lưu kết quả vào file.

        Args:
            result: Kết quả cần lưu
        """
        timestamp = time.strftime("%Y%m%d%H%M%S")
        filename = os.path.join(self.results_dir, f"result_{timestamp}.json")

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"Đã lưu kết quả vào file: {filename}")

    def batch_answer(self, questions_file: str, output_file: str = None) -> List[str]:
        """
        Xử lý hàng loạt câu hỏi từ file.

        Args:
            questions_file: Đường dẫn đến file chứa danh sách câu hỏi
            output_file: Đường dẫn đến file để lưu kết quả

        Returns:
            Danh sách kết quả
        """
        logger.info(f"Đang xử lý hàng loạt câu hỏi từ file: {questions_file}")

        # Đọc danh sách câu hỏi
        with open(questions_file, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]

        results = []
        total_questions = len(questions)

        # Xử lý từng câu hỏi
        for i, question in enumerate(questions, 1):
            logger.info(f"Đang xử lý câu hỏi {i}/{total_questions}: {question}")
            result = self.answer_question(question, log_result=False)
            # answer = result["answer"].strip()

            results.append(result)
            time.sleep(1)  # Thêm thời gian nghỉ giữa các câu hỏi để tránh quá tải API

            # In tiến độ
            print(f"Đã xử lý {i}/{total_questions} câu hỏi")

        # # Lưu tất cả kết quả vào file nếu được yêu cầu
        # if output_file:
        #     with open(output_file, "w", encoding="utf-8") as f:
        #         for ans in results:
        #             f.write(ans.strip() + "\n")
        #     logger.info(f"Đã lưu kết quả vào: {output_file}")

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Đã lưu kết quả vào: {output_file}")

        return results

    def truncate_context(self, context_text: str, max_tokens: int = 1024) -> str:
        """
        Cắt ngắn context nếu vượt quá số token cho phép
        """
        tokens = self.tokenizer.encode(context_text, add_special_tokens=False)
        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens)


if __name__ == "__main__":
    retriever = Retriever(
        index_path="../embeddings/index.faiss",
        metadata_path="../embeddings/metadata.json",
        embedding_model="bkai-foundation-models/vietnamese-bi-encoder",
        top_k=5,
    )

    qa_system = QASystem(
        retriever=retriever,
        # model_name=MODEL_NAME,
    )

    prompt = "Ai là hiệu trưởng của trường Đại Học Công Nghệ"
    output = qa_system.answer_question(prompt)
    print(f"Câu hỏi: {prompt}")
    print(output)
