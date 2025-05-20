import os
import json
import logging
from typing import List, Dict, Any, Optional
import time
from dotenv import load_dotenv
from retrieve import Retriever
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")


class QASystem:
    def __init__(
        self,
        retriever: Retriever,
        model_name: str = MODEL_NAME,
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

        # Tạo tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, token=HF_TOKEN
        )
        print(f"Đã tải tokenizer cho mô hình")

        # Tạo mô hình
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="cuda",
            token=HF_TOKEN,
        )
        print(f"Đã tải mô hình")

        print(f"{model_name}")

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

Dựa vào thông tin trên, hãy trả lời câu hỏi sau một cách ngắn gọn và chính xác nhất có thể:
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
        # inputs = self.tokenizer(prompt, return_tensors="pt")

        # device = next(self.model.parameters()).device
        # inputs = {k: v.to(device) for k, v in inputs.items()}

        # Format prompt theo dạng chat
        messages = [
            {"role": "user", "content": prompt},
        ]

        # # Áp dụng template đặc biệt của mô hình chat
        # inputs = self.tokenizer.apply_chat_template(
        #     messages,
        #     return_tensors="pt",
        #     add_generation_prompt=True  # tùy vào model
        # )

        # Apply chat template and encode as input_ids
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, add_special_tokens=True, return_tensors="pt"
        )

        # Chuyển input_ids sang device
        device = next(self.model.parameters()).device
        prompt_ids = prompt_ids.to(device)

        # Tạo attention mask từ input_ids
        attention_mask = torch.ones_like(prompt_ids)

        # Generate output
        outputs = self.model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.split("assistant")[-1].strip()
        answer = answer.split("user")[-1].strip()

        # ==== Tách phần trả lời thật sự ====
        if prompt in answer:
            answer = answer.split(prompt)[-1].strip()

        return answer.strip()

    def answer_question(
        self,
        query: str,
        top_k: int = 5,
        max_context_length: int = 1000,
        log_result: bool = True,
    ) -> Dict[str, Any]:
        """
        Trả lời câu hỏi dựa trên thông tin truy xuất được.

        Args:
            query: Câu hỏi
            top_k: Số lượng đoạn văn bản liên quan cần truy xuất
            max_context_length: Độ dài tối đa của ngữ cảnh
            log_result: Có lưu kết quả hay không

        Returns:
            Dict: Kết quả chứa câu hỏi, câu trả lời và thông tin liên quan
        """
        # Bắt đầu tính thời gian xử lý
        start_time = time.time()

        # Lấy ngữ cảnh liên quan từ retriever
        logger.info(f"Đang truy xuất thông tin cho câu hỏi: {query}")
        retrieval_result = self.retriever.retrieve_with_context(
            query, top_k=top_k, max_context_length=max_context_length
        )
        context = retrieval_result["context"]
        sources = retrieval_result["sources"]

        # Gộp các đoạn context thành 1 chuỗi
        context_text = context

        # Cắt context nếu quá dài
        context_text = self.truncate_context(context_text, max_tokens=2000)

        # Tạo prompt
        prompt = self._construct_prompt(query, context_text)

        # Kiểm tra nội dung
        print(prompt)

        # Cảnh báo nếu prompt quá dài
        total_tokens = len(self.tokenizer.encode(prompt))
        if total_tokens > 2048:
            logger.warning(
                f"Prompt dài {total_tokens} tokens — có thể bị cắt bởi model!"
            )

        logger.info("Đang gọi mô hình ngôn ngữ để trả lời")
        answer = self._call_model(prompt)

        # Tính thời gian xử lý
        processing_time = time.time() - start_time

        # Tạo kết quả
        result = {
            "query": query,
            "answer": answer,
            # "sources": sources,
            "processing_time": processing_time,
            # "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Ghi log kết quả nếu được yêu cầu
        if log_result and self.logging_enabled:
            self._log_result(result)

        return result

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
        model_name=MODEL_NAME,
    )

    prompt = "Ai là hiệu trưởng của trường Đại Học Công Nghệ"
    output = qa_system.answer_question(prompt)
    print(f"Câu hỏi: {prompt}")
    print(output)
