import os
import logging
import argparse
import time
from dotenv import load_dotenv
from retrieve import Retriever
from answer import QASystem
from config import load_config



load_dotenv()

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")


def main():

    parser = argparse.ArgumentParser(description="Hệ thống Hỏi Đáp dạng batch")
    parser.add_argument(
        "--questions",
        type=str,
        required=True,
        help="Đường dẫn file chứa danh sách câu hỏi (mỗi dòng một câu)",
    )
    parser.add_argument("--output", type=str, help="File để lưu kết quả")
    parser.add_argument(
        "--model",
        type=str,
        default="llama3-8b-8192",
        help="Tên mô hình ngôn ngữ",
    )
    parser.add_argument(
        "--top_k", type=int, default=8, help="Số đoạn context truy xuất"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Độ đa dạng của câu trả lời"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=2000, help="Số tokens tối đa cho câu trả lời"
    )

    args = parser.parse_args()

    config = load_config()
    qa_system = QASystem.from_config(config)

    output_file = args.output or f"../system_outputs/results.txt"
    results = qa_system.batch_answer(args.questions, output_file)
    print(f"Đã xử lý {len(results)} câu hỏi và lưu kết quả vào: {output_file}")


if __name__ == "__main__":
    main()
