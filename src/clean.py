import nltk
from dotenv import load_dotenv
import re
import os

nltk.download("punkt")


load_dotenv()


def clean_text_for_chunking(text):
    # 1. Thay thế ký tự Unicode lỗi
    replacements = {
        "": "=",
        "": "-",
        "": "×",
        "": "÷",
        "": "∑",
        "": "±",
        "": "→",
        "": "→",
        " ": " ",
        "": "<=",
        "": ">=",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # 2. Tách dòng và xử lý dòng
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 3. Loại bỏ dòng rời rạc in HOA
        if len(line.split()) < 5 and line.isupper():
            continue
        # 4. Loại bỏ dòng chứa link hoặc biểu mẫu
        if re.search(
            r"http[s]?://|facebook\.com|Email:|Mật khẩu|Đăng nhập|CCCD|Họ tên",
            line,
            re.IGNORECASE,
        ):
            continue
        # 5. Chuẩn hóa dấu chấm câu cuối câu (nếu thiếu và không phải danh sách chấm đầu dòng)
        if (
            not re.search(r"[.!?…:]$", line)
            and not line.startswith("-")
            and not line.startswith("+")
        ):
            line += "."
        cleaned_lines.append(line)
    # 6. Gộp lại thành văn bản sạch
    return " ".join(cleaned_lines)


if __name__ == "__main__":
    INPUT_DIR = "data_clean"
    OUTPUT_DIR = "data_clean_verified"

    file_count = 0
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for folder in os.listdir(INPUT_DIR):
        input_subdir = os.path.join(INPUT_DIR, folder)
        output_subdir = os.path.join(OUTPUT_DIR, folder)
        os.makedirs(output_subdir, exist_ok=True)

        if not os.path.isdir(input_subdir):
            continue

        for fname in os.listdir(input_subdir):
            if not fname.endswith(".txt"):
                continue

            input_path = os.path.join(input_subdir, fname)
            output_path = os.path.join(output_subdir, fname)

            with open(input_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            cleaned_text = clean_text_for_chunking(raw_text)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            file_count += 1
            print(f"Cleaned: {folder}/{fname}")

    print(f"Đã tạo {file_count} file trong: {OUTPUT_DIR}")
