# RAG System

## Overview

This project is a question-answering system using Retrieval-Augmented Generation (RAG) to answer questions related to Vietnam National University (VNU) and the University of Engineering and Technology (UET).

## Data Format

**Input**(`questions.txt`):  Each line contains one question.

**Output**(`result.json`): Each line contains the system-generated answer corresponding to the question on the same line in `questions.txt`.

**Reference**(`reference_answers.txt`): Each line contains one or more correct answers`,` separated by `;`.

## Building Knowledge Resource

### Compiling a knowledge resource:

+ VNU official site (introduction, history, organization)

+ Wikipedia - Vietnam National University, Hanoi

+ Admissions portals and sub-universities (USSH, ULIS, etc.)

+ Regulations (e.g., doctoral training policies)

+ Academic programs, especially international

### Collecting raw data:

My knowledge resource might include a mix of HTML pages, PDFs, and plain text documents. So I will need to clean this data and convert it into a file format that suites my model development. Here are some tools that I could use:

+ To process HTML pages: [beautifulsoup4](https://pypi.org/project/beautifulsoup4/).
+ To parse PDF documents into plain text: [pypdf](https://github.com/py-pdf/pypdf) or [pdfplumber](https://github.com/jsvine/pdfplumber).

By the end of this step, I will have a collection of documents that will serve as the knowledge resource for my RAG system.


##  Annotating Data

### Purpose

+ Training: `data/train/questions.txt` + `reference_answers.txt`

+ Testing: `data/test/questions.txt` + `reference_answers.txt`

### Guidelines

+ Ensure relevance to domain (VNU, UET)

+ Cover diverse question types

+ Ensure quality: annotate manually if needed

+ Ensure quantity: enough to distinguish model performance



## RAG System Components

### Document & Query Embedder

+ Convert queries/documents into vector space

+ e.g. bkai-foundation-models/vietnamese-bi-encoder

### Retriever

Fetch relevant documents using similarity search

+ e.g. FAISS, Chroma

### Reader (LLM)

+ Generate answer based on query + retrieved context

+ e.g. LLama3

## Implement 


1. Clone repository:

```bash
git clone <repository-url>
```

2. Implement library:

```bash
pip install -r requirements.txt
```

3. Configuration `config.yaml`

4. Create a .env file in the src/ directory:

```bash
HF_TOKEN=your_huggingface_token
API_KEY=your_groq_api_key
```


### Batch QA
```bash
cd src/
python main.py --questions ../data/test/questions.txt --output ../system_outputs/system_output_1.json
```

### Web Interface with Gradio
```bash
cd src/
python app.py
```

Open your browser and go to the address shown in the terminal (usually http://127.0.0.1:7860).
Start asking questions about VNU!

## Note 
+ Uses FAISS for efficient document retrieval.

+ Question answering is handled by the LLaMA3-8B-8192 model.

+ Embedding is done using the Vietnamese bi-encoder from BKAI.




