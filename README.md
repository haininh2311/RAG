# RAG System

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system as part of the course assignment. The system is designed to answer questions using documents compiled from publicly available sources.

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

+ e.g. Qwen2.5, LLaMA-2-chat, Phi-2


