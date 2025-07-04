#!/usr/bin/env python3
"""
Script to initialize the FAISS vector database with Q&A CSV data
"""
import os
import sys
import json
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Default paths
CSV_PATH = './data/Dataset_chatbot_new.csv'
FAISS_INDEX_PATH = './faiss_index.index'
METADATA_PATH = './metadata.json'


def detect_qa_columns(df):
    columns = df.columns.tolist()
    question_patterns = ['question', 'q', 'query', 'questions', 'ask']
    answer_patterns = ['answer', 'a', 'response', 'answers', 'reply']

    question_col = next((col for col in columns if any(p in col.lower() for p in question_patterns)), None)
    answer_col = next((col for col in columns if any(p in col.lower() for p in answer_patterns)), None)

    if not question_col or not answer_col:
        if len(columns) >= 2:
            question_col, answer_col = columns[0], columns[1]

    return question_col, answer_col


def prepare_qa_documents(df, question_col, answer_col):
    documents = []
    metadatas = []

    for idx, row in df.iterrows():
        if pd.isna(row[question_col]) or pd.isna(row[answer_col]):
            continue

        question = str(row[question_col]).strip()
        answer = str(row[answer_col]).strip()

        if not question or not answer:
            continue

        # Question
        documents.append(question)
        metadatas.append({'type': 'question', 'question': question, 'answer': answer})

        # Answer
        documents.append(answer)
        metadatas.append({'type': 'answer', 'question': question, 'answer': answer})

        # Combined
        combined = f"Q: {question}\nA: {answer}"
        documents.append(combined)
        metadatas.append({'type': 'qa_pair', 'question': question, 'answer': answer})

    return documents, metadatas


def main():
    print("🚀 Starting Q&A Vector Database Initialization with FAISS...")

    csv_path = CSV_PATH  # local scope

    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found at {csv_path}")
        os.makedirs('./data', exist_ok=True)
        print("✅ Created data/ directory")
        custom_path = input("Enter the full path to your CSV file: ").strip()
        if os.path.exists(custom_path):
            csv_path = custom_path
        else:
            print("❌ File not found. Exiting.")
            return

    print(f"📄 Using CSV file: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
        print(f"📋 CSV Info: {df.shape[0]} rows, {df.shape[1]} columns")

        question_col, answer_col = detect_qa_columns(df)
        if not question_col or not answer_col:
            print("❌ Could not auto-detect Q&A columns.")
            question_col = input("Enter question column name: ").strip()
            answer_col = input("Enter answer column name: ").strip()

        print(f"✅ Detected Q: '{question_col}', A: '{answer_col}'")

        documents, metadatas = prepare_qa_documents(df, question_col, answer_col)
        print(f"📝 Prepared {len(documents)} documents for embedding.")

        print("🔢 Generating embeddings...")
        embeddings = embedding_model.encode(documents, convert_to_numpy=True, show_progress_bar=True)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, FAISS_INDEX_PATH)
        print(f"💾 FAISS index saved to {FAISS_INDEX_PATH}")

        with open(METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(metadatas, f, ensure_ascii=False, indent=2)
        print(f"🗃️ Metadata saved to {METADATA_PATH}")

        print("\n✅ Initialization complete. You can now run queries using the FAISS index.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
