import pandas as pd
from typing import List, Dict, Tuple
import os

class QACSVProcessor:
    """
    Specialized CSV processor for Question-Answer datasets
    Optimized for chatbot applications using FAISS
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.question_col = None
        self.answer_col = None
        self.load_csv()
        self.detect_qa_columns()

    def load_csv(self):
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"✅ Loaded CSV with {len(self.df)} rows and {len(self.df.columns)} columns")
            print(f"📋 Columns: {list(self.df.columns)}")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            raise

    def detect_qa_columns(self):
        columns = self.df.columns.tolist()
        question_patterns = ['question', 'q', 'query', 'questions', 'ask', 'prompt']
        answer_patterns = ['answer', 'a', 'response', 'answers', 'reply', 'solution']

        for col in columns:
            if any(pattern in col.lower() for pattern in question_patterns):
                self.question_col = col
                break

        for col in columns:
            if any(pattern in col.lower() for pattern in answer_patterns):
                self.answer_col = col
                break

        if not self.question_col or not self.answer_col:
            if len(columns) >= 2:
                self.question_col = columns[0]
                self.answer_col = columns[1]
                print(f"🔍 Auto-detected: Question='{self.question_col}', Answer='{self.answer_col}'")

        if self.question_col and self.answer_col:
            print(f"✅ Detected Q&A columns: '{self.question_col}' -> '{self.answer_col}'")
        else:
            print("❌ Could not detect Q&A columns automatically")

    def set_qa_columns(self, question_col: str, answer_col: str):
        if question_col not in self.df.columns:
            raise ValueError(f"Question column '{question_col}' not found")
        if answer_col not in self.df.columns:
            raise ValueError(f"Answer column '{answer_col}' not found")

        self.question_col = question_col
        self.answer_col = answer_col
        print(f"✅ Set Q&A columns: '{question_col}' -> '{answer_col}'")

    def clean_qa_data(self) -> pd.DataFrame:
        if not self.question_col or not self.answer_col:
            raise ValueError("Q&A columns not set")

        clean_df = self.df.copy()
        initial_count = len(clean_df)
        clean_df = clean_df.dropna(subset=[self.question_col, self.answer_col])
        clean_df = clean_df[
            (clean_df[self.question_col].str.strip() != '') &
            (clean_df[self.answer_col].str.strip() != '')
        ]
        clean_df[self.question_col] = clean_df[self.question_col].str.strip()
        clean_df[self.answer_col] = clean_df[self.answer_col].str.strip()
        final_count = len(clean_df)
        print(f"🧹 Cleaned data: {initial_count} -> {final_count} rows")
        return clean_df

    def prepare_documents_for_faiss(self, strategy: str = 'comprehensive') -> Tuple[List[str], List[Dict], List[str]]:
        clean_df = self.clean_qa_data()
        documents = []
        metadatas = []
        ids = []

        for idx, row in clean_df.iterrows():
            question = str(row[self.question_col]).strip()
            answer = str(row[self.answer_col]).strip()

            base_metadata = {
                'question': question,
                'answer': answer,
                'row_index': idx,
                'source': 'qa_csv'
            }

            if strategy in ['questions_only', 'comprehensive']:
                documents.append(question)
                metadatas.append({**base_metadata, 'type': 'question'})
                ids.append(f"q_{idx}")

            if strategy in ['answers_only', 'comprehensive']:
                documents.append(answer)
                metadatas.append({**base_metadata, 'type': 'answer'})
                ids.append(f"a_{idx}")

            if strategy in ['qa_pairs', 'comprehensive']:
                combined = f"Question: {question}\nAnswer: {answer}"
                documents.append(combined)
                metadatas.append({**base_metadata, 'type': 'qa_pair'})
                ids.append(f"qa_{idx}")

        print(f"📄 Generated {len(documents)} documents using '{strategy}' strategy")
        return documents, metadatas, ids

    def get_qa_statistics(self) -> Dict:
        if not self.question_col or not self.answer_col:
            return {}

        clean_df = self.clean_qa_data()
        stats = {
            'total_pairs': len(clean_df),
            'avg_question_length': clean_df[self.question_col].str.len().mean(),
            'avg_answer_length': clean_df[self.answer_col].str.len().mean(),
            'max_question_length': clean_df[self.question_col].str.len().max(),
            'max_answer_length': clean_df[self.answer_col].str.len().max(),
            'question_column': self.question_col,
            'answer_column': self.answer_col
        }
        return stats

    def preview_qa_pairs(self, n: int = 5) -> List[Dict]:
        if not self.question_col or not self.answer_col:
            return []
        clean_df = self.clean_qa_data()
        pairs = []
        for i in range(min(n, len(clean_df))):
            row = clean_df.iloc[i]
            pairs.append({
                'question': row[self.question_col],
                'answer': row[self.answer_col],
                'index': i
            })
        return pairs

    def export_cleaned_csv(self, output_path: str):
        clean_df = self.clean_qa_data()
        clean_df.to_csv(output_path, index=False)
        print(f"📁 Exported cleaned data to {output_path}")
        return output_path