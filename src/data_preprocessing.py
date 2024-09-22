import pandas as pd
from transformers import DistilBertTokenizer
from Enum.model_type import ModelType
import re


class DataPreprocessor:
    def __init__(self, tokenizer_name=ModelType.DISTILBERT.value):
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)

    def load_data(self, file_paths):
        """Load data from CSV files into pandas DataFrames."""
        data = {}
        for name, path in file_paths.items():
            try:
                print(f"Loading data from: {path}")
                df = pd.read_csv(path)

                if df.empty:
                    print(f"Warning: {path} is empty.")
                else:
                    data[name] = df
                    print(f"Data loaded for {name}, with columns: {data[name].columns}")

            except FileNotFoundError:
                print(f"Error: {path} not found.")
            except pd.errors.EmptyDataError:
                print(f"Error: {path} is empty.")
            except Exception as e:
                print(f"Unexpected error while loading {path}: {e}")

        return data

    def clean_text(self, text):
        """Clean a given text by lowercasing, removing extra spaces, and special characters."""
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        return text

    def preprocess_text(self, data, question_col, answer_cols):
        """Preprocess the text data by combining question and answers into a clean format."""
        required_columns = [question_col] + answer_cols
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise KeyError(f"Missing columns in data: {missing_columns}")

        # Clean the text data efficiently
        try:
            cleaned_questions = data.get(question_col, "").apply(self.clean_text).tolist()
            cleaned_answers = data[answer_cols].apply(lambda row: ' '.join(row.fillna('')), axis=1).apply(self.clean_text).tolist()

            cleaned_texts = [q + " " + a for q, a in zip(cleaned_questions, cleaned_answers)]

            return {'cleaned_text': cleaned_texts, 'labels': data.get('CorrectAnswer', None)}

        except KeyError as e:
            raise KeyError(f"Error during preprocessing: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during text preprocessing: {e}")
