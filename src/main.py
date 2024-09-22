import torch
from data_preprocessing import DataPreprocessor
from embedding_generation import EmbeddingGenerator
from Enum.data_type import DataType
from training import Trainer
import torch.nn as nn
import torch.optim as optim
from model import SimpleNN  # Import the model

def main():
    # File paths to datasets
    file_paths = {
        DataType.TRAIN.value: 'data/train.csv',
        DataType.TEST.value: 'data/test.csv',
        DataType.MISCONCEPTION.value: 'data/misconception_mapping.csv'
    }

    # Load and preprocess data
    preprocessor = DataPreprocessor()
    data = preprocessor.load_data(file_paths)

    # Clean and preprocess text
    processed_train_data = preprocessor.preprocess_text(data[DataType.TRAIN.value], 'QuestionText', ['AnswerAText', 'AnswerBText', 'AnswerCText', 'AnswerDText'])
    processed_test_data = preprocessor.preprocess_text(data[DataType.TEST.value], 'QuestionText', ['AnswerAText', 'AnswerBText', 'AnswerCText', 'AnswerDText'])

    # Generate embeddings
    embedding_generator = EmbeddingGenerator('distilbert-base-uncased')
    train_embeddings = embedding_generator.generate_embeddings(processed_train_data['cleaned_text'])
    test_embeddings = embedding_generator.generate_embeddings(processed_test_data['cleaned_text'])

    # Labels
    train_labels = processed_train_data['labels']
    test_labels = processed_test_data['labels']

    # Initialize model, criterion, optimizer
    model = SimpleNN()  # Use the defined SimpleNN model
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Use Adam optimizer

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Train the model
    trainer = Trainer(model, criterion, optimizer, batch_size=32, num_epochs=10, use_amp=True)
    trainer.train(train_embeddings, train_labels)

if __name__ == "__main__":
    main()
