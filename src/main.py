import torch
from data_preprocessing import DataPreprocessor
from embedding_generation import EmbeddingGenerator
from Enum.data_type import DataType
from training import Trainer, arithmetic_loss 
import torch.nn as nn
import torch.optim as optim
from model import SimpleNN  # Import the updated SimpleNN model

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

    model = SimpleNN(num_classes=4)  # Update with number of output classes
    criterion = nn.CrossEntropyLoss()  # Use a standard loss function like CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    use_amp = torch.cuda.is_available()  # Enable AMP only if CUDA is available
    trainer = Trainer(model, criterion, optimizer, batch_size=32, num_epochs=10, use_amp=use_amp)
    trainer.train(train_embeddings, train_labels)

if __name__ == "__main__":
    main()
