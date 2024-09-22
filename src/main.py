import os
import torch
import pandas as pd
from data_preprocessing import DataPreprocessor
from embedding_generation import EmbeddingGenerator
from Enum.data_type import DataType
from training import Trainer, arithmetic_loss 
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
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

    # Ensure labels are encoded as integers
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)
    test_labels = label_encoder.transform(test_labels)  # Use transform for test labels

    model = SimpleNN(num_classes=4)  # Update with number of output classes
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    use_amp = torch.cuda.is_available()
    trainer = Trainer(model, criterion, optimizer, batch_size=32, num_epochs=10, use_amp=use_amp)
    trainer.train(train_embeddings, train_labels)

    # Evaluate the model
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        test_embeddings = test_embeddings.to(device)
        test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)
        test_outputs = model(test_embeddings)
        test_loss = criterion(test_outputs, test_labels)
        print(f"Test Loss: {test_loss.item():.4f}")

        # Calculate accuracy
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == test_labels).float().mean().item()
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

        # Save predictions to output folder
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

        output_file_path = os.path.join(output_dir, 'predictions.csv')
        predictions_df = pd.DataFrame({
            'True Labels': test_labels.cpu().numpy(),
            'Predicted Labels': predicted.cpu().numpy()
        })
        predictions_df.to_csv(output_file_path, index=False)
        print(f"Predictions saved to {output_file_path}")

if __name__ == "__main__":
    main()
