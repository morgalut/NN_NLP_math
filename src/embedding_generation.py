import pandas as pd
import torch
from transformers import DistilBertModel, DistilBertTokenizer

class EmbeddingGenerator:
    def __init__(self, model_name='distilbert-base-uncased', batch_size=16):
        """
        Initializes the EmbeddingGenerator with the specified model and tokenizer.
        Moves the model to GPU if available, otherwise uses CPU.

        Args:
            model_name (str): The name of the transformer model.
            batch_size (int): The batch size for processing inputs.
        """
        self.model = DistilBertModel.from_pretrained(model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
        self.model.to(self.device)

    def generate_embeddings(self, texts):
        """
        Generates embeddings for the input texts.

        Args:
            texts (list or pd.Series or str): Input texts to generate embeddings from.

        Returns:
            torch.Tensor: Generated embeddings as a tensor.
        """
        # Ensure texts is a list, even if a single string or pandas Series is provided
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, pd.Series):
            texts = texts.tolist()

        embeddings = []
        self.model.eval()  # Set model to evaluation mode
        
        with torch.no_grad():
            # Process in batches to avoid memory overload
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i+self.batch_size]
                # Tokenize the batch with padding and truncation
                inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move inputs to the correct device
                # Generate embeddings for the batch
                batch_embeddings = self.model(**inputs).last_hidden_state.mean(dim=1).cpu()  # Average over tokens
                embeddings.append(batch_embeddings)

        # Concatenate all batch embeddings into a single tensor
        embeddings = torch.cat(embeddings, dim=0)
        return embeddings

    def save_embeddings(self, embeddings, file_path):
        """
        Saves the embeddings to a file.

        Args:
            embeddings (torch.Tensor): The embeddings to save.
            file_path (str): The file path where the embeddings will be saved.
        """
        torch.save(embeddings, file_path)
        print(f"Embeddings saved to {file_path}")
