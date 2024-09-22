# In training.py

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move arithmetic_loss outside of the Trainer class
def arithmetic_loss(predictions, labels):
    # Custom logic for arithmetic error handling
    base_loss = nn.CrossEntropyLoss()(predictions, labels)
    
    # Add custom arithmetic penalties if needed
    penalty = 0.0  # Modify this if you have custom logic
    total_loss = base_loss + penalty
    return total_loss

class Trainer:
    def __init__(self, model, criterion, optimizer, batch_size=32, num_epochs=10, use_amp=False):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.use_amp = use_amp
        self.scaler = GradScaler('cuda') if self.use_amp else None
        self.label_encoder = LabelEncoder()

    def encode_labels(self, labels):
        return self.label_encoder.fit_transform(labels)

    def train(self, embeddings, labels):
        if isinstance(labels[0], str):
            labels = self.encode_labels(labels)

        embeddings = embeddings.clone().detach().to(device)  # Proper way to move tensors to device
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        dataset = TensorDataset(embeddings, labels)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for batch in loader:
                batch_embeddings, batch_labels = batch
                self.optimizer.zero_grad()

                if self.use_amp:
                    with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                        outputs = self.model(batch_embeddings)
                        loss = self.criterion(outputs, batch_labels)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(batch_embeddings)
                    loss = self.criterion(outputs, batch_labels)
                    loss.backward()
                    self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}")
