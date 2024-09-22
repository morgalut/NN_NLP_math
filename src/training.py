import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler  # Corrected torch.amp import
from sklearn.preprocessing import LabelEncoder  

from decorators import mixed_precision_decorator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer:
    def __init__(self, model, criterion, optimizer, batch_size=32, num_epochs=10, use_amp=False):
        self.model = model.to(device)  
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.use_amp = use_amp
        self.scaler = GradScaler() if self.use_amp else None
        self.label_encoder = LabelEncoder()

    # Define the missing encode_labels method
    def encode_labels(self, labels):
        return self.label_encoder.fit_transform(labels)

    def train(self, embeddings, labels):
        if isinstance(labels[0], str):
            labels = self.encode_labels(labels)

        # Move tensors to the device
        embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device) if not isinstance(embeddings, torch.Tensor) else embeddings.to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device) if not isinstance(labels, torch.Tensor) else labels.to(device)

        dataset = TensorDataset(embeddings, labels)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for batch in loader:
                batch_embeddings, batch_labels = batch
                batch_embeddings, batch_labels = batch_embeddings.to(device), batch_labels.to(device)  

                self.optimizer.zero_grad()

                if self.use_amp:
                    with autocast(device.type):
                        outputs = self.model(batch_embeddings)
                        loss = self.criterion(outputs, batch_labels)

                    # Perform mixed precision scaling if enabled
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
