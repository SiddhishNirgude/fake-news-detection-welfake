"""
models.py
---------
Neural network model architectures for the WELFake fake news
detection project. Includes BiLSTM baseline and 3-branch hybrid
model combining TF-IDF, GloVe/LSTM, and linguistic features.

All models are implemented in PyTorch.

Author: Siddhish Nirgude
Course: CMSE 928 - Applied Machine Learning
"""

import os

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# BiLSTM Classifier
# ---------------------------------------------------------------------------

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM model for binary text classification.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embedding_dim : int
        Dimension of word embeddings.
    lstm_units : int
        Number of LSTM units per direction.
    dropout : float
        Dropout rate applied before the output layer.
    """

    def __init__(self, vocab_size, embedding_dim, lstm_units, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_units,
            bidirectional=True,
            batch_first=True,
        )
        self.linear1  = nn.Linear(lstm_units * 2, 64)
        self.relu     = nn.ReLU()
        self.dropout  = nn.Dropout(dropout)
        self.linear2  = nn.Linear(64, 1)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Integer token indices of shape (batch, seq_len).

        Returns
        -------
        torch.Tensor
            Sigmoid-activated output of shape (batch, 1).
        """
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        # Global max pooling over the sequence dimension
        x = torch.max(lstm_out, dim=1).values
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.linear2(x))
        return x


def build_bilstm(
    vocab_size,
    embedding_dim=128,
    max_len=300,
    lstm_units=128,
    dropout=0.3,
    num_classes=1,
):
    """
    Build a Bidirectional LSTM model for binary text classification.

    Architecture:
      Embedding(vocab_size, embedding_dim)
      BiLSTM(embedding_dim, lstm_units, bidirectional=True)
      GlobalMaxPool over sequence dimension
      Dense(lstm_units * 2, 64, relu)
      Dropout(dropout)
      Dense(64, 1, sigmoid)

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embedding_dim : int
        Dimension of word embeddings. Default 128.
    max_len : int
        Maximum sequence length. Default 300.
    lstm_units : int
        Number of LSTM units per direction. Default 128.
    dropout : float
        Dropout rate. Default 0.3.
    num_classes : int
        Number of output classes. Default 1 for binary.

    Returns
    -------
    nn.Module
        Compiled BiLSTM model as a PyTorch nn.Module.
    """
    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        dropout=dropout,
    )

    print("BiLSTM Model Summary")
    print("=" * 45)
    total_params = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n
        print(f"  {name:<35} {n:>10,}")
    print("-" * 45)
    print(f"  {'Total parameters':<35} {total_params:>10,}")
    print(f"  max_len (external truncation)          {max_len}")
    print("=" * 45)

    return model


# ---------------------------------------------------------------------------
# Hybrid Classifier
# ---------------------------------------------------------------------------

class HybridClassifier(nn.Module):
    """
    3-branch hybrid model combining TF-IDF, BiLSTM, and linguistic features.

    Parameters
    ----------
    tfidf_dim : int
        Dimension of the TF-IDF feature vector.
    vocab_size : int
        Vocabulary size for the embedding layer.
    embedding_dim : int
        Embedding dimension.
    lstm_units : int
        BiLSTM units per direction.
    linguistic_dim : int
        Number of handcrafted linguistic features.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        tfidf_dim,
        vocab_size,
        embedding_dim,
        lstm_units,
        linguistic_dim,
        dropout,
    ):
        super().__init__()

        # Branch 1: TF-IDF dense path
        self.branch1 = nn.Sequential(
            nn.Linear(tfidf_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Branch 2: Sequential text path via embedding and BiLSTM
        self.embedding   = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm        = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_units,
            bidirectional=True,
            batch_first=True,
        )
        self.linear_lstm = nn.Linear(lstm_units * 2, 128)
        self.relu        = nn.ReLU()

        # Branch 3: Handcrafted linguistic features path
        self.branch3 = nn.Sequential(
            nn.Linear(linguistic_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Fusion layer: concatenated output dims = 128 + 128 + 32 = 288
        self.fusion = nn.Sequential(
            nn.Linear(288, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x_tfidf, x_seq, x_linguistic):
        """
        Forward pass through all three branches and the fusion layer.

        Parameters
        ----------
        x_tfidf : torch.Tensor
            TF-IDF features of shape (batch, tfidf_dim).
        x_seq : torch.Tensor
            Token index sequences of shape (batch, seq_len).
        x_linguistic : torch.Tensor
            Linguistic features of shape (batch, linguistic_dim).

        Returns
        -------
        torch.Tensor
            Sigmoid-activated output of shape (batch, 1).
        """
        out1 = self.branch1(x_tfidf)

        embedded = self.embedding(x_seq)
        lstm_out, _ = self.lstm(embedded)
        # Global max pooling collapses the sequence dimension
        out2 = self.relu(self.linear_lstm(torch.max(lstm_out, dim=1).values))

        out3 = self.branch3(x_linguistic)

        combined = torch.cat([out1, out2, out3], dim=1)
        return torch.sigmoid(self.fusion(combined))


def build_hybrid_model(
    tfidf_dim,
    vocab_size,
    embedding_dim=128,
    max_len=300,
    lstm_units=128,
    linguistic_dim=10,
    dropout=0.3,
):
    """
    Build a 3-branch hybrid model combining TF-IDF, GloVe + BiLSTM,
    and handcrafted linguistic features.

    Architecture:
      Branch 1 (TF-IDF):
        Input(tfidf_dim) -> Dense(256, relu) -> Dropout -> Dense(128, relu)

      Branch 2 (LSTM):
        Input(max_len integers) -> Embedding(vocab_size, embedding_dim)
        -> BiLSTM(lstm_units) -> GlobalMaxPool -> Dense(128, relu)

      Branch 3 (Linguistic):
        Input(linguistic_dim) -> Dense(32, relu) -> Dropout

      Fusion:
        Concatenate(128 + 128 + 32 = 288 dims)
        -> Dense(128, relu) -> Dropout(dropout) -> Dense(1, sigmoid)

    Parameters
    ----------
    tfidf_dim : int
        Dimension of TF-IDF feature vector.
    vocab_size : int
        Vocabulary size for embedding layer.
    embedding_dim : int
        Embedding dimension. Default 128.
    max_len : int
        Max sequence length. Default 300.
    lstm_units : int
        BiLSTM units per direction. Default 128.
    linguistic_dim : int
        Number of linguistic features. Default 10.
    dropout : float
        Dropout rate. Default 0.3.

    Returns
    -------
    nn.Module
        3-branch hybrid model as PyTorch nn.Module.
    """
    model = HybridClassifier(
        tfidf_dim=tfidf_dim,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        linguistic_dim=linguistic_dim,
        dropout=dropout,
    )

    print("Hybrid Model Summary")
    print("=" * 45)
    total_params = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n
        print(f"  {name:<40} {n:>10,}")
    print("-" * 45)
    print(f"  {'Total parameters':<40} {total_params:>10,}")
    print(f"  max_len (external truncation)             {max_len}")
    print("=" * 45)

    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    model,
    train_loader,
    val_loader,
    epochs=10,
    lr=0.001,
    patience=3,
    device="cpu",
):
    """
    Train a PyTorch model with early stopping.
    Uses Binary Cross Entropy loss and Adam optimizer.
    Stops training if validation loss does not improve
    for patience consecutive epochs.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to train.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    epochs : int
        Maximum number of epochs. Default 10.
    lr : float
        Learning rate for Adam optimizer. Default 0.001.
    patience : int
        Early stopping patience. Default 3.
    device : str
        Device to train on. Default cpu.

    Returns
    -------
    dict
        Training history with keys:
        train_loss, val_loss, train_acc, val_acc.
        Each value is a list of floats, one per epoch.
    """
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "val_loss":   [],
        "train_acc":  [],
        "val_acc":    [],
    }

    best_val_loss   = float("inf")
    best_weights    = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # --- Training phase ---
        model.train()
        train_loss_sum = 0.0
        train_correct  = 0
        train_total    = 0

        for batch in train_loader:
            # Unpack: single tensor input (BiLSTM) or tuple (Hybrid)
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, labels = batch
            else:
                *inputs, labels = batch
                inputs = inputs[0] if len(inputs) == 1 else inputs

            labels = labels.float().to(device)

            optimizer.zero_grad()

            if isinstance(inputs, (list, tuple)):
                inputs = [x.to(device) for x in inputs]
                outputs = model(*inputs).squeeze(1)
            else:
                inputs = inputs.to(device)
                outputs = model(inputs).squeeze(1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            preds = (outputs >= 0.5).long()
            train_correct += (preds == labels.long()).sum().item()
            train_total   += labels.size(0)

        train_loss = train_loss_sum / train_total
        train_acc  = train_correct / train_total

        # --- Validation phase ---
        model.eval()
        val_loss_sum = 0.0
        val_correct  = 0
        val_total    = 0

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, labels = batch
                else:
                    *inputs, labels = batch
                    inputs = inputs[0] if len(inputs) == 1 else inputs

                labels = labels.float().to(device)

                if isinstance(inputs, (list, tuple)):
                    inputs = [x.to(device) for x in inputs]
                    outputs = model(*inputs).squeeze(1)
                else:
                    inputs = inputs.to(device)
                    outputs = model(inputs).squeeze(1)

                loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * labels.size(0)
                preds = (outputs >= 0.5).long()
                val_correct += (preds == labels.long()).sum().item()
                val_total   += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_acc  = val_correct / val_total

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(round(val_acc, 4))

        print(
            f"Epoch {epoch:>3}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        # Early stopping: track best weights and count stagnant epochs
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    # Restore the weights from the best validation epoch
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print("Best model weights restored.")

    return history


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_pytorch_model(model, filepath):
    """
    Save a trained PyTorch model state dict to disk.
    Creates parent directories if they do not exist.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model.
    filepath : str
        Full path including filename and .pt extension.

    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved: {filepath}")


def load_pytorch_model(model, filepath, device="cpu"):
    """
    Load saved state dict into a PyTorch model instance.
    The model architecture must be created before calling
    this function.

    Parameters
    ----------
    model : nn.Module
        Uninitialised model with correct architecture.
    filepath : str
        Path to saved .pt state dict file.
    device : str
        Device to load model onto. Default cpu.

    Returns
    -------
    nn.Module
        Model with loaded weights set to eval mode.
    """
    state_dict = torch.load(filepath, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded: {filepath}")
    return model


def get_device():
    """
    Detect and return the best available device.
    Returns MPS for Apple Silicon, CUDA for NVIDIA GPU,
    or CPU as fallback.

    Returns
    -------
    torch.device
        Best available device.
    """
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    return torch.device(device)
