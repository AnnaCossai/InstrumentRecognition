import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import pandas as pd
import os
from torch import nn
from torchsummary import summary
from tqdm.auto import tqdm
import random
import numpy as np
from sklearn.model_selection import train_test_split
import librosa
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

NUM_RUN = 0  # Change this for each run with different seeds

class AudioDataset(Dataset):
    def __init__(self, metadata_path, audio_dir, transform=None):
        self.metadata = pd.read_csv(metadata_path)

        self.metadata.reset_index(inplace=True)

        self.audio_dir = audio_dir
        self.transform = transform

        # Map instrument_id to [0, num_classes-1]
        unique_ids = sorted(self.metadata['instrument_id'].unique())
        self.id_to_idx = {id_: idx for idx, id_ in enumerate(unique_ids)}
        self.metadata['instrument_id_mapped'] = self.metadata['instrument_id'].map(self.id_to_idx)

        # Create a dictionary mapping instrument_id to instrument name
        self.id_to_instrument = dict(zip(self.metadata['instrument_id_mapped'], self.metadata['instrument']))

        # Load all audio into memory
        self.audio_data = []
        print("Loading audio into memory...")
        # Filter out rows with missing or invalid entries if necessary
        self.metadata.dropna(subset=['subset', 'instrument_id', 'uuid4'], inplace=True)
        self.metadata.reset_index(drop=True, inplace=True)

        for idx in tqdm(range(len(self.metadata))):
            audio_info = self.metadata.iloc[idx]
            audio_name = 'Medley-solos-DB_'+audio_info['subset']+'-'+str(audio_info['instrument_id'])+"_"+audio_info['uuid4']+'.wav'
            audio_path = os.path.join(self.audio_dir, audio_name)

            # Load audio using librosa
            try:
                y, _ = librosa.load(audio_path, sr = 22050)
                audio_tensor = torch.from_numpy(y).float() # Convert numpy array to torch tensor and to float
            except Exception as e:
                print(f"Error loading audio file {audio_path}: {e}")
                continue

            if self.transform:
                 audio_tensor = self.transform(audio_tensor)

            self.audio_data.append(audio_tensor)
        print("Audio loading complete.")


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        audio_info = self.metadata.iloc[idx]
        audio_tensor = self.audio_data[idx] # Get pre-loaded audio tensor

        label_name = audio_info['instrument']
        label = audio_info['instrument_id_mapped']

        return audio_tensor, label, label_name

# Define paths
metadata_path = '8_instruments_metadata.csv'
audio_dir = '8_instruments_dataset'

# Define transformations
class Scale01(object):
    def __call__(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val > min_val:
            return (tensor - min_val) / (max_val - min_val)
        else:
            return tensor

transform = transforms.Compose([
    Scale01()
])

# Create the dataset instance
audio_dataset = AudioDataset(metadata_path=metadata_path, audio_dir=audio_dir, transform=transform)
print(f"Number of samples in the dataset: {len(audio_dataset)}")

metadata = audio_dataset.metadata
print(metadata.head())
print(audio_dataset[0][0])
print(audio_dataset[0][0].shape)

# Set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


BATCH_SIZE = 128

# Get the class column
class_col = 'instrument_id_mapped'

# Split off the test set (20%)
train_val_indices, test_indices = train_test_split(
    metadata.index,
    test_size=0.2,
    stratify=metadata[class_col],
    random_state=seed
)

# Split train and validation (10% of total, so 10/80 = 0.125 of remaining for val)
train_indices, val_indices = train_test_split(
    train_val_indices,
    test_size=0.125,
    stratify=metadata.loc[train_val_indices, class_col],
    random_state=seed
)

# Normalize data
# Compute mean and std dev of the training set
train_audio = [audio_dataset[i][0] for i in train_indices]
train_audio_tensor = torch.stack(train_audio)
mean = train_audio_tensor.mean(dim=(0))
std = train_audio_tensor.std(dim=(0))
print(f"Train set mean: {mean}")
print(f"Train set std: {std}")

class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        if self.mean is not None and self.std is not None:
            return (tensor - self.mean) / self.std
        else:
            return tensor

# Update the transform to include normalization
transform = transforms.Compose([
    Scale01(),  
    Normalize(mean=mean, std=std)
])
# Re-instantiate the dataset with the new transform
audio_dataset = AudioDataset(metadata_path=metadata_path, audio_dir=audio_dir, transform=transform)

# Create Subset datasets
train_dataset = Subset(audio_dataset, list(train_indices))
val_dataset = Subset(audio_dataset, list(val_indices))
test_dataset = Subset(audio_dataset, list(test_indices))

print(f"Number of samples in training set: {len(train_dataset)}")
print(f"Number of samples in validation set: {len(val_dataset)}")
print(f"Number of samples in test set: {len(test_dataset)}")

# Re-create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Number of batches in training dataloader: {len(train_dataloader)}")
print(f"Number of batches in validation dataloader: {len(val_dataloader)}")
print(f"Number of batches in test dataloader: {len(test_dataloader)}")


# Define the CNN architecture
class CNNNetwork1D(nn.Module):

    def __init__(self, num_classes=8):
        super().__init__()
        self.num_classes = num_classes
        # 4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=8,
                stride=1,
                padding=1
            ),
            nn.BatchNorm1d(16),  # Batch normalization to stabilize learning
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=8,
                stride=1,
                padding=1
            ),
            nn.BatchNorm1d(32),  # Batch normalization to stabilize learning
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=8,
                stride=1,
                padding=1
            ),
            nn.BatchNorm1d(64),  # Batch normalization to stabilize learning
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=8,
                stride=1,
                padding=1
            ),
            nn.BatchNorm1d(128),  # Batch normalization to stabilize learning
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(523648, self.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)

        if self.training:
            x = nn.Dropout(p=0.3)(x)

        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions


# Training function for a single epoch
def train_single_epoch(model, data_loader, val_dataloader, loss_fn, optimiser, device):
    model.train()
    for input, target, _ in data_loader:
        input = input.unsqueeze(1)
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    # Calculate training and validation accuracy and loss for each epoch
    train_accuracy, train_loss, _ = test(model, data_loader, loss_fn, device)
    val_accuracy, val_loss, _ = test(model, val_dataloader, loss_fn, device)
    
    return train_accuracy, train_loss, val_accuracy, val_loss


# Training function
def train(model, data_loader, val_dataloader, loss_fn, optimiser, device, epochs, run_num):
    best_val_acc = 0.0
    best_model_state = None
    train_accuracy_list = []
    val_accuracy_list = []
    train_loss_list = []
    val_loss_list = []

    for i in tqdm(range(epochs)):
        train_accuracy, train_loss, val_accuracy, val_loss = train_single_epoch(model, data_loader, val_dataloader, loss_fn, optimiser, device)

        train_accuracy_list.append(train_accuracy)
        val_accuracy_list.append(val_accuracy)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        tqdm.write(f'Epoch {i+1}/{epochs}, Train Accuracy: {(100*train_accuracy):.2f}, Train Loss: {train_loss:.4f}, Val Acc: {(100*val_accuracy):.2f}%, Val Loss: {val_loss:.4f}', end='\r')
        
        # Save the best model based on validation accuracy
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_state = model.state_dict()

    print("\nFinished training")
    print(f"\nBest Validation Accuracy: {(100*best_val_acc):.2f}%")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    os.makedirs('Plots_rawAudio', exist_ok=True)

    # Plotting and saving the training and validation accuracy
    plt.figure(figsize=(12, 5))
    plt.plot(range(1, epochs + 1), train_accuracy_list, label='Train')
    plt.plot(range(1, epochs + 1), val_accuracy_list, label='Validation')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Model Accuracy', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.savefig(f'Plots_rawAudio/accuracy_plot_{run_num}.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'Plots_rawAudio/accuracy_plot_{run_num}.pdf', bbox_inches='tight', dpi=300)

    # Plotting and saving the training and validation loss
    plt.figure(figsize=(12, 5))
    plt.plot(range(1, epochs + 1), train_loss_list, label='Train')
    plt.plot(range(1, epochs + 1), val_loss_list, label='Validation')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Model Loss', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.savefig(f'Plots_rawAudio/loss_plot_{run_num}.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'Plots_rawAudio/loss_plot_{run_num}.pdf', bbox_inches='tight', dpi=300)

    return model

def test(model, data_loader, loss_fn, device):
    model.eval()
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    test_loss, accuracy = 0, 0
    prediction_list = []
    with torch.no_grad():
        for input, target, _ in data_loader:
            input = input.unsqueeze(1)
            input, target = input.to(device), target.to(device)
            prediction = model(input)
            test_loss += loss_fn(prediction, target).item()
            accuracy += (prediction.argmax(1) == target).type(torch.float).sum().item()
            prediction_list.append(prediction.argmax(1).cpu().numpy())
    test_loss /= num_batches
    accuracy /= size
    prediction_list = np.concatenate(prediction_list).tolist()

    return accuracy, test_loss, prediction_list


# Inizialize training

if torch.cuda.is_available():
  device = "cuda:0"
else:
  device = "cpu"
print(f"Using {device}")


EPOCHS = 300
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4

seed = seed + NUM_RUN
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
    
cnn = CNNNetwork1D(num_classes = 8).to(device)

summary(cnn, input_size=(1, 65536), device=device)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

#Train
best_model = train(cnn, train_dataloader, val_dataloader, loss_fn, optimizer, device, EPOCHS, NUM_RUN)

#Test on test set
test_accuracy, test_loss, prediction_list = test(best_model, test_dataloader, loss_fn, device)
print(f"Test Loss Run {NUM_RUN}: {test_loss:.4f}, Test Accuracy Run {NUM_RUN}: {(100*test_accuracy):.2f}%")

# Save the best model to disk
os.makedirs('models_rawAudio', exist_ok=True)
model_path = f'models_rawAudio/best_model_raw_audio_{NUM_RUN}.pth'
torch.save(best_model.state_dict(), model_path)
print(f"Best model saved to {model_path}")

# Get true labels for the test set
true_labels = []
for _, target, _ in test_dataloader:
    true_labels.extend(target.numpy().tolist())

# Compute confusion matrix
cm = confusion_matrix(true_labels, prediction_list, normalize='true')

# Get class names in the correct order
class_names = [audio_dataset.id_to_instrument[idx] for idx in sorted(audio_dataset.id_to_instrument.keys())]

# Plot and save the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xticks(rotation=45)
plt.xlabel('Predicted label', fontsize=14)
plt.ylabel('True label', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.savefig(f'Plots_rawAudio/confusion_matrix_{NUM_RUN}.png', bbox_inches='tight', dpi=300)
plt.savefig(f'Plots_rawAudio/confusion_matrix_{NUM_RUN}.pdf', bbox_inches='tight', dpi=300)

