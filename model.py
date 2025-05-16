import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.models import ResNet50_Weights
import logging
from tqdm import tqdm
import numpy as np
from datetime import datetime
import psutil
import time
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
import csv
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def get_available_devices():
    """Get list of available devices (CPU and GPUs)"""
    devices = ['cpu']  # Default to CPU
    if torch.cuda.is_available():
        devices.extend([f'cuda:{i}' for i in range(torch.cuda.device_count())])
        # Set default device to GPU
        torch.cuda.set_device(0)
    return devices

def check_system_resources():
    """Check system resources before training"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    gpu_memory = []
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory.append(torch.cuda.get_device_properties(i).total_memory)
    
    logging.info(f"CPU Usage: {cpu_percent}%")
    logging.info(f"Available Memory: {memory.available / (1024 * 1024 * 1024):.2f} GB")
    
    if torch.cuda.is_available():
        for i, mem in enumerate(gpu_memory):
            logging.info(f"GPU {i} Memory: {mem / (1024 * 1024 * 1024):.2f} GB")
    
    if cpu_percent > 80:
        logging.warning("High CPU usage detected. Consider closing other applications.")
    if memory.available < 2 * 1024 * 1024 * 1024:  # Less than 2GB available
        logging.warning("Low memory available. Training might be slow.")

def get_optimal_batch_size():
    """Get optimal batch size based on available memory"""
    if torch.cuda.is_available():
        # For GPU training, use a larger batch size
        return 32  # Increased batch size for GPU training
    return 2  # Default batch size for CPU training

class AdvancedHatefulMemeDetector(nn.Module):
    def __init__(self):
        super(AdvancedHatefulMemeDetector, self).__init__()
        
        # Image feature extraction using ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_projection = nn.Linear(2048, 1024)
        
        # Text feature extraction using BERT
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_projection = nn.Linear(768, 1792)  # Increased dimension
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=2816, num_heads=8)  # Updated dimension
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(2816, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(2816, 1024),  # Match checkpoint dimensions
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)  # Binary classification
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, images, text_input):
        # Process image
        image_features = self.image_encoder(images)
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.image_projection(image_features)
        
        # Process text
        text_outputs = self.text_encoder(**text_input)
        text_features = text_outputs.last_hidden_state
        text_features = torch.mean(text_features, dim=1)  # Average pooling
        text_features = self.text_projection(text_features)
        
        # Combine features
        combined_features = torch.cat([image_features, text_features], dim=1)
        
        # Apply attention
        attn_output, _ = self.attention(
            combined_features.unsqueeze(0),
            combined_features.unsqueeze(0),
            combined_features.unsqueeze(0)
        )
        attn_output = attn_output.squeeze(0)
        
        # Apply fusion
        fused_features = self.fusion(attn_output)
        
        # Classification
        logits = self.classifier(combined_features)
        
        return logits
    
    def predict(self, image, text):
        """
        Make a prediction for a single meme
        Args:
            image: PIL Image or tensor
            text: str or dict with 'input_ids' and 'attention_mask'
        Returns:
            dict: Prediction results with confidence scores
        """
        self.eval()
        device = next(self.parameters()).device
        
        try:
            # Process image
            if isinstance(image, str):  # If image path is provided
                image = Image.open(image).convert('RGB')
            if isinstance(image, Image.Image):
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                image = transform(image).unsqueeze(0)
            image = image.to(device)
            
            # Process text
            if isinstance(text, str):
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                text_input = tokenizer(
                    text,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
            else:
                text_input = text
            text_input = {k: v.to(device) for k, v in text_input.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self(image, text_input)
                probabilities = F.softmax(outputs, dim=1)
                confidence, prediction = torch.max(probabilities, dim=1)
            
            return {
                'hateful': bool(prediction.item()),
                'confidence': confidence.item(),
                'probabilities': {
                    'not_hateful': probabilities[0][0].item(),
                    'hateful': probabilities[0][1].item()
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

class MemeDataset(Dataset):
    def __init__(self, jsonl_file, img_dir, transform=None, max_text_length=128):
        self.data = []
        self.transform = transform
        self.max_text_length = max_text_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.img_dir = img_dir
        
        # Get list of available images
        available_images = set(os.listdir(img_dir))
        
        # Load data from JSONL file
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    # Extract just the filename from the path
                    img_filename = os.path.basename(item['img'].replace('\\', '/').replace('/', os.sep))
                    
                    # Try different variations of the filename
                    possible_filenames = [
                        img_filename,  # Original filename
                        img_filename.lstrip('0'),  # Remove leading zeros
                        f"{int(img_filename.split('.')[0]):05d}.png",  # Add leading zeros
                        img_filename.split('.')[0] + '.png'  # Ensure .png extension
                    ]
                    
                    # Find the first matching filename
                    found_filename = None
                    for possible_name in possible_filenames:
                        if possible_name in available_images:
                            found_filename = possible_name
                            break
                    
                    if found_filename:
                        img_path = os.path.join(self.img_dir, found_filename)
                        if os.path.exists(img_path):  # Double check file exists
                            self.data.append({
                                'id': item['id'],
                                'img_path': img_path,
                                'text': item['text'],
                                'label': item['label']
                            })
                        else:
                            logging.warning(f"Image file not found: {img_path}")
                    else:
                        logging.warning(f"Image not found for ID {item['id']}. Tried: {possible_filenames}")
                except Exception as e:
                    logging.error(f"Error processing line in {jsonl_file}: {str(e)}")
                    continue
        
        logging.info(f"Loaded {len(self.data)} examples from {jsonl_file}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and transform image
        try:
            if not os.path.exists(item['img_path']):
                raise FileNotFoundError(f"Image file not found: {item['img_path']}")
                
            image = Image.open(item['img_path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            logging.error(f"Error loading image {item['img_path']}: {str(e)}")
            # Return a blank image if loading fails
            image = torch.zeros((3, 224, 224))
        
        # Process text
        try:
            text = item['text']
            encoding = self.tokenizer(
                text,
                max_length=self.max_text_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            text_input = {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze()
            }
        except Exception as e:
            logging.error(f"Error processing text for item {item['id']}: {str(e)}")
            # Return zero tensors if text processing fails
            text_input = {
                'input_ids': torch.zeros(self.max_text_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_text_length, dtype=torch.long)
            }
        
        return {
            'id': item['id'],
            'image': image,
            'text': text_input,
            'label': torch.tensor(item['label'], dtype=torch.long)
        }

def train_model(model, train_loader, val_loader, num_epochs=6, learning_rate=2e-5, resume_training=False, checkpoint_path='checkpoints/best_model.pt'):
    """
    Train the model with checkpoint management and resumption capabilities
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        resume_training: Whether to resume from checkpoint
        checkpoint_path: Path to checkpoint file
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Log system information
    logging.info(f"Training on device: {device}")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Calculate class weights for imbalanced dataset
    labels = []
    for batch in train_loader:
        labels.extend(batch['label'].numpy())
    class_counts = np.bincount(labels)
    class_weights = torch.FloatTensor(1.0 / class_counts)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    
    logging.info(f"Class distribution: {dict(zip(range(len(class_counts)), class_counts))}")
    logging.info(f"Class weights: {class_weights.tolist()}")
    
    # Initialize optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Initialize loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Initialize training state
    start_epoch = 0
    best_val_acc = 0.0
    early_stopping_counter = 0
    early_stopping_patience = 2
    
    # Resume training if requested and checkpoint exists
    if resume_training and os.path.exists(checkpoint_path):
        try:
            logging.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_acc = checkpoint['val_acc']
            logging.info(f"Resumed training from epoch {start_epoch} with best validation accuracy: {best_val_acc:.2f}%")
            if 'train_acc' in checkpoint:
                logging.info(f"Previous training accuracy: {checkpoint['train_acc']:.2f}%")
            if 'train_loss' in checkpoint:
                logging.info(f"Previous training loss: {checkpoint['train_loss']:.4f}")
        except Exception as e:
            logging.error(f"Error loading checkpoint: {str(e)}")
            logging.info("Starting training from scratch")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        batch_times = []
        
        # Training loop with progress bar
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            batch_start_time = time.time()
            
            images = batch['image'].to(device)
            text_input = {k: v.to(device) for k, v in batch['text'].items()}
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images, text_input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # Log batch progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg_batch_time = np.mean(batch_times[-100:])
                current_acc = 100. * train_correct / train_total
                logging.info(f"Batch {batch_idx + 1}/{len(train_loader)} - "
                           f"Loss: {loss.item():.4f} - "
                           f"Acc: {current_acc:.2f}% - "
                           f"Avg Batch Time: {avg_batch_time:.3f}s")
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_batch_times = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc='Validation')):
                batch_start_time = time.time()
                
                images = batch['image'].to(device)
                text_input = {k: v.to(device) for k, v in batch['text'].items()}
                labels = batch['label'].to(device)
                
                outputs = model(images, text_input)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                batch_time = time.time() - batch_start_time
                val_batch_times.append(batch_time)
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Log epoch results
        logging.info(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        logging.info(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        logging.info(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        logging.info(f"Average Training Batch Time: {np.mean(batch_times):.3f}s")
        logging.info(f"Average Validation Batch Time: {np.mean(val_batch_times):.3f}s")
        
        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'train_acc': train_acc,
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, 'checkpoints/latest_checkpoint.pt')
        logging.info("Saved latest checkpoint")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stopping_counter = 0
            torch.save(checkpoint, checkpoint_path)
            logging.info(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        else:
            early_stopping_counter += 1
            logging.info(f'No improvement in validation accuracy. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}')
        
        # Early stopping
        if early_stopping_counter >= early_stopping_patience:
            logging.info('Early stopping triggered')
            break
    
    logging.info(f"\nTraining completed. Best validation accuracy: {best_val_acc:.2f}%")
    return model

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_trained_model(checkpoint_path='checkpoints/best_model.pt'):
    """
    Load a trained model from checkpoint
    Args:
        checkpoint_path: Path to the model checkpoint file
    Returns:
        model: Loaded model ready for inference
    """
    try:
        # Initialize the model
        model = AdvancedHatefulMemeDetector()
        
        # Load checkpoint with warning suppression
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load model state
        model.load_state_dict(state_dict, strict=False)
        
        # Set model to evaluation mode
        model.eval()
        
        logging.info(f"Successfully loaded model from {checkpoint_path}")
        if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
            logging.info(f"Model was trained for {checkpoint['epoch']} epochs")
        if isinstance(checkpoint, dict) and 'val_acc' in checkpoint:
            logging.info(f"Best validation accuracy: {checkpoint['val_acc']:.2f}%")
        
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise

def use_model_in_application(model, image_path, text):
    """
    Use the trained model for inference in an application
    Args:
        model: Trained model
        image_path: Path to the image file
        text: Text content of the meme
    Returns:
        dict: Prediction results with confidence scores
    """
    try:
        # Move model to appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Get prediction
        result = model.predict(image_path, text)
        
        return {
            'is_hateful': result['hateful'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities'],
            'timestamp': result['timestamp']
        }
    except Exception as e:
        logging.error(f"Error during inference: {str(e)}")
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

# Example usage in application
def example_application_usage():
    """
    Example of how to use the model in an application
    """
    try:
        # Load the trained model
        model = load_trained_model()
        
        # Example image and text
        image_path = "path/to/your/meme.jpg"
        text = "Example meme text"
        
        # Get prediction
        result = use_model_in_application(model, image_path, text)
        
        # Process results
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Is Hateful: {result['is_hateful']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Probabilities: {result['probabilities']}")
            print(f"Timestamp: {result['timestamp']}")
            
    except Exception as e:
        print(f"Application error: {str(e)}")

# Make sure the function is available for import
__all__ = ['AdvancedHatefulMemeDetector', 'load_trained_model']

def main():
    # Create necessary directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Verify required files exist
    required_files = ['data/train.jsonl', 'data/dev.jsonl']
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    try:
        # Load the trained model
        logging.info("Loading trained model...")
        model = load_trained_model('checkpoints/best_model.pt')
        logging.info("Model loaded successfully")
        
        # Load data for training
        logging.info("Loading training data...")
        train_data = load_jsonl('data/train.jsonl')
        dev_data = load_jsonl('data/dev.jsonl')
        
        if not train_data or not dev_data:
            raise ValueError("No data loaded from JSONL files")
        
        # Create datasets
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = MemeDataset('data/train.jsonl', 'data/img', transform=transform)
        val_dataset = MemeDataset('data/dev.jsonl', 'data/img', transform=transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=get_optimal_batch_size(), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=get_optimal_batch_size(), shuffle=False)
        
        # Resume training with more epochs
        logging.info("Resuming training with additional epochs...")
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=10,  # Increased from 6 to 10 epochs
            learning_rate=2e-5,
            resume_training=True,
            checkpoint_path='checkpoints/best_model.pt'
        )
        
        logging.info("Training completed successfully")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    # Set multiprocessing start method for Windows
    if sys.platform == 'win32':
        mp.set_start_method('spawn', force=True)
    main()
    
    # For using in application
    # example_application_usage() 