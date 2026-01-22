"""
Transformer-based Models for Financial Text Analysis

This module implements transformer-based models for financial text analysis,
including sentiment analysis and news classification using various transformer architectures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime
import warnings

# Transformer libraries
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Transformer models will not work.")

try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        RobertaTokenizer, RobertaModel, RobertaForSequenceClassification,
        DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification,
        TrainingArguments, Trainer, pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers library not available. Transformer models will not work.")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Some features will be limited.")

logger = logging.getLogger(__name__)

class TransformerSentimentModel:
    """
    Transformer-based sentiment analysis model for financial text.
    """
    
    def __init__(
        self,
        model_name: str = "roberta-base",
        num_labels: int = 3,
        max_length: int = 512,
        device: str = "auto"
    ):
        """
        Initialize transformer sentiment model.
        
        Args:
            model_name: Transformer model name
            num_labels: Number of sentiment labels
            max_length: Maximum sequence length
            device: Device to use for training/inference
        """
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            raise ImportError("PyTorch and Transformers libraries are required for transformer models")
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.model.to(self.device)
        
        # Label mapping
        self.label_mapping = {
            0: 'negative',
            1: 'neutral',
            2: 'positive'
        }
        
        # Training history
        self.training_history = []
        
        logger.info(f"Initialized Transformer Sentiment Model: {model_name} on {self.device}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for transformer input.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Basic text cleaning
        text = text.strip()
        
        # Truncate if too long
        if len(text) > self.max_length * 4:  # Rough character to token ratio
            text = text[:self.max_length * 4]
        
        return text
    
    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Sentiment prediction results
        """
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Tokenize
        inputs = self.tokenizer(
            processed_text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1)
        
        # Convert to CPU and extract values
        probabilities = probabilities.cpu().numpy()[0]
        predicted_class = predicted_class.cpu().numpy()[0]
        
        # Get confidence
        confidence = float(probabilities[predicted_class])
        
        # Get sentiment label
        sentiment_label = self.label_mapping[predicted_class]
        
        return {
            'text': text,
            'processed_text': processed_text,
            'sentiment_label': sentiment_label,
            'sentiment_score': float(probabilities[2] - probabilities[0]),  # positive - negative
            'confidence': confidence,
            'probabilities': {
                'negative': float(probabilities[0]),
                'neutral': float(probabilities[1]),
                'positive': float(probabilities[2])
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of sentiment predictions
        """
        results = []
        
        for text in texts:
            result = self.predict_sentiment(text)
            results.append(result)
        
        return results
    
    def train(
        self,
        train_texts: List[str],
        train_labels: List[Union[str, int]],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[Union[str, int]]] = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500
    ) -> Dict[str, Any]:
        """
        Train the transformer model.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            
        Returns:
            Training results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for training")
        
        logger.info("Starting transformer model training")
        
        # Encode labels
        label_encoder = LabelEncoder()
        train_labels_encoded = label_encoder.fit_transform(train_labels)
        
        if val_texts and val_labels:
            val_labels_encoded = label_encoder.transform(val_labels)
        else:
            val_texts = None
            val_labels_encoded = None
        
        # Create datasets
        train_dataset = FinancialNewsDataset(
            train_texts, train_labels_encoded, self.tokenizer, self.max_length
        )
        
        if val_texts:
            val_dataset = FinancialNewsDataset(
                val_texts, val_labels_encoded, self.tokenizer, self.max_length
            )
        else:
            val_dataset = None
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./transformer_sentiment_model',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="steps" if val_dataset else "no",
            eval_steps=100 if val_dataset else None,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True if val_dataset else False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train
        training_result = trainer.train()
        
        # Store training history
        self.training_history.append({
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'train_loss': training_result.training_loss,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Training completed. Final loss: {training_result.training_loss}")
        
        return {
            'training_loss': training_result.training_loss,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'timestamp': datetime.now().isoformat()
        }
    
    def evaluate(self, test_texts: List[str], test_labels: List[Union[str, int]]) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            test_texts: Test texts
            test_labels: Test labels
            
        Returns:
            Evaluation results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for evaluation")
        
        # Predict on test data
        predictions = self.predict_batch(test_texts)
        predicted_labels = [p['sentiment_label'] for p in predictions]
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predicted_labels)
        
        # Classification report
        report = classification_report(test_labels, predicted_labels, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, predicted_labels)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': predictions
        }
    
    def save_model(self, save_path: str) -> None:
        """
        Save the trained model.
        
        Args:
            save_path: Path to save the model
        """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str) -> None:
        """
        Load a trained model.
        
        Args:
            load_path: Path to load the model from
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model.to(self.device)
        
        logger.info(f"Model loaded from {load_path}")


class TransformerNewsModel:
    """
    Transformer-based news classification model for financial news.
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 7,
        max_length: int = 512,
        device: str = "auto"
    ):
        """
        Initialize transformer news model.
        
        Args:
            model_name: Transformer model name
            num_labels: Number of news categories
            max_length: Maximum sequence length
            device: Device to use for training/inference
        """
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            raise ImportError("PyTorch and Transformers libraries are required for transformer models")
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.model.to(self.device)
        
        # Financial news categories
        self.category_mapping = {
            0: 'earnings',
            1: 'mergers',
            2: 'ipo',
            3: 'dividend',
            4: 'analyst',
            5: 'regulatory',
            6: 'market'
        }
        
        # Training history
        self.training_history = []
        
        logger.info(f"Initialized Transformer News Model: {model_name} on {self.device}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for transformer input.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Basic text cleaning
        text = text.strip()
        
        # Truncate if too long
        if len(text) > self.max_length * 4:  # Rough character to token ratio
            text = text[:self.max_length * 4]
        
        return text
    
    def classify_news(self, text: str) -> Dict[str, Any]:
        """
        Classify news article into categories.
        
        Args:
            text: Input text
            
        Returns:
            News classification results
        """
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Tokenize
        inputs = self.tokenizer(
            processed_text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1)
        
        # Convert to CPU and extract values
        probabilities = probabilities.cpu().numpy()[0]
        predicted_class = predicted_class.cpu().numpy()[0]
        
        # Get confidence
        confidence = float(probabilities[predicted_class])
        
        # Get category label
        category_label = self.category_mapping[predicted_class]
        
        # Get top categories
        top_categories = []
        for i, prob in enumerate(probabilities):
            top_categories.append({
                'category': self.category_mapping[i],
                'probability': float(prob)
            })
        
        # Sort by probability
        top_categories.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            'text': text,
            'processed_text': processed_text,
            'category_label': category_label,
            'confidence': confidence,
            'top_categories': top_categories,
            'timestamp': datetime.now().isoformat()
        }
    
    def classify_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Classify a batch of news articles.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of classification results
        """
        results = []
        
        for text in texts:
            result = self.classify_news(text)
            results.append(result)
        
        return results
    
    def train(
        self,
        train_texts: List[str],
        train_labels: List[Union[str, int]],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[Union[str, int]]] = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500
    ) -> Dict[str, Any]:
        """
        Train the transformer news model.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            
        Returns:
            Training results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for training")
        
        logger.info("Starting transformer news model training")
        
        # Encode labels
        label_encoder = LabelEncoder()
        train_labels_encoded = label_encoder.fit_transform(train_labels)
        
        if val_texts and val_labels:
            val_labels_encoded = label_encoder.transform(val_labels)
        else:
            val_texts = None
            val_labels_encoded = None
        
        # Create datasets
        train_dataset = FinancialNewsDataset(
            train_texts, train_labels_encoded, self.tokenizer, self.max_length
        )
        
        if val_texts:
            val_dataset = FinancialNewsDataset(
                val_texts, val_labels_encoded, self.tokenizer, self.max_length
            )
        else:
            val_dataset = None
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./transformer_news_model',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="steps" if val_dataset else "no",
            eval_steps=100 if val_dataset else None,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True if val_dataset else False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train
        training_result = trainer.train()
        
        # Store training history
        self.training_history.append({
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'train_loss': training_result.training_loss,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Training completed. Final loss: {training_result.training_loss}")
        
        return {
            'training_loss': training_result.training_loss,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'timestamp': datetime.now().isoformat()
        }
    
    def evaluate(self, test_texts: List[str], test_labels: List[Union[str, int]]) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            test_texts: Test texts
            test_labels: Test labels
            
        Returns:
            Evaluation results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for evaluation")
        
        # Predict on test data
        predictions = self.classify_batch(test_texts)
        predicted_labels = [p['category_label'] for p in predictions]
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predicted_labels)
        
        # Classification report
        report = classification_report(test_labels, predicted_labels, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, predicted_labels)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': predictions
        }
    
    def save_model(self, save_path: str) -> None:
        """
        Save the trained model.
        
        Args:
            save_path: Path to save the model
        """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str) -> None:
        """
        Load a trained model.
        
        Args:
            load_path: Path to load the model from
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model.to(self.device)
        
        logger.info(f"Model loaded from {load_path}")


class FinancialNewsDataset:
    """
    Dataset class for financial news data.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[Union[str, int]],
        tokenizer,
        max_length: int = 512
    ):
        """
        Initialize dataset.
        
        Args:
            texts: List of text samples
            labels: List of labels
            tokenizer: Transformer tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

