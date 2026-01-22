"""
Text Classification Models for Financial News and Market Intelligence

This module implements text classification models for categorizing
financial news and market-related text data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime
import re
from collections import Counter, defaultdict
import warnings

# Machine learning libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Text classification features will be limited.")

logger = logging.getLogger(__name__)

class TextClassifier:
    """
    Base text classifier for financial text data.
    """
    
    def __init__(
        self,
        model_name: str = "text_classifier",
        classifier_type: str = "logistic_regression",
        vectorizer_type: str = "tfidf",
        max_features: int = 10000
    ):
        """
        Initialize text classifier.
        
        Args:
            model_name: Name for the classifier
            classifier_type: Type of classifier to use
            vectorizer_type: Type of vectorizer to use
            max_features: Maximum number of features
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for text classification")
        
        self.model_name = model_name
        self.classifier_type = classifier_type
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        
        # Initialize vectorizer
        if vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        elif vectorizer_type == "count":
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        else:
            raise ValueError(f"Unsupported vectorizer type: {vectorizer_type}")
        
        # Initialize classifier
        if classifier_type == "logistic_regression":
            self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        elif classifier_type == "naive_bayes":
            self.classifier = MultinomialNB()
        elif classifier_type == "svm":
            self.classifier = SVC(kernel='linear', random_state=42, probability=True)
        elif classifier_type == "random_forest":
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.classifier)
        ])
        
        # Label encoder
        self.label_encoder = LabelEncoder()
        
        # Training history
        self.training_history = []
        
        # Classification history
        self.classification_history = []
        
        logger.info(f"Initialized Text Classifier: {model_name} ({classifier_type})")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for classification.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def train(
        self,
        texts: List[str],
        labels: List[Union[str, int]],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train the text classifier.
        
        Args:
            texts: Training texts
            labels: Training labels
            test_size: Test set size
            random_state: Random state for reproducibility
            
        Returns:
            Training results
        """
        logger.info(f"Training {self.model_name} with {len(texts)} samples")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, encoded_labels,
            test_size=test_size, random_state=random_state, stratify=encoded_labels
        )
        
        # Train pipeline
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.pipeline, processed_texts, encoded_labels, cv=5)
        
        # Store training history
        training_result = {
            'model_name': self.model_name,
            'classifier_type': self.classifier_type,
            'vectorizer_type': self.vectorizer_type,
            'training_samples': len(texts),
            'test_accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_history.append(training_result)
        
        logger.info(f"Training completed. Test accuracy: {accuracy:.4f}")
        
        return training_result
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict class for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Prediction results
        """
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Predict
        prediction = self.pipeline.predict([processed_text])[0]
        probabilities = self.pipeline.predict_proba([processed_text])[0]
        
        # Get class label
        class_label = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get confidence
        confidence = float(probabilities[prediction])
        
        # Get all class probabilities
        class_probabilities = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            class_probabilities[class_name] = float(probabilities[i])
        
        result = {
            'text': text,
            'processed_text': processed_text,
            'predicted_class': class_label,
            'confidence': confidence,
            'class_probabilities': class_probabilities,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in history
        self.classification_history.append(result)
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict classes for a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction results
        """
        results = []
        
        for text in texts:
            result = self.predict(text)
            results.append(result)
        
        return results
    
    def evaluate(self, test_texts: List[str], test_labels: List[Union[str, int]]) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            test_texts: Test texts
            test_labels: Test labels
            
        Returns:
            Evaluation results
        """
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in test_texts]
        
        # Encode labels
        encoded_labels = self.label_encoder.transform(test_labels)
        
        # Predict
        predictions = self.pipeline.predict(processed_texts)
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        # Calculate metrics
        accuracy = accuracy_score(encoded_labels, predictions)
        
        # Classification report
        report = classification_report(encoded_labels, predictions, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(encoded_labels, predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': predicted_labels.tolist()
        }
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get feature importance for each class.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature importance by class
        """
        if not hasattr(self.classifier, 'coef_'):
            logger.warning("Feature importance not available for this classifier type")
            return {}
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get coefficients
        coefficients = self.classifier.coef_
        
        feature_importance = {}
        
        for i, class_name in enumerate(self.label_encoder.classes_):
            # Get coefficients for this class
            class_coef = coefficients[i]
            
            # Create feature-coefficient pairs
            feature_coef_pairs = list(zip(feature_names, class_coef))
            
            # Sort by coefficient value
            feature_coef_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Get top features
            top_features = feature_coef_pairs[:top_n]
            
            feature_importance[class_name] = top_features
        
        return feature_importance
    
    def get_classification_statistics(self) -> Dict[str, Any]:
        """
        Get classification statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.classification_history:
            return {}
        
        # Extract statistics
        predicted_classes = [result['predicted_class'] for result in self.classification_history]
        confidences = [result['confidence'] for result in self.classification_history]
        
        # Count classes
        class_counts = Counter(predicted_classes)
        
        return {
            'total_classifications': len(self.classification_history),
            'class_distribution': dict(class_counts),
            'confidence_stats': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            }
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        import joblib
        
        model_data = {
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder,
            'model_name': self.model_name,
            'classifier_type': self.classifier_type,
            'vectorizer_type': self.vectorizer_type,
            'max_features': self.max_features,
            'training_history': self.training_history
        }
        
        joblib.dump(model_data, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model.
        
        Args:
            filepath: Path to load the model from
        """
        import joblib
        
        model_data = joblib.load(filepath)
        
        self.pipeline = model_data['pipeline']
        self.label_encoder = model_data['label_encoder']
        self.model_name = model_data['model_name']
        self.classifier_type = model_data['classifier_type']
        self.vectorizer_type = model_data['vectorizer_type']
        self.max_features = model_data['max_features']
        self.training_history = model_data['training_history']
        
        logger.info(f"Model loaded from {filepath}")


class NewsClassifier(TextClassifier):
    """
    Specialized news classifier for financial news categorization.
    """
    
    def __init__(
        self,
        model_name: str = "news_classifier",
        classifier_type: str = "logistic_regression",
        vectorizer_type: str = "tfidf",
        max_features: int = 10000
    ):
        """
        Initialize news classifier.
        
        Args:
            model_name: Name for the classifier
            classifier_type: Type of classifier to use
            vectorizer_type: Type of vectorizer to use
            max_features: Maximum number of features
        """
        super().__init__(model_name, classifier_type, vectorizer_type, max_features)
        
        # Financial news categories
        self.financial_categories = {
            'earnings': ['earnings', 'revenue', 'profit', 'loss', 'quarterly', 'annual', 'eps'],
            'mergers': ['merger', 'acquisition', 'takeover', 'buyout', 'deal', 'consolidation'],
            'ipo': ['ipo', 'initial public offering', 'public offering', 'listing', 'going public'],
            'dividend': ['dividend', 'payout', 'distribution', 'yield', 'dividend yield'],
            'analyst': ['analyst', 'rating', 'upgrade', 'downgrade', 'target', 'price target'],
            'regulatory': ['sec', 'regulation', 'compliance', 'investigation', 'fine', 'lawsuit'],
            'market': ['market', 'trading', 'volume', 'price', 'volatility', 'index', 'sector']
        }
        
        # News classification history
        self.news_classification_history = []
        
        logger.info(f"Initialized News Classifier: {model_name}")
    
    def preprocess_news_text(self, text: str) -> str:
        """
        Preprocess news text for classification.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Use base preprocessing
        processed_text = self.preprocess_text(text)
        
        # Additional news-specific preprocessing
        # Remove common news words that don't add value
        news_stopwords = {
            'said', 'says', 'according', 'reported', 'announced', 'statement',
            'company', 'firm', 'corporation', 'inc', 'ltd', 'llc'
        }
        
        words = processed_text.split()
        filtered_words = [word for word in words if word not in news_stopwords]
        
        return ' '.join(filtered_words)
    
    def classify_news(self, text: str) -> Dict[str, Any]:
        """
        Classify news article.
        
        Args:
            text: Input text
            
        Returns:
            News classification results
        """
        # Use specialized preprocessing
        processed_text = self.preprocess_news_text(text)
        
        # Predict using base classifier
        prediction_result = self.predict(processed_text)
        
        # Add news-specific information
        prediction_result['news_category'] = prediction_result['predicted_class']
        prediction_result['news_confidence'] = prediction_result['confidence']
        
        # Calculate category scores based on keyword matching
        category_scores = {}
        text_lower = text.lower()
        
        for category, keywords in self.financial_categories.items():
            score = 0
            for keyword in keywords:
                score += text_lower.count(keyword)
            
            # Normalize by text length
            text_length = len(text.split())
            if text_length > 0:
                score = score / text_length
            
            category_scores[category] = score
        
        prediction_result['category_scores'] = category_scores
        
        # Determine if prediction matches keyword-based analysis
        keyword_based_category = max(category_scores, key=category_scores.get)
        prediction_result['keyword_based_category'] = keyword_based_category
        prediction_result['prediction_consistency'] = (
            prediction_result['news_category'] == keyword_based_category
        )
        
        # Store in news classification history
        self.news_classification_history.append(prediction_result)
        
        return prediction_result
    
    def classify_news_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Classify a batch of news articles.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of news classification results
        """
        results = []
        
        for text in texts:
            result = self.classify_news(text)
            results.append(result)
        
        return results
    
    def get_news_classification_statistics(self) -> Dict[str, Any]:
        """
        Get news classification statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.news_classification_history:
            return {}
        
        # Extract statistics
        news_categories = [result['news_category'] for result in self.news_classification_history]
        news_confidences = [result['news_confidence'] for result in self.news_classification_history]
        prediction_consistencies = [result['prediction_consistency'] for result in self.news_classification_history]
        
        # Count categories
        category_counts = Counter(news_categories)
        
        # Count consistency
        consistency_count = sum(prediction_consistencies)
        consistency_rate = consistency_count / len(prediction_consistencies)
        
        return {
            'total_news_classifications': len(self.news_classification_history),
            'news_category_distribution': dict(category_counts),
            'news_confidence_stats': {
                'mean': np.mean(news_confidences),
                'std': np.std(news_confidences),
                'min': np.min(news_confidences),
                'max': np.max(news_confidences)
            },
            'prediction_consistency_rate': consistency_rate,
            'consistent_predictions': consistency_count,
            'inconsistent_predictions': len(prediction_consistencies) - consistency_count
        }
    
    def get_category_keywords(self, category: str) -> List[str]:
        """
        Get keywords for a specific category.
        
        Args:
            category: News category
            
        Returns:
            List of keywords for the category
        """
        return self.financial_categories.get(category, [])
    
    def add_category_keywords(self, category: str, keywords: List[str]) -> None:
        """
        Add keywords for a specific category.
        
        Args:
            category: News category
            keywords: List of keywords to add
        """
        if category not in self.financial_categories:
            self.financial_categories[category] = []
        
        self.financial_categories[category].extend(keywords)
        
        logger.info(f"Added {len(keywords)} keywords to category {category}")
    
    def remove_category_keywords(self, category: str, keywords: List[str]) -> None:
        """
        Remove keywords from a specific category.
        
        Args:
            category: News category
            keywords: List of keywords to remove
        """
        if category in self.financial_categories:
            for keyword in keywords:
                if keyword in self.financial_categories[category]:
                    self.financial_categories[category].remove(keyword)
            
            logger.info(f"Removed {len(keywords)} keywords from category {category}")
    
    def get_all_categories(self) -> List[str]:
        """
        Get all available categories.
        
        Returns:
            List of all categories
        """
        return list(self.financial_categories.keys())
    
    def get_category_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for all categories.
        
        Returns:
            Category statistics dictionary
        """
        return {
            'total_categories': len(self.financial_categories),
            'categories': list(self.financial_categories.keys()),
            'category_keyword_counts': {
                category: len(keywords) 
                for category, keywords in self.financial_categories.items()
            }
        }

