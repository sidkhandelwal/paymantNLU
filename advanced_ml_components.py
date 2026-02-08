"""
Advanced Payment Entity Extraction System
Includes: ML Intent Classification, NER, Multi-language, Fuzzy Matching, Date Parsing
"""

import re
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict

# Note: In production, these imports would be used:
# import spacy
# from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# import dateparser
# from rapidfuzz import fuzz, process
# from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker


class IntentType(Enum):
    """Payment intent types"""
    MAKE_PAYMENT = "make_payment"
    TRANSFER = "transfer"
    FETCH_TRANSACTION = "fetch_transaction"
    GET_STATUS = "get_status"
    CANCEL_PAYMENT = "cancel_payment"
    SCHEDULE_PAYMENT = "schedule_payment"
    RECURRING_PAYMENT = "recurring_payment"
    UNKNOWN = "unknown"


class PaymentStatus(Enum):
    """Payment status types"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PaymentEntity:
    """Stores extracted entities from payment commands"""
    intent: IntentType = IntentType.UNKNOWN
    amount: Optional[float] = None
    currency: Optional[str] = None
    recipient: Optional[str] = None
    source_account: Optional[str] = None
    payment_method: Optional[str] = None
    transaction_id: Optional[str] = None
    date: Optional[datetime] = None
    count: Optional[int] = None
    raw_text: str = ""
    language: str = "en"
    confidence: float = 0.0
    missing_slots: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['intent'] = self.intent.value
        data['date'] = self.date.isoformat() if self.date else None
        return data


class MLIntentClassifier:
    """
    Machine Learning based Intent Classifier
    Uses TF-IDF + Naive Bayes (lightweight) or BERT (production)
    """
    
    def __init__(self, use_transformer: bool = False):
        self.use_transformer = use_transformer
        self.vectorizer = None
        self.classifier = None
        self.label_encoder = {}
        self.is_trained = False
        
        # Training data for intent classification
        self.training_data = self._get_training_data()
        
        if not use_transformer:
            self._train_sklearn_model()
    
    def _get_training_data(self) -> List[Tuple[str, IntentType]]:
        """Get training data for intent classification"""
        return [
            # Make Payment
            ("make a payment to John", IntentType.MAKE_PAYMENT),
            ("pay 100 USD to Alice", IntentType.MAKE_PAYMENT),
            ("send money to Bob", IntentType.MAKE_PAYMENT),
            ("I want to pay someone", IntentType.MAKE_PAYMENT),
            ("process payment", IntentType.MAKE_PAYMENT),
            
            # Transfer
            ("transfer 500 from savings", IntentType.TRANSFER),
            ("move money between accounts", IntentType.TRANSFER),
            ("transfer funds to checking", IntentType.TRANSFER),
            ("internal transfer", IntentType.TRANSFER),
            
            # Fetch Transaction
            ("show my transactions", IntentType.FETCH_TRANSACTION),
            ("get transaction history", IntentType.FETCH_TRANSACTION),
            ("fetch last 10 payments", IntentType.FETCH_TRANSACTION),
            ("list recent transactions", IntentType.FETCH_TRANSACTION),
            ("show payment history", IntentType.FETCH_TRANSACTION),
            
            # Get Status
            ("check payment status", IntentType.GET_STATUS),
            ("what is the status of transaction", IntentType.GET_STATUS),
            ("get swift status", IntentType.GET_STATUS),
            ("track my payment", IntentType.GET_STATUS),
            
            # Cancel Payment
            ("cancel the payment", IntentType.CANCEL_PAYMENT),
            ("abort transaction", IntentType.CANCEL_PAYMENT),
            ("stop the payment", IntentType.CANCEL_PAYMENT),
            
            # Schedule Payment
            ("schedule a payment for tomorrow", IntentType.SCHEDULE_PAYMENT),
            ("set up future payment", IntentType.SCHEDULE_PAYMENT),
            
            # Recurring Payment
            ("set up monthly payment", IntentType.RECURRING_PAYMENT),
            ("create recurring transfer", IntentType.RECURRING_PAYMENT),
        ]
    
    def _train_sklearn_model(self):
        """Train scikit-learn model (fallback when transformers not available)"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.naive_bayes import MultinomialNB
            
            texts = [text for text, _ in self.training_data]
            labels = [intent.value for _, intent in self.training_data]
            
            # Create label encoder
            unique_labels = list(set(labels))
            self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
            self.reverse_label_encoder = {idx: label for label, idx in self.label_encoder.items()}
            
            # Encode labels
            encoded_labels = [self.label_encoder[label] for label in labels]
            
            # Train vectorizer and classifier
            self.vectorizer = TfidfVectorizer(max_features=100)
            X = self.vectorizer.fit_transform(texts)
            
            self.classifier = MultinomialNB()
            self.classifier.fit(X, encoded_labels)
            
            self.is_trained = True
        except ImportError:
            print("scikit-learn not available, using rule-based classification")
            self.is_trained = False
    
    def predict(self, text: str) -> Tuple[IntentType, float]:
        """Predict intent using ML model"""
        if self.is_trained and self.classifier is not None:
            try:
                X = self.vectorizer.transform([text])
                prediction = self.classifier.predict(X)[0]
                probabilities = self.classifier.predict_proba(X)[0]
                confidence = max(probabilities)
                
                intent_str = self.reverse_label_encoder[prediction]
                intent = IntentType(intent_str)
                
                return intent, confidence
            except Exception as e:
                print(f"ML prediction failed: {e}, falling back to rules")
        
        # Fallback to rule-based
        return self._rule_based_predict(text), 0.5
    
    def _rule_based_predict(self, text: str) -> IntentType:
        """Rule-based fallback"""
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ['cancel', 'abort', 'stop']):
            return IntentType.CANCEL_PAYMENT
        if any(kw in text_lower for kw in ['schedule', 'future', 'later']):
            return IntentType.SCHEDULE_PAYMENT
        if any(kw in text_lower for kw in ['recurring', 'monthly', 'weekly', 'daily']):
            return IntentType.RECURRING_PAYMENT
        if 'swift' in text_lower or ('status' in text_lower and 'transaction' in text_lower):
            return IntentType.GET_STATUS
        if any(kw in text_lower for kw in ['fetch', 'list', 'show', 'history']):
            return IntentType.FETCH_TRANSACTION
        if 'transfer' in text_lower:
            return IntentType.TRANSFER
        if any(kw in text_lower for kw in ['pay', 'payment', 'send']):
            return IntentType.MAKE_PAYMENT
        
        return IntentType.UNKNOWN


class NERExtractor:
    """
    Named Entity Recognition using spaCy or Transformers
    Extracts: PERSON (recipient), MONEY, DATE, ORG (banks), etc.
    """
    
    def __init__(self, use_transformer: bool = False):
        self.use_transformer = use_transformer
        self.nlp = None
        self.ner_pipeline = None
        
        # In production, load models:
        # if use_transformer:
        #     self.ner_pipeline = pipeline("ner", model="dslim/bert-base-NER")
        # else:
        #     self.nlp = spacy.load("en_core_web_sm")
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        entities = {
            'PERSON': [],
            'MONEY': [],
            'DATE': [],
            'ORG': [],
            'CARDINAL': []
        }
        
        # Since we can't load real models, use enhanced regex
        return self._regex_ner(text)

    def extract_slots(self, text: str) -> Dict[str, Optional[str]]:
        """Extract slot values from text using lightweight NLU heuristics."""
        slots: Dict[str, Optional[str]] = {
            "amount": None,
            "currency": None,
            "recipient": None,
            "source_account": None,
            "payment_method": None,
            "transaction_id": None,
            "date": None,
            "count": None
        }

        money_pattern = r'(\d+(?:\.\d{2})?)\s*(USD|EUR|GBP|INR|dollars?|euros?|pounds?|rupees?)'
        money_match = re.search(money_pattern, text, re.IGNORECASE)
        if money_match:
            slots["amount"] = money_match.group(1)
            slots["currency"] = money_match.group(2)

        recipient_pattern = r'(?:to|recipient|beneficiary)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
        recipient_match = re.search(recipient_pattern, text)
        if recipient_match:
            slots["recipient"] = recipient_match.group(1)

        account_pattern = r'(?:from|account)\s+([A-Z][a-z]+\s+[A-Z][a-z]+|Account\s+\d+|\d{7,})'
        account_match = re.search(account_pattern, text, re.IGNORECASE)
        if account_match:
            slots["source_account"] = account_match.group(1)

        method_pattern = r'(?:using|via|through)\s+([A-Z][a-z]+(?:Payment)?)'
        method_match = re.search(method_pattern, text, re.IGNORECASE)
        if method_match:
            slots["payment_method"] = method_match.group(1)

        transaction_pattern = r'(?:Payment\s+|transaction\s+|ID\s+)?([A-Z]{2}\s*\d{9,})'
        transaction_match = re.search(transaction_pattern, text, re.IGNORECASE)
        if transaction_match:
            slots["transaction_id"] = transaction_match.group(1).strip()

        date_pattern = r'(tomorrow|today|yesterday|next\s+\w+|\d{4}-\d{2}-\d{2})'
        date_match = re.search(date_pattern, text, re.IGNORECASE)
        if date_match:
            slots["date"] = date_match.group(1)

        count_pattern = r'(?:last|recent)\s+(\d+)'
        count_match = re.search(count_pattern, text, re.IGNORECASE)
        if count_match:
            slots["count"] = count_match.group(1)

        return slots
    
    def _regex_ner(self, text: str) -> Dict[str, List[str]]:
        """Regex-based NER as fallback"""
        entities = defaultdict(list)
        
        # Money
        money_pattern = r'(\d+(?:\.\d{2})?)\s*(USD|EUR|GBP|INR|dollars?|euros?)'
        for match in re.finditer(money_pattern, text, re.IGNORECASE):
            entities['MONEY'].append(match.group(0))
        
        # Person names (capitalized words after 'to')
        person_pattern = r'(?:to|recipient)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
        for match in re.finditer(person_pattern, text):
            entities['PERSON'].append(match.group(1))
        
        # Dates
        date_pattern = r'\b(tomorrow|today|yesterday|\d{4}-\d{2}-\d{2})\b'
        for match in re.finditer(date_pattern, text, re.IGNORECASE):
            entities['DATE'].append(match.group(1))
        
        # Cardinals
        cardinal_pattern = r'\b(\d+)\b'
        for match in re.finditer(cardinal_pattern, text):
            entities['CARDINAL'].append(match.group(1))
        
        return dict(entities)


class MultiLanguageTranslator:
    """
    Multi-language support for payment commands
    Supports: English, Spanish, French, German, Hindi
    """
    
    def __init__(self):
        # Translation dictionaries for common payment terms
        self.translations = {
            'es': {  # Spanish
                'pagar': 'pay',
                'transferir': 'transfer',
                'enviar': 'send',
                'dinero': 'money',
                'cuenta': 'account',
                'estado': 'status',
                'transacción': 'transaction',
            },
            'fr': {  # French
                'payer': 'pay',
                'transférer': 'transfer',
                'envoyer': 'send',
                'argent': 'money',
                'compte': 'account',
                'statut': 'status',
                'transaction': 'transaction',
            },
            'de': {  # German
                'bezahlen': 'pay',
                'überweisen': 'transfer',
                'senden': 'send',
                'geld': 'money',
                'konto': 'account',
                'status': 'status',
                'transaktion': 'transaction',
            },
            'hi': {  # Hindi (transliterated)
                'bhugtan': 'payment',
                'transfer': 'transfer',
                'bhejein': 'send',
                'paisa': 'money',
                'khata': 'account',
            }
        }
    
    def detect_language(self, text: str) -> str:
        """Detect language of input text"""
        # Simple detection based on keywords
        text_lower = text.lower()
        
        for lang, translations in self.translations.items():
            if any(word in text_lower for word in translations.keys()):
                return lang
        
        return 'en'
    
    def translate_to_english(self, text: str, source_lang: str = None) -> str:
        """Translate text to English"""
        if source_lang is None:
            source_lang = self.detect_language(text)
        
        if source_lang == 'en':
            return text
        
        translated = text
        if source_lang in self.translations:
            for foreign_word, english_word in self.translations[source_lang].items():
                translated = re.sub(
                    r'\b' + foreign_word + r'\b',
                    english_word,
                    translated,
                    flags=re.IGNORECASE
                )
        
        return translated


class FuzzyMatcher:
    """
    Fuzzy matching for names, accounts, and payment methods
    Handles typos and variations
    """
    
    def __init__(self):
        # Known entities database
        self.known_recipients = [
            "John Smith", "Alice Johnson", "Bob Williams", "Charlie Brown",
            "RamKrishna", "Ramesh", "David Miller", "Emma Davis"
        ]
        
        self.known_accounts = [
            "Salary Account", "Savings Account", "Checking Account",
            "Business Account", "Investment Account"
        ]
        
        self.known_payment_methods = [
            "FasterPayment", "SWIFT", "ACH", "Wire Transfer",
            "SEPA", "NEFT", "RTGS", "IMPS"
        ]
    
    def fuzzy_match_recipient(self, query: str, threshold: int = 80) -> Optional[Tuple[str, int]]:
        """Find best matching recipient using fuzzy matching"""
        return self._fuzzy_match(query, self.known_recipients, threshold)
    
    def fuzzy_match_account(self, query: str, threshold: int = 80) -> Optional[Tuple[str, int]]:
        """Find best matching account"""
        return self._fuzzy_match(query, self.known_accounts, threshold)
    
    def fuzzy_match_payment_method(self, query: str, threshold: int = 80) -> Optional[Tuple[str, int]]:
        """Find best matching payment method"""
        return self._fuzzy_match(query, self.known_payment_methods, threshold)
    
    def _fuzzy_match(self, query: str, choices: List[str], threshold: int) -> Optional[Tuple[str, int]]:
        """Generic fuzzy matching using Levenshtein distance"""
        try:
            # In production, use rapidfuzz:
            # from rapidfuzz import process
            # result = process.extractOne(query, choices, score_cutoff=threshold)
            # return result if result else None
            
            # Fallback: simple similarity
            best_match = None
            best_score = 0
            
            for choice in choices:
                score = self._simple_similarity(query.lower(), choice.lower())
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = choice
            
            return (best_match, int(best_score)) if best_match else None
        except:
            return None
    
    def _simple_similarity(self, s1: str, s2: str) -> float:
        """Simple similarity score (0-100)"""
        if s1 == s2:
            return 100.0
        
        # Jaccard similarity on character bigrams
        bigrams1 = set(s1[i:i+2] for i in range(len(s1)-1))
        bigrams2 = set(s2[i:i+2] for i in range(len(s2)-1))
        
        if not bigrams1 or not bigrams2:
            return 0.0
        
        intersection = bigrams1.intersection(bigrams2)
        union = bigrams1.union(bigrams2)
        
        return (len(intersection) / len(union)) * 100


class AdvancedDateParser:
    """
    Advanced date parsing supporting natural language
    Uses dateparser library for complex date expressions
    """
    
    def __init__(self):
        self.supported_languages = ['en', 'es', 'fr', 'de', 'hi']
    
    def parse(self, date_string: str, language: str = 'en') -> Optional[datetime]:
        """Parse date string to datetime object"""
        # In production:
        # import dateparser
        # return dateparser.parse(date_string, languages=[language])
        
        # Fallback implementation
        return self._simple_parse(date_string)
    
    def _simple_parse(self, date_string: str) -> Optional[datetime]:
        """Simple date parsing fallback"""
        date_string_lower = date_string.lower().strip()
        now = datetime.now()
        
        # Relative dates
        if date_string_lower == 'today':
            return now
        elif date_string_lower == 'tomorrow':
            return now.replace(hour=0, minute=0, second=0, microsecond=0) + \
                   __import__('datetime').timedelta(days=1)
        elif date_string_lower == 'yesterday':
            return now.replace(hour=0, minute=0, second=0, microsecond=0) - \
                   __import__('datetime').timedelta(days=1)
        
        # ISO format
        try:
            return datetime.fromisoformat(date_string)
        except:
            pass
        
        # Common formats
        formats = [
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%d-%m-%Y',
            '%m-%d-%Y'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_string, fmt)
            except:
                continue
        
        return None


class PaymentDatabase:
    """
    Database integration for transaction storage
    Uses SQLAlchemy with SQLite (can be switched to PostgreSQL/MySQL)
    """
    
    def __init__(self, db_url: str = "sqlite:///payments.db"):
        self.db_url = db_url
        self.engine = None
        self.Session = None
        
        # In production:
        # from sqlalchemy import create_engine
        # from sqlalchemy.orm import sessionmaker
        # self.engine = create_engine(db_url)
        # self.Session = sessionmaker(bind=self.engine)
        # Base.metadata.create_all(self.engine)
        
        # For demo, use simple dict storage
        self.transactions = {}
        self.transaction_counter = 1
    
    def save_transaction(self, entity: PaymentEntity) -> str:
        """Save transaction to database"""
        transaction_id = f"TXN{self.transaction_counter:08d}"
        self.transaction_counter += 1
        
        self.transactions[transaction_id] = {
            'id': transaction_id,
            'intent': entity.intent.value,
            'amount': entity.amount,
            'currency': entity.currency,
            'recipient': entity.recipient,
            'source_account': entity.source_account,
            'payment_method': entity.payment_method,
            'date': entity.date.isoformat() if entity.date else None,
            'status': PaymentStatus.PENDING.value,
            'created_at': datetime.now().isoformat(),
            'metadata': entity.metadata
        }
        
        return transaction_id
    
    def get_transaction(self, transaction_id: str) -> Optional[Dict]:
        """Retrieve transaction by ID"""
        return self.transactions.get(transaction_id)
    
    def get_recent_transactions(self, count: int = 10) -> List[Dict]:
        """Get recent transactions"""
        sorted_txns = sorted(
            self.transactions.values(),
            key=lambda x: x['created_at'],
            reverse=True
        )
        return sorted_txns[:count]
    
    def update_transaction_status(self, transaction_id: str, status: PaymentStatus) -> bool:
        """Update transaction status"""
        if transaction_id in self.transactions:
            self.transactions[transaction_id]['status'] = status.value
            return True
        return False


class PaymentAPIClient:
    """
    Payment API integration for real payment processing
    Simulates calls to payment gateways (Stripe, PayPal, bank APIs, etc.)
    """
    
    def __init__(self, api_key: str = "demo_key", environment: str = "sandbox"):
        self.api_key = api_key
        self.environment = environment
        self.base_url = "https://api.payment-gateway.com" if environment == "production" else \
                        "https://sandbox.payment-gateway.com"
    
    def process_payment(self, entity: PaymentEntity) -> Dict[str, Any]:
        """Process payment through API"""
        # In production, make actual API call:
        # import requests
        # response = requests.post(
        #     f"{self.base_url}/v1/payments",
        #     headers={"Authorization": f"Bearer {self.api_key}"},
        #     json={
        #         "amount": entity.amount,
        #         "currency": entity.currency,
        #         "recipient": entity.recipient,
        #         "source": entity.source_account,
        #         "method": entity.payment_method
        #     }
        # )
        # return response.json()
        
        # Simulation
        import random
        success = random.choice([True, True, True, False])  # 75% success rate
        
        return {
            "success": success,
            "transaction_id": f"API_{random.randint(100000, 999999)}",
            "status": "completed" if success else "failed",
            "message": "Payment processed successfully" if success else "Payment failed",
            "timestamp": datetime.now().isoformat()
        }
    
    def check_status(self, transaction_id: str) -> Dict[str, Any]:
        """Check payment status"""
        # Simulation
        import random
        statuses = ["pending", "processing", "completed", "failed"]
        
        return {
            "transaction_id": transaction_id,
            "status": random.choice(statuses),
            "last_updated": datetime.now().isoformat()
        }
    
    def cancel_payment(self, transaction_id: str) -> Dict[str, Any]:
        """Cancel a pending payment"""
        return {
            "transaction_id": transaction_id,
            "status": "cancelled",
            "message": "Payment cancelled successfully"
        }


class ContextManager:
    """Enhanced context manager with persistence"""
    
    def __init__(self, db: Optional[PaymentDatabase] = None):
        self.history: List[PaymentEntity] = []
        self.current_entity: Optional[PaymentEntity] = None
        self.pending_slots: List[str] = []
        self.db = db
    
    def add_to_history(self, entity: PaymentEntity):
        """Add entity to conversation history"""
        self.history.append(entity)
        
        # Persist to database if available
        if self.db and entity.intent != IntentType.UNKNOWN:
            self.db.save_transaction(entity)
    
    def get_last_entity(self) -> Optional[PaymentEntity]:
        """Get the most recent entity"""
        return self.history[-1] if self.history else None
    
    def fill_from_context(self, entity: PaymentEntity) -> PaymentEntity:
        """Fill missing slots from conversation history"""
        if not self.history:
            return entity
        
        last_entity = self.get_last_entity()
        
        # Fill missing slots from previous context
        if entity.currency is None and last_entity.currency:
            entity.currency = last_entity.currency
        if entity.source_account is None and last_entity.source_account:
            entity.source_account = last_entity.source_account
        if entity.payment_method is None and last_entity.payment_method:
            entity.payment_method = last_entity.payment_method
        
        return entity

    def reset(self):
        """Clear conversation history and pending context."""
        self.history = []
        self.current_entity = None
        self.pending_slots = []


# Export all classes
__all__ = [
    'IntentType', 'PaymentStatus', 'PaymentEntity',
    'MLIntentClassifier', 'NERExtractor', 'MultiLanguageTranslator',
    'FuzzyMatcher', 'AdvancedDateParser', 'PaymentDatabase',
    'PaymentAPIClient', 'ContextManager'
]
