"""
Enhanced Payment Entity Extraction System
Integrates all advanced features: ML, NER, Multi-language, Fuzzy Matching, Database, API
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from advanced_ml_components import (
    IntentType, PaymentStatus, PaymentEntity,
    MLIntentClassifier, NERExtractor, MultiLanguageTranslator,
    FuzzyMatcher, AdvancedDateParser, PaymentDatabase,
    PaymentAPIClient, ContextManager
)


class EnhancedPaymentExtractor:
    """
    Advanced payment entity extractor with ML and NLP features
    """
    
    def __init__(
        self,
        use_ml: bool = True,
        use_ner: bool = True,
        use_fuzzy: bool = True,
        db_url: str = "sqlite:///payments.db",
        api_key: str = "demo_key"
    ):
        # Initialize components
        self.use_ml = use_ml
        self.use_ner = use_ner
        self.use_fuzzy = use_fuzzy
        
        # ML Intent Classifier
        self.intent_classifier = MLIntentClassifier() if use_ml else None
        
        # Named Entity Recognition
        self.ner_extractor = NERExtractor() if use_ner else None
        
        # Multi-language support
        self.translator = MultiLanguageTranslator()
        
        # Fuzzy matching
        self.fuzzy_matcher = FuzzyMatcher() if use_fuzzy else None
        
        # Date parser
        self.date_parser = AdvancedDateParser()
        
        # Database
        self.database = PaymentDatabase(db_url)
        
        # Payment API client
        self.api_client = PaymentAPIClient(api_key=api_key, environment="sandbox")
        
        # Context manager
        self.context = ContextManager(db=self.database)
        
        # Regex patterns (fallback)
        self.patterns = {
            'amount': r'(\d+(?:\.\d{2})?)\s*(USD|EUR|GBP|INR|dollars?|euros?|pounds?|rupees?)',
            'recipient': r'(?:to|recipient|beneficiary)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:from|using|$)',
            'source_account': r'(?:from|account)\s+([A-Z][a-z]+\s+[A-Z][a-z]+|Account\s+\d+|\d{7,})',
            'payment_method': r'(?:using|via|through)\s+([A-Z][a-z]+(?:Payment)?)',
            'transaction_id': r'(?:Payment\s+|transaction\s+|ID\s+)?([A-Z]{2}\s*\d{9,})',
            'count': r'(?:last|recent)\s+(\d+)',
        }
    
    def parse(self, text: str) -> PaymentEntity:
        """
        Parse payment command with all advanced features
        """
        entity = PaymentEntity(raw_text=text)
        
        # 1. Language Detection & Translation
        detected_lang = self.translator.detect_language(text)
        entity.language = detected_lang
        
        if detected_lang != 'en':
            translated_text = self.translator.translate_to_english(text, detected_lang)
            print(f"[Translation] {detected_lang} → en: {translated_text}")
        else:
            translated_text = text
        
        # 2. ML Intent Classification
        if self.use_ml and self.intent_classifier:
            intent, confidence = self.intent_classifier.predict(translated_text)
            entity.intent = intent
            entity.confidence = confidence
        else:
            entity.intent = self._extract_intent_regex(translated_text)
            entity.confidence = 0.5
        
        # 3. Named Entity Recognition
        if self.use_ner and self.ner_extractor:
            ner_entities = self.ner_extractor.extract_entities(translated_text)
            
            # Use NER results if available
            if ner_entities.get('PERSON'):
                entity.recipient = ner_entities['PERSON'][0]
            if ner_entities.get('MONEY'):
                self._parse_money_from_ner(ner_entities['MONEY'][0], entity)
            if ner_entities.get('DATE'):
                entity.date = self.date_parser.parse(ner_entities['DATE'][0], detected_lang)
        
        # 4. Regex Extraction (fallback or complement to NER)
        if not entity.amount:
            entity.amount, entity.currency = self._extract_amount_currency(translated_text)
        
        if not entity.recipient:
            entity.recipient = self._extract_entity('recipient', translated_text)
        
        entity.source_account = self._extract_entity('source_account', translated_text)
        entity.payment_method = self._extract_entity('payment_method', translated_text)
        entity.transaction_id = self._extract_entity('transaction_id', translated_text)
        
        if not entity.date:
            date_str = self._extract_entity('date', translated_text)
            if date_str:
                entity.date = self.date_parser.parse(date_str, detected_lang)
        
        count_str = self._extract_entity('count', translated_text)
        entity.count = int(count_str) if count_str else None
        
        # 5. Fuzzy Matching for corrections
        if self.use_fuzzy and self.fuzzy_matcher:
            entity = self._apply_fuzzy_matching(entity)
        
        # 6. Fill from context
        entity = self.context.fill_from_context(entity)
        
        # 7. Identify missing slots
        entity.missing_slots = self._identify_missing_slots(entity)
        
        # 8. Add to context
        self.context.add_to_history(entity)
        
        return entity
    
    def process_followup(self, text: str) -> PaymentEntity:
        """Process follow-up message to fill missing slots"""
        entity = PaymentEntity(raw_text=text)
        
        # Detect language
        detected_lang = self.translator.detect_language(text)
        entity.language = detected_lang
        
        if detected_lang != 'en':
            text = self.translator.translate_to_english(text, detected_lang)
        
        # Extract new information
        entity.amount, entity.currency = self._extract_amount_currency(text)
        entity.recipient = self._extract_entity('recipient', text)
        entity.source_account = self._extract_entity('source_account', text)
        entity.payment_method = self._extract_entity('payment_method', text)
        entity.transaction_id = self._extract_entity('transaction_id', text)
        
        date_str = self._extract_entity('date', text)
        if date_str:
            entity.date = self.date_parser.parse(date_str, detected_lang)
        
        count_str = self._extract_entity('count', text)
        entity.count = int(count_str) if count_str else None
        
        # Merge with last entity
        if self.context.history:
            last_entity = self.context.get_last_entity()
            entity.intent = last_entity.intent
            
            # Fill from follow-up or keep from previous
            if entity.amount is None:
                entity.amount = last_entity.amount
            if entity.currency is None:
                entity.currency = last_entity.currency
            if entity.recipient is None:
                entity.recipient = last_entity.recipient
            if entity.source_account is None:
                entity.source_account = last_entity.source_account
            if entity.payment_method is None:
                entity.payment_method = last_entity.payment_method
            if entity.transaction_id is None:
                entity.transaction_id = last_entity.transaction_id
            if entity.count is None:
                entity.count = last_entity.count
            if entity.date is None:
                entity.date = last_entity.date
        
        # Fuzzy matching
        if self.use_fuzzy and self.fuzzy_matcher:
            entity = self._apply_fuzzy_matching(entity)
        
        entity.missing_slots = self._identify_missing_slots(entity)
        self.context.add_to_history(entity)
        
        return entity
    
    def execute_payment(self, entity: PaymentEntity) -> Dict[str, Any]:
        """
        Execute payment through API if all required slots are filled
        """
        if entity.missing_slots:
            return {
                "success": False,
                "error": "Missing required information",
                "missing_slots": entity.missing_slots
            }
        
        # Process through API
        result = self.api_client.process_payment(entity)
        
        # Update database
        if result.get("success"):
            transaction_id = self.database.save_transaction(entity)
            self.database.update_transaction_status(
                transaction_id,
                PaymentStatus.COMPLETED if result["status"] == "completed" else PaymentStatus.FAILED
            )
            result["internal_transaction_id"] = transaction_id
        
        return result
    
    def get_transaction_history(self, count: int = 10) -> List[Dict]:
        """Retrieve transaction history from database"""
        return self.database.get_recent_transactions(count)
    
    def check_payment_status(self, transaction_id: str) -> Dict[str, Any]:
        """Check payment status via API"""
        # Check local database first
        local_txn = self.database.get_transaction(transaction_id)
        if local_txn:
            return {
                "source": "local",
                **local_txn
            }
        
        # Query API
        api_result = self.api_client.check_status(transaction_id)
        return {
            "source": "api",
            **api_result
        }
    
    # Helper methods
    
    def _parse_money_from_ner(self, money_text: str, entity: PaymentEntity):
        """Parse money from NER result"""
        amount, currency = self._extract_amount_currency(money_text)
        if amount:
            entity.amount = amount
        if currency:
            entity.currency = currency
    
    def _extract_intent_regex(self, text: str) -> IntentType:
        """Regex-based intent extraction"""
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ['cancel', 'abort', 'stop']):
            return IntentType.CANCEL_PAYMENT
        if any(kw in text_lower for kw in ['schedule', 'future']):
            return IntentType.SCHEDULE_PAYMENT
        if any(kw in text_lower for kw in ['recurring', 'monthly', 'weekly']):
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
    
    def _extract_amount_currency(self, text: str) -> Tuple[Optional[float], Optional[str]]:
        """Extract amount and currency"""
        match = re.search(self.patterns['amount'], text, re.IGNORECASE)
        if match:
            amount = float(match.group(1))
            currency_raw = match.group(2).upper()
            
            currency_map = {
                'DOLLARS': 'USD', 'DOLLAR': 'USD',
                'EUROS': 'EUR', 'EURO': 'EUR',
                'POUNDS': 'GBP', 'POUND': 'GBP',
                'RUPEES': 'INR', 'RUPEE': 'INR'
            }
            currency = currency_map.get(currency_raw, currency_raw)
            
            return amount, currency
        return None, None
    
    def _extract_entity(self, pattern_name: str, text: str) -> Optional[str]:
        """Extract a single entity using regex pattern"""
        if pattern_name == 'date':
            date_pattern = r'(tomorrow|today|yesterday|next\s+\w+|\d{4}-\d{2}-\d{2})'
            match = re.search(date_pattern, text, re.IGNORECASE)
        else:
            match = re.search(self.patterns.get(pattern_name, ''), text, re.IGNORECASE)
        
        return match.group(1).strip() if match else None
    
    def _apply_fuzzy_matching(self, entity: PaymentEntity) -> PaymentEntity:
        """Apply fuzzy matching to improve entity recognition"""
        # Fuzzy match recipient
        if entity.recipient:
            match = self.fuzzy_matcher.fuzzy_match_recipient(entity.recipient)
            if match:
                corrected, score = match
                if score > 70:
                    entity.metadata['original_recipient'] = entity.recipient
                    entity.metadata['fuzzy_match_score'] = score
                    entity.recipient = corrected
        
        # Fuzzy match account
        if entity.source_account:
            match = self.fuzzy_matcher.fuzzy_match_account(entity.source_account)
            if match:
                corrected, score = match
                if score > 70:
                    entity.metadata['original_account'] = entity.source_account
                    entity.metadata['fuzzy_account_score'] = score
                    entity.source_account = corrected
        
        # Fuzzy match payment method
        if entity.payment_method:
            match = self.fuzzy_matcher.fuzzy_match_payment_method(entity.payment_method)
            if match:
                corrected, score = match
                if score > 70:
                    entity.metadata['original_method'] = entity.payment_method
                    entity.metadata['fuzzy_method_score'] = score
                    entity.payment_method = corrected
        
        return entity
    
    def _identify_missing_slots(self, entity: PaymentEntity) -> List[str]:
        """Identify missing required slots"""
        missing = []
        
        if entity.intent in [IntentType.MAKE_PAYMENT, IntentType.TRANSFER]:
            if entity.amount is None:
                missing.append('amount')
            if entity.currency is None:
                missing.append('currency')
            if entity.recipient is None:
                missing.append('recipient')
            if entity.source_account is None:
                missing.append('source_account')
        
        elif entity.intent == IntentType.GET_STATUS:
            if entity.transaction_id is None:
                missing.append('transaction_id')
        
        elif entity.intent == IntentType.FETCH_TRANSACTION:
            if entity.count is None:
                missing.append('count')
        
        return missing


class EnhancedNLGGenerator:
    """Enhanced Natural Language Generation with multi-language support"""
    
    def __init__(self):
        self.templates = {
            'en': {
                'confirmation': "I'll process a payment of {amount} {currency} to {recipient} from your {account}{method}{date}. Would you like me to proceed?",
                'fetch': "Fetching the last {count} transaction(s) from your account. Please wait a moment...",
                'status': "Checking the status for payment ID {transaction_id}. One moment please...",
                'missing_amount': "What amount would you like to send?",
                'missing_recipient': "Who should I send this payment to?",
                'missing_account': "Which account should I use for this payment?",
                'success': "Payment of {amount} {currency} to {recipient} was successful! Transaction ID: {txn_id}",
                'failed': "Payment failed: {error}",
            },
            'es': {
                'confirmation': "Procesaré un pago de {amount} {currency} a {recipient} desde su {account}{method}{date}. ¿Desea continuar?",
                'missing_amount': "¿Qué cantidad desea enviar?",
                'missing_recipient': "¿A quién debo enviar este pago?",
            },
            # Add more languages as needed
        }
    
    def generate_confirmation(self, entity: PaymentEntity, language: str = 'en') -> str:
        """Generate confirmation message"""
        if entity.missing_slots:
            return self.generate_slot_request(entity, language)
        
        templates = self.templates.get(language, self.templates['en'])
        
        if entity.intent in [IntentType.MAKE_PAYMENT, IntentType.TRANSFER]:
            date_str = f" scheduled for {entity.date.strftime('%Y-%m-%d')}" if entity.date else ""
            method_str = f" using {entity.payment_method}" if entity.payment_method else ""
            
            return templates['confirmation'].format(
                amount=entity.amount,
                currency=entity.currency,
                recipient=entity.recipient,
                account=entity.source_account,
                method=method_str,
                date=date_str
            )
        
        elif entity.intent == IntentType.FETCH_TRANSACTION:
            return templates['fetch'].format(count=entity.count)
        
        elif entity.intent == IntentType.GET_STATUS:
            return templates['status'].format(transaction_id=entity.transaction_id)
        
        return "I've processed your request."
    
    def generate_slot_request(self, entity: PaymentEntity, language: str = 'en') -> str:
        """Generate request for missing information"""
        if not entity.missing_slots:
            return self.generate_confirmation(entity, language)
        
        templates = self.templates.get(language, self.templates['en'])
        
        slot_questions = {
            'amount': templates.get('missing_amount', "What amount would you like to send?"),
            'currency': "Which currency? (USD, EUR, GBP, INR)",
            'recipient': templates.get('missing_recipient', "Who should I send this payment to?"),
            'source_account': templates.get('missing_account', "Which account should I use?"),
            'transaction_id': "Please provide the transaction or payment ID.",
            'count': "How many transactions would you like to see?"
        }
        
        first_missing = entity.missing_slots[0]
        question = slot_questions.get(first_missing, f"Please provide the {first_missing}.")
        
        if len(entity.missing_slots) > 1:
            return f"{question} I'll also need the {', '.join(entity.missing_slots[1:])}."
        
        return question
    
    def generate_execution_result(self, result: Dict[str, Any], entity: PaymentEntity, language: str = 'en') -> str:
        """Generate message for execution result"""
        templates = self.templates.get(language, self.templates['en'])
        
        if result.get('success'):
            return templates['success'].format(
                amount=entity.amount,
                currency=entity.currency,
                recipient=entity.recipient,
                txn_id=result.get('internal_transaction_id', result.get('transaction_id', 'N/A'))
            )
        else:
            return templates['failed'].format(
                error=result.get('message', 'Unknown error')
            )


# Export main classes
__all__ = ['EnhancedPaymentExtractor', 'EnhancedNLGGenerator']
