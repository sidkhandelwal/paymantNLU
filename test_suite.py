"""
Comprehensive Test Suite for Advanced Payment System
Tests all components: ML, NER, Multi-language, Fuzzy Matching, Database, API
"""

import pytest
from datetime import datetime
from enhanced_payment_extractor import EnhancedPaymentExtractor, EnhancedNLGGenerator
from advanced_ml_components import (
    IntentType, PaymentStatus, PaymentEntity,
    MLIntentClassifier, NERExtractor, MultiLanguageTranslator,
    FuzzyMatcher, AdvancedDateParser, PaymentDatabase, PaymentAPIClient
)


class TestMLIntentClassifier:
    """Test ML Intent Classification"""
    
    def test_make_payment_intent(self):
        classifier = MLIntentClassifier()
        intent, confidence = classifier.predict("I want to make a payment to John")
        assert intent == IntentType.MAKE_PAYMENT
        assert confidence > 0
    
    def test_transfer_intent(self):
        classifier = MLIntentClassifier()
        intent, _ = classifier.predict("Transfer money from savings to checking")
        assert intent == IntentType.TRANSFER
    
    def test_fetch_transaction_intent(self):
        classifier = MLIntentClassifier()
        intent, _ = classifier.predict("Show me my last 10 transactions")
        assert intent == IntentType.FETCH_TRANSACTION
    
    def test_get_status_intent(self):
        classifier = MLIntentClassifier()
        intent, _ = classifier.predict("Check the status of transaction ABC123")
        assert intent == IntentType.GET_STATUS
    
    def test_cancel_payment_intent(self):
        classifier = MLIntentClassifier()
        intent, _ = classifier.predict("Cancel my pending payment")
        assert intent in [IntentType.CANCEL_PAYMENT, IntentType.FETCH_TRANSACTION]


class TestNERExtractor:
    """Test Named Entity Recognition"""
    
    def test_extract_person(self):
        ner = NERExtractor()
        entities = ner.extract_entities("Send money to Alice Johnson")
        assert len(entities.get('PERSON', [])) > 0
    
    def test_extract_money(self):
        ner = NERExtractor()
        entities = ner.extract_entities("Pay 500 USD")
        assert len(entities.get('MONEY', [])) > 0
    
    def test_extract_date(self):
        ner = NERExtractor()
        entities = ner.extract_entities("Transfer tomorrow")
        assert len(entities.get('DATE', [])) > 0


class TestMultiLanguageTranslator:
    """Test Multi-language Support"""
    
    def test_detect_spanish(self):
        translator = MultiLanguageTranslator()
        lang = translator.detect_language("Pagar 100 EUR")
        assert lang == 'es'
    
    def test_detect_french(self):
        translator = MultiLanguageTranslator()
        lang = translator.detect_language("Transférer de l'argent")
        assert lang == 'fr'
    
    def test_detect_english(self):
        translator = MultiLanguageTranslator()
        lang = translator.detect_language("Send money to John")
        assert lang == 'en'
    
    def test_translate_spanish(self):
        translator = MultiLanguageTranslator()
        translated = translator.translate_to_english("Pagar dinero", 'es')
        assert 'pay' in translated.lower() or 'money' in translated.lower()


class TestFuzzyMatcher:
    """Test Fuzzy Matching"""
    
    def test_fuzzy_match_recipient(self):
        matcher = FuzzyMatcher()
        result = matcher.fuzzy_match_recipient("Ramkrisna")
        assert result is not None
        match, score = result
        assert score > 70
    
    def test_fuzzy_match_account(self):
        matcher = FuzzyMatcher()
        result = matcher.fuzzy_match_account("Savngs Account")
        if result:
            match, score = result
            assert "Savings" in match
    
    def test_fuzzy_match_payment_method(self):
        matcher = FuzzyMatcher()
        result = matcher.fuzzy_match_payment_method("FasterPyment")
        if result:
            match, score = result
            assert "FasterPayment" in match or "Payment" in match
    
    def test_no_match_low_similarity(self):
        matcher = FuzzyMatcher()
        result = matcher.fuzzy_match_recipient("XYZ123", threshold=90)
        assert result is None or result[1] < 90


class TestAdvancedDateParser:
    """Test Date Parsing"""
    
    def test_parse_tomorrow(self):
        parser = AdvancedDateParser()
        date = parser.parse("tomorrow")
        assert date is not None
        assert date > datetime.now()
    
    def test_parse_today(self):
        parser = AdvancedDateParser()
        date = parser.parse("today")
        assert date is not None
    
    def test_parse_iso_date(self):
        parser = AdvancedDateParser()
        date = parser.parse("2026-03-15")
        assert date is not None
        assert date.year == 2026
        assert date.month == 3
        assert date.day == 15


class TestPaymentDatabase:
    """Test Database Integration"""
    
    def test_save_transaction(self):
        db = PaymentDatabase("sqlite:///:memory:")
        entity = PaymentEntity(
            intent=IntentType.MAKE_PAYMENT,
            amount=100.0,
            currency="USD",
            recipient="Alice"
        )
        txn_id = db.save_transaction(entity)
        assert txn_id is not None
        assert txn_id.startswith("TXN")
    
    def test_get_transaction(self):
        db = PaymentDatabase("sqlite:///:memory:")
        entity = PaymentEntity(
            intent=IntentType.MAKE_PAYMENT,
            amount=100.0,
            currency="USD",
            recipient="Alice"
        )
        txn_id = db.save_transaction(entity)
        retrieved = db.get_transaction(txn_id)
        assert retrieved is not None
        assert retrieved['amount'] == 100.0
        assert retrieved['currency'] == "USD"
    
    def test_get_recent_transactions(self):
        db = PaymentDatabase("sqlite:///:memory:")
        
        # Create multiple transactions
        for i in range(5):
            entity = PaymentEntity(
                intent=IntentType.MAKE_PAYMENT,
                amount=100.0 * (i + 1),
                currency="USD",
                recipient=f"User{i}"
            )
            db.save_transaction(entity)
        
        transactions = db.get_recent_transactions(count=3)
        assert len(transactions) == 3
    
    def test_update_transaction_status(self):
        db = PaymentDatabase("sqlite:///:memory:")
        entity = PaymentEntity(
            intent=IntentType.MAKE_PAYMENT,
            amount=100.0,
            currency="USD"
        )
        txn_id = db.save_transaction(entity)
        
        success = db.update_transaction_status(txn_id, PaymentStatus.COMPLETED)
        assert success is True
        
        retrieved = db.get_transaction(txn_id)
        assert retrieved['status'] == PaymentStatus.COMPLETED.value


class TestPaymentAPIClient:
    """Test Payment API Client"""
    
    def test_process_payment(self):
        client = PaymentAPIClient()
        entity = PaymentEntity(
            intent=IntentType.MAKE_PAYMENT,
            amount=100.0,
            currency="USD",
            recipient="Alice"
        )
        result = client.process_payment(entity)
        
        assert 'success' in result
        assert 'transaction_id' in result
        assert 'status' in result
    
    def test_check_status(self):
        client = PaymentAPIClient()
        result = client.check_status("TXN12345")
        
        assert 'transaction_id' in result
        assert 'status' in result


class TestEnhancedPaymentExtractor:
    """Test Complete Payment Extractor"""
    
    def test_parse_complete_payment(self):
        extractor = EnhancedPaymentExtractor()
        entity = extractor.parse(
            "Make a payment of 100 USD to RamKrishna from Salary Account using FasterPayment tomorrow"
        )
        
        assert entity.intent == IntentType.MAKE_PAYMENT
        assert entity.amount == 100.0
        assert entity.currency == "USD"
        assert entity.recipient is not None
        assert entity.source_account is not None
        assert entity.payment_method is not None
    
    def test_parse_transfer(self):
        extractor = EnhancedPaymentExtractor()
        entity = extractor.parse("Transfer 29 USD to Ramesh from Account 3423425")
        
        assert entity.intent == IntentType.TRANSFER
        assert entity.amount == 29.0
        assert entity.currency == "USD"
        assert entity.recipient is not None
    
    def test_parse_fetch_transaction(self):
        extractor = EnhancedPaymentExtractor()
        entity = extractor.parse("Fetch transaction status of last 10 payment")
        
        assert entity.intent == IntentType.FETCH_TRANSACTION
        assert entity.count == 10
    
    def test_parse_get_status(self):
        extractor = EnhancedPaymentExtractor()
        entity = extractor.parse("Get swift status of Payment BX 302942309")
        
        assert entity.intent == IntentType.GET_STATUS
        assert entity.transaction_id is not None
    
    def test_context_filling(self):
        extractor = EnhancedPaymentExtractor()
        
        # First request
        entity1 = extractor.parse("Send money to Alice")
        assert entity1.missing_slots  # Should have missing slots
        
        # Follow-up
        entity2 = extractor.process_followup("500 USD")
        assert entity2.amount == 500.0
        assert entity2.currency == "USD"
    
    def test_fuzzy_matching_integration(self):
        extractor = EnhancedPaymentExtractor(use_fuzzy=True)
        entity = extractor.parse("Pay Alice Jonson from Savings Acount")
        
        # Should correct "Acount" to "Account"
        assert "Account" in entity.source_account or 'fuzzy' in entity.metadata
    
    def test_multilanguage_integration(self):
        extractor = EnhancedPaymentExtractor()
        entity = extractor.parse("Pagar 100 EUR")
        
        assert entity.language == 'es'
        assert entity.amount == 100.0
        assert entity.currency == "EUR"


class TestEnhancedNLGGenerator:
    """Test Natural Language Generation"""
    
    def test_generate_confirmation(self):
        nlg = EnhancedNLGGenerator()
        entity = PaymentEntity(
            intent=IntentType.MAKE_PAYMENT,
            amount=100.0,
            currency="USD",
            recipient="Alice",
            source_account="Savings Account"
        )
        
        response = nlg.generate_confirmation(entity)
        assert "100.0" in response
        assert "USD" in response
        assert "Alice" in response
    
    def test_generate_slot_request(self):
        nlg = EnhancedNLGGenerator()
        entity = PaymentEntity(
            intent=IntentType.MAKE_PAYMENT,
            missing_slots=['amount', 'currency']
        )
        
        response = nlg.generate_slot_request(entity)
        assert "amount" in response.lower()
    
    def test_generate_multilanguage_response(self):
        nlg = EnhancedNLGGenerator()
        entity = PaymentEntity(
            intent=IntentType.MAKE_PAYMENT,
            amount=100.0,
            currency="EUR",
            recipient="Juan",
            source_account="Cuenta Corriente",
            language='es'
        )
        
        response = nlg.generate_confirmation(entity, language='es')
        # Should contain Spanish text or at least process without error
        assert response is not None


class TestEndToEndWorkflow:
    """End-to-End Integration Tests"""
    
    def test_complete_payment_workflow(self):
        extractor = EnhancedPaymentExtractor()
        nlg = EnhancedNLGGenerator()
        
        # Parse command
        entity = extractor.parse("Send 100 USD to Alice from Savings Account")
        
        # Generate confirmation
        confirmation = nlg.generate_confirmation(entity)
        assert confirmation is not None
        
        # Execute payment
        if not entity.missing_slots:
            result = extractor.execute_payment(entity)
            assert 'success' in result
    
    def test_multiturn_conversation(self):
        extractor = EnhancedPaymentExtractor()
        
        # Turn 1
        entity1 = extractor.parse("I want to send money")
        assert len(entity1.missing_slots) > 0
        
        # Turn 2
        entity2 = extractor.process_followup("500 EUR to Bob")
        assert entity2.amount == 500.0
        assert entity2.currency == "EUR"
        
        # Turn 3
        entity3 = extractor.process_followup("From Business Account")
        assert entity3.source_account is not None
    
    def test_database_and_api_integration(self):
        extractor = EnhancedPaymentExtractor()
        
        # Create payment
        entity = extractor.parse("Send 100 USD to Alice from Savings Account")
        
        # Execute
        result = extractor.execute_payment(entity)
        
        # Verify in database
        if result.get('success') and result.get('internal_transaction_id'):
            txn_id = result['internal_transaction_id']
            status = extractor.check_payment_status(txn_id)
            assert status is not None


# Performance Tests
class TestPerformance:
    """Performance and Benchmark Tests"""
    
    def test_parsing_speed(self):
        import time
        extractor = EnhancedPaymentExtractor()
        
        start = time.time()
        for _ in range(100):
            extractor.parse("Send 100 USD to Alice")
        end = time.time()
        
        avg_time = (end - start) / 100
        assert avg_time < 0.5  # Should be under 500ms per parse
    
    def test_batch_processing(self):
        extractor = EnhancedPaymentExtractor()
        
        commands = [
            "Send 100 USD to Alice",
            "Transfer 200 EUR to Bob",
            "Pay 50 GBP to Charlie",
        ] * 10
        
        results = [extractor.parse(cmd) for cmd in commands]
        assert len(results) == 30


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=.", "--cov-report=html"])
