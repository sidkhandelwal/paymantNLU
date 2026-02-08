"""
Comprehensive Demo of Enhanced Payment System
Demonstrates all advanced features
"""

from enhanced_payment_extractor import EnhancedPaymentExtractor, EnhancedNLGGenerator
from advanced_ml_components import IntentType, PaymentStatus


def demo_ml_intent_classification():
    """Demo 1: ML-based Intent Classification"""
    print("=" * 80)
    print("DEMO 1: ML-BASED INTENT CLASSIFICATION")
    print("=" * 80)
    print()
    
    extractor = EnhancedPaymentExtractor(use_ml=True)
    nlg = EnhancedNLGGenerator()
    
    test_cases = [
        "I want to make a payment to John",
        "Transfer money from savings to checking",
        "Show me my last 5 transactions",
        "What's the status of transaction ABC123456789",
        "Cancel my pending payment",
        "Schedule a payment for next week",
    ]
    
    for text in test_cases:
        entity = extractor.parse(text)
        print(f"Input: {text}")
        print(f"Intent: {entity.intent.value} (confidence: {entity.confidence:.2f})")
        print(f"Response: {nlg.generate_confirmation(entity)}")
        print()


def demo_ner_extraction():
    """Demo 2: Named Entity Recognition"""
    print("=" * 80)
    print("DEMO 2: NAMED ENTITY RECOGNITION (NER)")
    print("=" * 80)
    print()
    
    extractor = EnhancedPaymentExtractor(use_ner=True)
    
    test_cases = [
        "Send 500 USD to Alice Johnson tomorrow",
        "Transfer 1000 EUR to Bob Williams from Business Account",
        "Pay 250.50 GBP to Charlie Brown",
    ]
    
    for text in test_cases:
        entity = extractor.parse(text)
        print(f"Input: {text}")
        print(f"Extracted Entities:")
        print(f"  - Amount: {entity.amount} {entity.currency}")
        print(f"  - Recipient: {entity.recipient}")
        print(f"  - Account: {entity.source_account}")
        if entity.date:
            print(f"  - Date: {entity.date.strftime('%Y-%m-%d')}")
        print()


def demo_multilanguage():
    """Demo 3: Multi-language Support"""
    print("=" * 80)
    print("DEMO 3: MULTI-LANGUAGE SUPPORT")
    print("=" * 80)
    print()
    
    extractor = EnhancedPaymentExtractor()
    nlg = EnhancedNLGGenerator()
    
    test_cases = [
        ("Pagar 100 USD a Juan", "es"),  # Spanish
        ("Transférer 200 EUR à Marie", "fr"),  # French
        ("Bezahlen 150 EUR an Hans", "de"),  # German
        ("Send 300 USD to Alice", "en"),  # English
    ]
    
    for text, lang in test_cases:
        entity = extractor.parse(text)
        print(f"Input ({lang}): {text}")
        print(f"Detected Language: {entity.language}")
        print(f"Intent: {entity.intent.value}")
        if entity.amount:
            print(f"Amount: {entity.amount} {entity.currency}")
        print(f"Response: {nlg.generate_confirmation(entity, entity.language)}")
        print()


def demo_fuzzy_matching():
    """Demo 4: Fuzzy Matching for Names"""
    print("=" * 80)
    print("DEMO 4: FUZZY MATCHING (TYPO CORRECTION)")
    print("=" * 80)
    print()
    
    extractor = EnhancedPaymentExtractor(use_fuzzy=True)
    
    test_cases = [
        "Send 100 USD to Ramkrisna from Salery Account",  # Typos
        "Transfer to Ramsh using FasterPyment",  # Typos
        "Pay Alice Jonson from Savings Acount",  # Typos
    ]
    
    for text in test_cases:
        entity = extractor.parse(text)
        print(f"Input: {text}")
        print(f"Corrected Entities:")
        if 'original_recipient' in entity.metadata:
            print(f"  - Recipient: {entity.metadata['original_recipient']} → {entity.recipient} (score: {entity.metadata.get('fuzzy_match_score', 0)})")
        if 'original_account' in entity.metadata:
            print(f"  - Account: {entity.metadata['original_account']} → {entity.source_account} (score: {entity.metadata.get('fuzzy_account_score', 0)})")
        if 'original_method' in entity.metadata:
            print(f"  - Method: {entity.metadata['original_method']} → {entity.payment_method} (score: {entity.metadata.get('fuzzy_method_score', 0)})")
        print()


def demo_date_parsing():
    """Demo 5: Advanced Date Parsing"""
    print("=" * 80)
    print("DEMO 5: ADVANCED DATE PARSING")
    print("=" * 80)
    print()
    
    extractor = EnhancedPaymentExtractor()
    
    test_cases = [
        "Send 100 USD to John tomorrow",
        "Schedule payment for 2026-03-15",
        "Transfer today",
        "Pay yesterday",
    ]
    
    for text in test_cases:
        entity = extractor.parse(text)
        print(f"Input: {text}")
        if entity.date:
            print(f"Parsed Date: {entity.date.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"Parsed Date: None")
        print()


def demo_database_integration():
    """Demo 6: Database Integration"""
    print("=" * 80)
    print("DEMO 6: DATABASE INTEGRATION (Transaction Storage)")
    print("=" * 80)
    print()
    
    extractor = EnhancedPaymentExtractor()
    nlg = EnhancedNLGGenerator()
    
    # Process some payments
    commands = [
        "Send 100 USD to Alice from Savings Account",
        "Transfer 200 EUR to Bob from Checking Account",
        "Pay 50 GBP to Charlie from Business Account",
    ]
    
    print("Processing payments...")
    for cmd in commands:
        entity = extractor.parse(cmd)
        print(f"  - {cmd}")
    
    print("\nRetrieving transaction history...")
    history = extractor.get_transaction_history(count=5)
    
    print(f"\nFound {len(history)} transactions:")
    for txn in history:
        print(f"  ID: {txn['id']}")
        print(f"  Amount: {txn['amount']} {txn['currency']}")
        print(f"  Recipient: {txn['recipient']}")
        print(f"  Status: {txn['status']}")
        print(f"  Created: {txn['created_at']}")
        print()


def demo_api_integration():
    """Demo 7: Payment API Integration"""
    print("=" * 80)
    print("DEMO 7: PAYMENT API INTEGRATION (Execute Real Payments)")
    print("=" * 80)
    print()
    
    extractor = EnhancedPaymentExtractor()
    nlg = EnhancedNLGGenerator()
    
    # Complete payment request
    cmd = "Send 100 USD to Alice from Savings Account"
    entity = extractor.parse(cmd)
    
    print(f"Command: {cmd}")
    print(f"Confirmation: {nlg.generate_confirmation(entity)}")
    print()
    
    # Execute payment
    print("Executing payment through API...")
    result = extractor.execute_payment(entity)
    
    print(f"API Response:")
    print(f"  Success: {result.get('success')}")
    print(f"  Status: {result.get('status')}")
    print(f"  Message: {result.get('message')}")
    print(f"  Transaction ID: {result.get('transaction_id')}")
    if result.get('internal_transaction_id'):
        print(f"  Internal ID: {result.get('internal_transaction_id')}")
    print()


def demo_context_management():
    """Demo 8: Context Management & Slot Filling"""
    print("=" * 80)
    print("DEMO 8: CONTEXT MANAGEMENT & MULTI-TURN SLOT FILLING")
    print("=" * 80)
    print()
    
    extractor = EnhancedPaymentExtractor()
    nlg = EnhancedNLGGenerator()
    
    conversation = [
        "I want to send money",
        "500 EUR",
        "To Alice Johnson",
        "From my Business Account",
    ]
    
    print("Multi-turn conversation:")
    for i, turn in enumerate(conversation, 1):
        print(f"\nTurn {i}")
        print(f"User: {turn}")
        
        if i == 1:
            entity = extractor.parse(turn)
        else:
            entity = extractor.process_followup(turn)
        
        print(f"Bot: {nlg.generate_confirmation(entity)}")
        
        if not entity.missing_slots:
            print("\n[All slots filled! Ready to execute payment]")
            result = extractor.execute_payment(entity)
            print(nlg.generate_execution_result(result, entity))
            break


def demo_status_check():
    """Demo 9: Payment Status Check"""
    print("=" * 80)
    print("DEMO 9: PAYMENT STATUS CHECK")
    print("=" * 80)
    print()
    
    extractor = EnhancedPaymentExtractor()
    
    # First, create a transaction
    cmd = "Send 100 USD to Bob from Savings Account"
    entity = extractor.parse(cmd)
    result = extractor.execute_payment(entity)
    
    if result.get('success'):
        txn_id = result.get('internal_transaction_id')
        print(f"Created transaction: {txn_id}")
        print()
        
        # Check status
        print("Checking status...")
        status = extractor.check_payment_status(txn_id)
        
        print(f"Status Response:")
        print(f"  Source: {status.get('source')}")
        print(f"  Transaction ID: {status.get('id', status.get('transaction_id'))}")
        print(f"  Status: {status.get('status')}")
        print(f"  Amount: {status.get('amount')} {status.get('currency')}")
        print()


def demo_complete_workflow():
    """Demo 10: Complete Real-World Workflow"""
    print("=" * 80)
    print("DEMO 10: COMPLETE REAL-WORLD WORKFLOW")
    print("=" * 80)
    print()
    
    extractor = EnhancedPaymentExtractor()
    nlg = EnhancedNLGGenerator()
    
    print("Scenario: User wants to make a payment with typos, multi-turn")
    print()
    
    # Turn 1: Initial request with typo
    print("Turn 1:")
    print("User: Pagar 250 EUR to Ramkrisna")  # Spanish + typo
    entity1 = extractor.parse("Pagar 250 EUR to Ramkrisna")
    print(f"Bot: {nlg.generate_confirmation(entity1)}")
    print()
    
    # Turn 2: Provide account
    print("Turn 2:")
    print("User: From Salery Account")  # Typo
    entity2 = extractor.process_followup("From Salery Account")
    print(f"Bot: {nlg.generate_confirmation(entity2)}")
    print()
    
    # Execute
    if not entity2.missing_slots:
        print("Executing payment...")
        result = extractor.execute_payment(entity2)
        print(nlg.generate_execution_result(result, entity2))
        print()
        
        # Show what was corrected
        print("Corrections made:")
        if 'original_recipient' in entity2.metadata:
            print(f"  - Recipient: '{entity2.metadata['original_recipient']}' → '{entity2.recipient}'")
        if 'original_account' in entity2.metadata:
            print(f"  - Account: '{entity2.metadata['original_account']}' → '{entity2.source_account}'")


def run_all_demos():
    """Run all demonstrations"""
    demos = [
        demo_ml_intent_classification,
        demo_ner_extraction,
        demo_multilanguage,
        demo_fuzzy_matching,
        demo_date_parsing,
        demo_database_integration,
        demo_api_integration,
        demo_context_management,
        demo_status_check,
        demo_complete_workflow,
    ]
    
    for i, demo in enumerate(demos, 1):
        demo()
        if i < len(demos):
            print("\n" + "▼" * 80 + "\n")
            input("Press Enter to continue to next demo...")
            print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        demo_map = {
            "ml": demo_ml_intent_classification,
            "ner": demo_ner_extraction,
            "multilang": demo_multilanguage,
            "fuzzy": demo_fuzzy_matching,
            "date": demo_date_parsing,
            "db": demo_database_integration,
            "api": demo_api_integration,
            "context": demo_context_management,
            "status": demo_status_check,
            "workflow": demo_complete_workflow,
            "all": run_all_demos,
        }
        
        demo_name = sys.argv[1].lower()
        if demo_name in demo_map:
            demo_map[demo_name]()
        else:
            print(f"Unknown demo: {demo_name}")
            print(f"Available demos: {', '.join(demo_map.keys())}")
    else:
        run_all_demos()
