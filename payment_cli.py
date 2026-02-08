#!/usr/bin/env python3
"""
Interactive Payment Assistant CLI
Production-ready command-line interface for payment processing
"""

import sys
import json
from datetime import datetime
from typing import Optional

try:
    from enhanced_payment_extractor import EnhancedPaymentExtractor, EnhancedNLGGenerator
    from advanced_ml_components import IntentType, PaymentStatus
except ImportError:
    print("Error: Required modules not found. Please ensure all files are in the same directory.")
    sys.exit(1)


class PaymentCLI:
    """Interactive CLI for payment processing"""
    
    def __init__(self):
        self.extractor = EnhancedPaymentExtractor(
            use_ml=True,
            use_ner=True,
            use_fuzzy=True,
            db_url="sqlite:///payments.db",
            api_key="demo_key"
        )
        self.nlg = EnhancedNLGGenerator()
        self.session_active = True
        
    def print_banner(self):
        """Print welcome banner"""
        print("\n" + "=" * 80)
        print("  ADVANCED PAYMENT ASSISTANT".center(80))
        print("  Powered by ML, NER, Multi-language Support".center(80))
        print("=" * 80)
        print("\nFeatures:")
        print("  ✓ Machine Learning Intent Classification")
        print("  ✓ Named Entity Recognition")
        print("  ✓ Multi-language Support (EN, ES, FR, DE, HI)")
        print("  ✓ Fuzzy Matching & Typo Correction")
        print("  ✓ Context-Aware Conversations")
        print("  ✓ Database Transaction Logging")
        print("  ✓ Real-time Payment Processing")
        print("\nType 'help' for commands, 'quit' to exit\n")
    
    def print_help(self):
        """Print help message"""
        print("\n" + "-" * 80)
        print("AVAILABLE COMMANDS:")
        print("-" * 80)
        print("  help              - Show this help message")
        print("  quit / exit       - Exit the application")
        print("  reset             - Clear conversation context")
        print("  history [n]       - Show last n transactions (default: 10)")
        print("  status <id>       - Check payment status")
        print("  stats             - Show session statistics")
        print("  lang <code>       - Set preferred language (en, es, fr, de, hi)")
        print("\nPAYMENT COMMANDS:")
        print("-" * 80)
        print("  Make payment:")
        print("    Send 100 USD to Alice from Savings Account")
        print("    Pay 500 EUR to Bob")
        print("\n  Transfer:")
        print("    Transfer 200 USD from Checking to Savings")
        print("\n  Check status:")
        print("    Get swift status of Payment BX 302942309")
        print("    Check transaction ABC123456789")
        print("\n  Fetch history:")
        print("    Show last 10 transactions")
        print("    Fetch transaction history")
        print("\n  Multi-language:")
        print("    Pagar 100 EUR a Juan (Spanish)")
        print("    Transférer 200 EUR à Marie (French)")
        print("-" * 80 + "\n")
    
    def print_stats(self):
        """Print session statistics"""
        history = self.extractor.context.history
        
        print("\n" + "-" * 80)
        print("SESSION STATISTICS:")
        print("-" * 80)
        print(f"  Total Commands: {len(history)}")
        
        if history:
            intent_counts = {}
            for entity in history:
                intent_counts[entity.intent.value] = intent_counts.get(entity.intent.value, 0) + 1
            
            print(f"\n  Intent Breakdown:")
            for intent, count in intent_counts.items():
                print(f"    - {intent}: {count}")
            
            # Language stats
            lang_counts = {}
            for entity in history:
                lang_counts[entity.language] = lang_counts.get(entity.language, 0) + 1
            
            print(f"\n  Language Usage:")
            for lang, count in lang_counts.items():
                print(f"    - {lang}: {count}")
            
            # Avg confidence
            avg_conf = sum(e.confidence for e in history) / len(history)
            print(f"\n  Average Confidence: {avg_conf:.2%}")
        
        print("-" * 80 + "\n")
    
    def handle_history(self, count: int = 10):
        """Display transaction history"""
        transactions = self.extractor.get_transaction_history(count)
        
        if not transactions:
            print("\n  No transactions found.\n")
            return
        
        print("\n" + "-" * 80)
        print(f"TRANSACTION HISTORY (Last {len(transactions)} transactions):")
        print("-" * 80)
        
        for i, txn in enumerate(transactions, 1):
            print(f"\n{i}. Transaction ID: {txn['id']}")
            print(f"   Intent: {txn['intent']}")
            if txn['amount']:
                print(f"   Amount: {txn['amount']} {txn['currency']}")
            if txn['recipient']:
                print(f"   Recipient: {txn['recipient']}")
            if txn['source_account']:
                print(f"   Source: {txn['source_account']}")
            print(f"   Status: {txn['status']}")
            print(f"   Created: {txn['created_at']}")
        
        print("-" * 80 + "\n")
    
    def handle_status(self, transaction_id: str):
        """Check payment status"""
        print(f"\nChecking status for transaction {transaction_id}...")
        
        status = self.extractor.check_payment_status(transaction_id)
        
        print("\n" + "-" * 80)
        print("PAYMENT STATUS:")
        print("-" * 80)
        print(f"  Transaction ID: {status.get('id', status.get('transaction_id'))}")
        print(f"  Status: {status.get('status')}")
        print(f"  Source: {status.get('source')}")
        
        if 'amount' in status:
            print(f"  Amount: {status['amount']} {status.get('currency')}")
        if 'recipient' in status:
            print(f"  Recipient: {status['recipient']}")
        if 'last_updated' in status:
            print(f"  Last Updated: {status['last_updated']}")
        
        print("-" * 80 + "\n")
    
    def process_payment_command(self, text: str):
        """Process payment-related command"""
        # Check if this is a follow-up
        if self.extractor.context.history:
            last_entity = self.extractor.context.get_last_entity()
            if last_entity and last_entity.missing_slots:
                entity = self.extractor.process_followup(text)
            else:
                entity = self.extractor.parse(text)
        else:
            entity = self.extractor.parse(text)
        
        # Show extraction details
        print(f"\n  [Intent: {entity.intent.value}]", end="")
        if entity.confidence > 0:
            print(f" [Confidence: {entity.confidence:.2%}]", end="")
        if entity.language != 'en':
            print(f" [Language: {entity.language}]", end="")
        print()
        
        # Show metadata if fuzzy matching was applied
        if entity.metadata:
            if 'original_recipient' in entity.metadata:
                print(f"  [Fuzzy Match] Corrected recipient: {entity.metadata['original_recipient']} → {entity.recipient}")
            if 'original_account' in entity.metadata:
                print(f"  [Fuzzy Match] Corrected account: {entity.metadata['original_account']} → {entity.source_account}")
        
        # Generate response
        response = self.nlg.generate_confirmation(entity, entity.language)
        print(f"\n  Assistant: {response}\n")
        
        # If all slots filled, offer to execute
        if not entity.missing_slots and entity.intent in [IntentType.MAKE_PAYMENT, IntentType.TRANSFER]:
            execute = input("  Execute this payment? (yes/no): ").strip().lower()
            
            if execute in ['yes', 'y']:
                print("\n  Processing payment...")
                result = self.extractor.execute_payment(entity)
                
                print("\n" + "-" * 80)
                print("  PAYMENT RESULT:")
                print("-" * 80)
                print(f"  Success: {result.get('success')}")
                print(f"  Status: {result.get('status')}")
                print(f"  Message: {result.get('message')}")
                if result.get('transaction_id'):
                    print(f"  Transaction ID: {result.get('transaction_id')}")
                if result.get('internal_transaction_id'):
                    print(f"  Internal ID: {result.get('internal_transaction_id')}")
                print("-" * 80 + "\n")
    
    def run(self):
        """Main CLI loop"""
        self.print_banner()
        
        while self.session_active:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit']:
                    print("\nThank you for using Payment Assistant. Goodbye!\n")
                    break
                
                elif user_input.lower() == 'help':
                    self.print_help()
                
                elif user_input.lower() == 'reset':
                    self.extractor = EnhancedPaymentExtractor()
                    print("\n  [Context reset]\n")
                
                elif user_input.lower() == 'stats':
                    self.print_stats()
                
                elif user_input.lower().startswith('history'):
                    parts = user_input.split()
                    count = int(parts[1]) if len(parts) > 1 else 10
                    self.handle_history(count)
                
                elif user_input.lower().startswith('status'):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1:
                        self.handle_status(parts[1])
                    else:
                        print("\n  Please provide a transaction ID\n")
                
                else:
                    # Process as payment command
                    self.process_payment_command(user_input)
            
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit.\n")
            
            except Exception as e:
                print(f"\n  Error: {str(e)}\n")
                if '--debug' in sys.argv:
                    import traceback
                    traceback.print_exc()


def main():
    """Entry point"""
    cli = PaymentCLI()
    cli.run()


if __name__ == "__main__":
    main()
