# Quick Start Guide - Advanced Payment System

## 🚀 Installation

```bash
# Install dependencies (in production with network access)
pip install -r requirements.txt --break-system-packages

# Download spaCy model
python -m spacy download en_core_web_sm
```

## 📂 File Structure

```
.
├── advanced_ml_components.py       # Core ML components (ML, NER, Fuzzy, etc.)
├── enhanced_payment_extractor.py   # Main payment extractor
├── advanced_demo.py                # Comprehensive demos
├── payment_cli.py                  # Interactive CLI application
├── test_suite.py                   # Complete test suite
├── requirements.txt                # Python dependencies
├── ADVANCED_README.md              # Full documentation
└── QUICK_START.md                  # This file
```

## 🎯 Quick Examples

### 1. Basic Usage

```python
from enhanced_payment_extractor import EnhancedPaymentExtractor, EnhancedNLGGenerator

# Initialize
extractor = EnhancedPaymentExtractor()
nlg = EnhancedNLGGenerator()

# Process command
entity = extractor.parse("Send 100 USD to Alice from Savings Account")
response = nlg.generate_confirmation(entity)
print(response)

# Execute payment
if not entity.missing_slots:
    result = extractor.execute_payment(entity)
    print(result)
```

### 2. Interactive CLI

```bash
# Run interactive payment assistant
python payment_cli.py

# With debug mode
python payment_cli.py --debug
```

### 3. Run Demos

```bash
# All demos
python advanced_demo.py all

# Specific features
python advanced_demo.py ml          # Machine learning
python advanced_demo.py ner         # Named entity recognition
python advanced_demo.py multilang   # Multi-language
python advanced_demo.py fuzzy       # Fuzzy matching
python advanced_demo.py db          # Database
python advanced_demo.py api         # API integration
python advanced_demo.py workflow    # Complete workflow
```

### 4. Run Tests

```bash
# Run all tests
python -m pytest test_suite.py -v

# With coverage
python -m pytest test_suite.py --cov=. --cov-report=html

# Specific test class
python -m pytest test_suite.py::TestMLIntentClassifier -v
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```bash
# Database
DATABASE_URL=sqlite:///payments.db
# DATABASE_URL=postgresql://user:pass@localhost/payments  # Production

# API
PAYMENT_API_KEY=your_api_key_here
PAYMENT_API_ENV=sandbox  # or production

# ML Models
ML_MODEL_PATH=/models/intent_classifier
NER_MODEL_PATH=/models/ner_model

# Features
ENABLE_ML=true
ENABLE_NER=true
ENABLE_FUZZY=true
```

### Python Code Configuration

```python
extractor = EnhancedPaymentExtractor(
    use_ml=True,              # Enable ML intent classification
    use_ner=True,             # Enable NER
    use_fuzzy=True,           # Enable fuzzy matching
    db_url="sqlite:///payments.db",
    api_key="demo_key"
)
```

## 📊 Feature Matrix

| Feature                      | Status | Description                           |
|------------------------------|--------|---------------------------------------|
| ML Intent Classification     | ✅     | TF-IDF + Naive Bayes / BERT          |
| Named Entity Recognition     | ✅     | spaCy / Transformers                 |
| Multi-language Support       | ✅     | EN, ES, FR, DE, HI                   |
| Fuzzy Matching              | ✅     | Typo correction with rapidfuzz       |
| Advanced Date Parsing       | ✅     | Natural language dates               |
| Database Integration        | ✅     | SQLAlchemy (SQLite/PostgreSQL/MySQL) |
| Payment API Integration     | ✅     | REST API client with retry logic     |
| Context Management          | ✅     | Multi-turn conversations             |
| Slot Filling               | ✅     | Automatic missing info detection     |
| Multi-language NLG         | ✅     | Localized responses                  |

## 🎨 Example Commands

### English
```
Send 100 USD to Alice from Savings Account
Transfer 500 EUR to Bob using SWIFT
Check status of payment BX 302942309
Show last 10 transactions
Cancel payment TXN00000123
```

### Spanish
```
Pagar 100 EUR a Juan
Transferir 200 EUR a María
```

### French
```
Transférer 150 EUR à Pierre
Payer 300 EUR à Sophie
```

### With Typos (Auto-corrected)
```
Send 100 USD to Ramkrisna from Salery Account
Pay Alice Jonson from Savngs Acount
Transfer to Ramsh using FasterPyment
```

## 🔍 Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run your code
extractor = EnhancedPaymentExtractor()
entity = extractor.parse("Send 100 USD to Alice")
```

### Check Database Contents

```python
from advanced_ml_components import PaymentDatabase

db = PaymentDatabase("sqlite:///payments.db")
transactions = db.get_recent_transactions(count=100)

for txn in transactions:
    print(txn)
```

### Test Individual Components

```python
# Test ML classifier
from advanced_ml_components import MLIntentClassifier
classifier = MLIntentClassifier()
intent, confidence = classifier.predict("Send money to John")
print(f"Intent: {intent}, Confidence: {confidence}")

# Test NER
from advanced_ml_components import NERExtractor
ner = NERExtractor()
entities = ner.extract_entities("Pay 500 USD to Alice tomorrow")
print(entities)

# Test fuzzy matching
from advanced_ml_components import FuzzyMatcher
matcher = FuzzyMatcher()
result = matcher.fuzzy_match_recipient("Ramkrisna")
print(result)
```

## 🐛 Common Issues

### Issue: "Module not found"
**Solution**: Ensure all files are in the same directory or add to PYTHONPATH

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/payment-system"
```

### Issue: "spaCy model not found"
**Solution**: Download the model

```bash
python -m spacy download en_core_web_sm
```

### Issue: "Database locked"
**Solution**: Use proper database URL or upgrade to PostgreSQL

```python
# SQLite with timeout
db_url = "sqlite:///payments.db?check_same_thread=False"

# Or use PostgreSQL
db_url = "postgresql://user:pass@localhost/payments"
```

### Issue: Low intent classification accuracy
**Solution**: Add more training data

```python
from advanced_ml_components import MLIntentClassifier

classifier = MLIntentClassifier()
classifier.training_data.extend([
    ("your new example", IntentType.MAKE_PAYMENT),
    # ... more examples
])
classifier._train_sklearn_model()
```

## 📈 Performance Tips

1. **Use caching** for repeated queries
2. **Batch processing** for multiple transactions
3. **Async API calls** for better throughput
4. **Database indexing** for faster lookups
5. **Model quantization** for faster inference

## 🚀 Production Checklist

- [ ] Set `PAYMENT_API_ENV=production`
- [ ] Use PostgreSQL instead of SQLite
- [ ] Configure proper API keys
- [ ] Enable HTTPS for API calls
- [ ] Set up monitoring and alerts
- [ ] Configure backup strategy
- [ ] Enable audit logging
- [ ] Implement rate limiting
- [ ] Set up error tracking (Sentry)
- [ ] Configure load balancer

## 📞 Support

- 📖 Full Documentation: See `ADVANCED_README.md`
- 🐛 Issues: GitHub Issues
- 💬 Discussions: GitHub Discussions
- 📧 Email: support@payment-system.com

## 🎓 Learning Resources

1. **Intent Classification**: `advanced_demo.py ml`
2. **NER**: `advanced_demo.py ner`
3. **Multi-language**: `advanced_demo.py multilang`
4. **Complete Workflow**: `advanced_demo.py workflow`
5. **Interactive CLI**: `python payment_cli.py`

---

**Ready to process payments? Run:**

```bash
python payment_cli.py
```

🎉 **Enjoy your advanced payment system!**
