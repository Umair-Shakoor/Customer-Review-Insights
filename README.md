# AI-Powered Customer Review Insights System

A comprehensive system that converts unstructured customer review text into actionable business insights using advanced NLP and machine learning techniques.

## 🚀 Features

### Core Capabilities
- **Automated Metadata Extraction**: Review ID, date, rating validation and normalization
- **Advanced Sentiment Analysis**: Multi-model approach using VADER and TextBlob
- **Intelligent Topic Detection**: Business-category mapping and keyword extraction  
- **Problem Identification**: Pattern-based issue detection with severity scoring
- **Solution Generation**: Actionable improvement recommendations with priority levels
- **Batch Processing**: Efficient processing of large review datasets
- **Export Functionality**: JSON and CSV output formats

### Business Intelligence
- **Priority-based Alerting**: Automatic flagging of reviews requiring immediate attention
- **Trend Analysis**: Topic and sentiment patterns over time
- **Actionable Insights**: Direct recommendations for operational improvements
- **Performance Metrics**: Confidence scoring and accuracy indicators

## 🏗️ Architecture

```
Raw Reviews → Data Processing → NLP/AI Analysis → Structured Insights → Business Actions
     ↓              ↓               ↓                ↓                  ↓
• JSON/CSV    • Validation    • Sentiment      • JSON Export    • Dashboards
• API Feed    • Cleaning      • Topics         • CSV Reports    • Alerts  
• Real-time   • Metadata      • Problems       • Database       • KPIs
• Batch       • Normalization • Solutions      • Analytics      • Actions
```

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-review-insights.git
cd ai-review-insights
```

2. **Create virtual environment**
```bash
python -m venv review_insights_env
source review_insights_env/bin/activate  # On Windows: review_insights_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```bash
python -c "import nltk; nltk.download(['vader_lexicon', 'punkt', 'stopwords'])"
```

### Dependencies (requirements.txt)
```
nltk>=3.8
textblob>=0.17.1
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
```

## 🛠️ Usage

### Basic Usage

```python
from review_insights_system import ReviewInsightsEngine

# Initialize the engine
engine = ReviewInsightsEngine()

# Process a single review
review_data = {
    "review_id": "R12345",
    "date": "2025-01-20",
    "rating": "★★★★☆ (4 stars)", 
    "text": "Great food quality but delivery was a bit slow."
}

insights = engine.process_review(review_data)
print(f"Sentiment: {insights.sentiment.overall_sentiment}")
print(f"Topics: {insights.topics.primary_topics}")
print(f"Requires Attention: {insights.problems.requires_attention}")
```

### Batch Processing

```python
# Process multiple reviews
reviews = [
    {
        "review_id": "R001",
        "date": "2025-01-20",
        "rating": "★★★★★ (5 stars)",
        "text": "Excellent service!"
    },
    {
        "review_id": "R002", 
        "date": "2025-01-21",
        "rating": "★★☆☆☆ (2 stars)",
        "text": "Food was cold and late."
    }
]

results = engine.process_batch(reviews)

# Export results
json_file = engine.export_insights(results, 'json')
csv_file = engine.export_insights(results, 'csv')
```

### Advanced Usage

```python
# Custom configuration
engine = ReviewInsightsEngine()

# Add custom business categories
engine.business_categories['customer_service'] = [
    'support', 'help', 'staff', 'representative', 'agent'
]

# Process with custom settings
insights = engine.process_review(review_data)

# Access detailed results
print(f"Sentiment Confidence: {insights.sentiment.confidence}")
print(f"Business Categories: {insights.topics.business_categories}")
print(f"Problem Severity: {insights.problems.severity_score}")
print(f"Suggested Actions: {insights.solutions.suggested_actions}")
```

## 📊 Output Format

### JSON Structure
```json
{
  "metadata": {
    "review_id": "R12345",
    "date": "2025-01-20", 
    "rating": 4,
    "rating_stars": "★★★★☆ (4 stars)",
    "is_valid": true,
    "extracted_at": "2025-01-20T10:30:00"
  },
  "sentiment": {
    "overall_sentiment": "positive",
    "confidence": 0.756,
    "positive_score": 0.423,
    "negative_score": 0.156,
    "neutral_score": 0.421,
    "compound_score": 0.267
  },
  "topics": {
    "primary_topics": ["food_quality", "delivery"],
    "topic_keywords": {
      "food_quality": ["food", "quality", "great"],
      "delivery": ["delivery", "slow"]
    },
    "business_categories": {
      "food_quality": 0.125,
      "delivery": 0.083
    },
    "key_phrases": ["Great food quality", "delivery was a bit slow"]
  },
  "problems": {
    "problems_found": [
      {
        "type": "slow",
        "severity": "medium_severity",
        "matches": ["slow"],
        "context": "Great food quality but delivery was a bit slow."
      }
    ],
    "problem_categories": ["medium_severity"],
    "severity_score": 0.325,
    "requires_attention": true
  },
  "solutions": {
    "suggested_actions": [
      "Improve delivery time tracking and notifications",
      "Implement delivery partner training program"
    ],
    "improvement_areas": ["delivery_operations"],
    "priority_level": "medium",
    "actionable_insights": [
      "Focus improvement on: food_quality, delivery"
    ]
  }
}
```

### CSV Format
The CSV export includes flattened key metrics:
- review_id, date, rating
- sentiment, sentiment_confidence  
- primary_topics, problems_count
- severity_score, requires_attention
- priority_level, suggested_actions_count

## 🧪 Testing

### Run All Tests
```bash
python test_cases.py
```

### Run Specific Test Cases
```bash
python -m unittest test_cases.TestReviewIns