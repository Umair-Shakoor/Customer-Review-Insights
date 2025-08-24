import json
import re
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import logging

# NLP and ML libraries
import nltk
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    print("Warning: Could not download NLTK data. Some features may not work.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReviewMetadata:
    """Structure for review metadata"""
    review_id: str
    date: str
    rating: int
    rating_stars: str
    is_valid: bool
    extracted_at: str


@dataclass
class SentimentAnalysis:
    """Structure for sentiment analysis results"""
    overall_sentiment: str
    confidence: float
    positive_score: float
    negative_score: float
    neutral_score: float
    compound_score: float


@dataclass
class TopicInsights:
    """Structure for topic analysis results"""
    primary_topics: List[str]
    topic_keywords: Dict[str, List[str]]
    business_categories: Dict[str, float]
    key_phrases: List[str]


@dataclass
class ProblemDetection:
    """Structure for identified problems"""
    problems_found: List[Dict[str, Any]]
    problem_categories: List[str]
    severity_score: float
    requires_attention: bool


@dataclass
class SolutionSuggestions:
    """Structure for improvement suggestions"""
    suggested_actions: List[str]
    improvement_areas: List[str]
    priority_level: str
    actionable_insights: List[str]


@dataclass
class ReviewInsights:
    """Complete review insights structure"""
    metadata: ReviewMetadata
    sentiment: SentimentAnalysis
    topics: TopicInsights
    problems: ProblemDetection
    solutions: SolutionSuggestions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class ReviewInsightsEngine:
    """
    Main engine for processing customer reviews and extracting structured insights.
    """
    
    def __init__(self):
        """Initialize the review insights engine"""
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Business-specific categories and keywords
        self.business_categories = {
            'delivery': ['delivery', 'deliver', 'fast', 'quick', 'slow', 'late', 'arrived', 'time'],
            'food_quality': ['food', 'taste', 'fresh', 'hot', 'cold', 'quality', 'delicious', 'stale'],
            'service': ['service', 'staff', 'polite', 'rude', 'helpful', 'support', 'customer'],
            'app_ui': ['app', 'ui', 'interface', 'navigate', 'confusing', 'easy', 'checkout', 'website'],
            'pricing': ['price', 'expensive', 'cheap', 'cost', 'value', 'money', 'affordable'],
            'packaging': ['packaging', 'packed', 'container', 'box', 'wrap', 'leak']
        }
        
        # Problem patterns and severity indicators
        self.problem_patterns = {
            'high_severity': [
                r'missing.*item', r'wrong.*order', r'food.*poison', r'refund',
                r'terrible', r'worst', r'disgusting', r'horrible'
            ],
            'medium_severity': [
                r'late.*delivery', r'cold.*food', r'poor.*service', r'confusing',
                r'difficult', r'slow', r'problem', r'issue'
            ],
            'low_severity': [
                r'could.*better', r'suggest', r'improve', r'minor', r'small'
            ]
        }
        
        logger.info("ReviewInsightsEngine initialized successfully")
    
    def extract_metadata(self, review_data: Dict[str, Any]) -> ReviewMetadata:
        """Extract and validate review metadata"""
        try:
            # Extract review ID
            review_id = review_data.get('review_id', 'UNKNOWN')
            
            # Parse date
            date_str = review_data.get('date', '')
            try:
                # Try to parse date and standardize format
                parsed_date = datetime.strptime(date_str, '%Y-%m-%d')
                standardized_date = parsed_date.strftime('%Y-%m-%d')
            except:
                standardized_date = date_str
            
            # Extract rating
            rating_raw = review_data.get('rating', '')
            rating_stars = rating_raw
            
            # Extract numeric rating from star format
            rating_match = re.search(r'(\d+)\s*star', rating_raw, re.IGNORECASE)
            rating = int(rating_match.group(1)) if rating_match else 0
            
            # Validation
            is_valid = all([
                review_id != 'UNKNOWN',
                date_str != '',
                1 <= rating <= 5,
                'text' in review_data
            ])
            
            return ReviewMetadata(
                review_id=review_id,
                date=standardized_date,
                rating=rating,
                rating_stars=rating_stars,
                is_valid=is_valid,
                extracted_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            return ReviewMetadata('ERROR', '', 0, '', False, datetime.now().isoformat())
    
    def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """Perform comprehensive sentiment analysis"""
        try:
            # VADER sentiment analysis
            vader_scores = self.sentiment_analyzer.polarity_scores(text)
            
            # TextBlob sentiment analysis
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            
            # Combine scores for more robust analysis
            compound_score = (vader_scores['compound'] + textblob_polarity) / 2
            
            # Determine overall sentiment
            if compound_score >= 0.1:
                overall_sentiment = 'positive'
                confidence = min(compound_score, 1.0)
            elif compound_score <= -0.1:
                overall_sentiment = 'negative' 
                confidence = min(abs(compound_score), 1.0)
            else:
                overall_sentiment = 'neutral'
                confidence = 1.0 - abs(compound_score)
            
            return SentimentAnalysis(
                overall_sentiment=overall_sentiment,
                confidence=round(confidence, 3),
                positive_score=round(vader_scores['pos'], 3),
                negative_score=round(vader_scores['neg'], 3),
                neutral_score=round(vader_scores['neu'], 3),
                compound_score=round(compound_score, 3)
            )
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return SentimentAnalysis('neutral', 0.0, 0.0, 0.0, 1.0, 0.0)
    
    def extract_topics(self, text: str) -> TopicInsights:
        """Extract topics and business categories from review text"""
        try:
            # Clean and tokenize text
            cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
            words = cleaned_text.split()
            
            # Calculate business category relevance
            business_categories = {}
            for category, keywords in self.business_categories.items():
                score = sum(1 for word in words if word in keywords) / len(words)
                if score > 0:
                    business_categories[category] = round(score, 3)
            
            # Extract key phrases using simple n-gram approach
            key_phrases = []
            sentences = text.split('.')
            for sentence in sentences[:3]:  # Top 3 sentences
                if len(sentence.strip()) > 10:
                    key_phrases.append(sentence.strip())
            
            # Extract primary topics (top business categories)
            primary_topics = sorted(business_categories.keys(), 
                                  key=business_categories.get, reverse=True)[:3]
            
            # Extract keywords for each topic
            topic_keywords = {}
            for topic in primary_topics:
                relevant_keywords = [word for word in words 
                                   if word in self.business_categories.get(topic, [])]
                topic_keywords[topic] = list(set(relevant_keywords))[:5]
            
            return TopicInsights(
                primary_topics=primary_topics,
                topic_keywords=topic_keywords,
                business_categories=business_categories,
                key_phrases=key_phrases
            )
            
        except Exception as e:
            logger.error(f"Error in topic extraction: {str(e)}")
            return TopicInsights([], {}, {}, [])
    
    def detect_problems(self, text: str, rating: int) -> ProblemDetection:
        """Detect and categorize problems mentioned in the review"""
        try:
            problems_found = []
            problem_categories = []
            text_lower = text.lower()
            
            # Check for problem patterns
            for severity, patterns in self.problem_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, text_lower)
                    if matches:
                        problems_found.append({
                            'type': pattern.replace(r'.*', '').replace(r'\w+', 'issue'),
                            'severity': severity,
                            'matches': matches,
                            'context': text[:100] + '...' if len(text) > 100 else text
                        })
                        problem_categories.append(severity)
            
            # Calculate severity score based on rating and detected problems
            rating_severity = (5 - rating) / 4  # Lower rating = higher severity
            problem_severity = len([p for p in problems_found if p['severity'] == 'high_severity']) * 0.4
            problem_severity += len([p for p in problems_found if p['severity'] == 'medium_severity']) * 0.2
            problem_severity += len([p for p in problems_found if p['severity'] == 'low_severity']) * 0.1
            
            severity_score = min((rating_severity + problem_severity) / 2, 1.0)
            requires_attention = severity_score > 0.3 or rating <= 2
            
            # Remove duplicates from categories
            problem_categories = list(set(problem_categories))
            
            return ProblemDetection(
                problems_found=problems_found,
                problem_categories=problem_categories,
                severity_score=round(severity_score, 3),
                requires_attention=requires_attention
            )
            
        except Exception as e:
            logger.error(f"Error in problem detection: {str(e)}")
            return ProblemDetection([], [], 0.0, False)
    
    def generate_solutions(self, problems: ProblemDetection, 
                          topics: TopicInsights, rating: int) -> SolutionSuggestions:
        """Generate actionable improvement suggestions"""
        try:
            suggested_actions = []
            improvement_areas = []
            
            # Generate solutions based on detected problems
            if 'delivery' in topics.primary_topics:
                if rating <= 3:
                    suggested_actions.append("Improve delivery time tracking and notifications")
                    suggested_actions.append("Implement delivery partner training program")
                    improvement_areas.append("delivery_operations")
            
            if 'food_quality' in topics.primary_topics:
                if rating <= 3:
                    suggested_actions.append("Review food quality control processes")
                    suggested_actions.append("Implement temperature monitoring during transit")
                    improvement_areas.append("food_preparation")
            
            if 'service' in topics.primary_topics:
                if rating <= 3:
                    suggested_actions.append("Enhance customer service training")
                    suggested_actions.append("Implement faster response system")
                    improvement_areas.append("customer_service")
            
            if 'app_ui' in topics.primary_topics:
                suggested_actions.append("Simplify user interface design")
                suggested_actions.append("Conduct UX testing and improvements")
                improvement_areas.append("technology")
            
            # Determine priority level
            if problems.severity_score > 0.6 or rating <= 2:
                priority_level = "high"
            elif problems.severity_score > 0.3 or rating <= 3:
                priority_level = "medium"
            else:
                priority_level = "low"
            
            # Generate actionable insights
            actionable_insights = []
            if problems.requires_attention:
                actionable_insights.append("Immediate customer follow-up recommended")
            if rating >= 4:
                actionable_insights.append("Positive experience - consider case study")
            if len(topics.primary_topics) > 0:
                actionable_insights.append(f"Focus improvement on: {', '.join(topics.primary_topics)}")
            
            return SolutionSuggestions(
                suggested_actions=suggested_actions,
                improvement_areas=improvement_areas,
                priority_level=priority_level,
                actionable_insights=actionable_insights
            )
            
        except Exception as e:
            logger.error(f"Error in solution generation: {str(e)}")
            return SolutionSuggestions([], [], "low", [])
    
    def process_review(self, review_data: Dict[str, Any]) -> ReviewInsights:
        """
        Main method to process a single review and extract all insights.
        
        Args:
            review_data: Dictionary containing review information
                        (review_id, date, rating, text)
        
        Returns:
            ReviewInsights: Complete structured insights object
        """
        try:
            logger.info(f"Processing review: {review_data.get('review_id', 'UNKNOWN')}")
            
            # Extract metadata
            metadata = self.extract_metadata(review_data)
            
            if not metadata.is_valid:
                logger.warning(f"Invalid review data: {review_data.get('review_id', 'UNKNOWN')}")
            
            # Get review text
            text = review_data.get('text', '')
            
            # Perform all analyses
            sentiment = self.analyze_sentiment(text)
            topics = self.extract_topics(text)
            problems = self.detect_problems(text, metadata.rating)
            solutions = self.generate_solutions(problems, topics, metadata.rating)
            
            # Create comprehensive insights object
            insights = ReviewInsights(
                metadata=metadata,
                sentiment=sentiment,
                topics=topics,
                problems=problems,
                solutions=solutions
            )
            
            logger.info(f"Successfully processed review: {metadata.review_id}")
            return insights
            
        except Exception as e:
            logger.error(f"Error processing review: {str(e)}")
            # Return minimal insights object on error
            return ReviewInsights(
                metadata=ReviewMetadata('ERROR', '', 0, '', False, datetime.now().isoformat()),
                sentiment=SentimentAnalysis('neutral', 0.0, 0.0, 0.0, 1.0, 0.0),
                topics=TopicInsights([], {}, {}, []),
                problems=ProblemDetection([], [], 0.0, False),
                solutions=SolutionSuggestions([], [], 'low', [])
            )
    
    def process_batch(self, reviews: List[Dict[str, Any]]) -> List[ReviewInsights]:
        """
        Process multiple reviews in batch.
        
        Args:
            reviews: List of review dictionaries
        
        Returns:
            List of ReviewInsights objects
        """
        logger.info(f"Processing batch of {len(reviews)} reviews")
        
        results = []
        for i, review in enumerate(reviews):
            try:
                insights = self.process_review(review)
                results.append(insights)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(reviews)} reviews")
                    
            except Exception as e:
                logger.error(f"Error processing review {i}: {str(e)}")
                continue
        
        logger.info(f"Batch processing complete: {len(results)}/{len(reviews)} successful")
        return results
    
    def export_insights(self, insights: List[ReviewInsights], 
                       format_type: str = 'json', filename: str = None) -> str:
        """
        Export insights to file.
        
        Args:
            insights: List of ReviewInsights objects
            format_type: Export format ('json' or 'csv')
            filename: Output filename (auto-generated if None)
        
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"review_insights_{timestamp}.{format_type}"
        
        try:
            if format_type.lower() == 'json':
                # Export as JSON
                insights_dict = [insight.to_dict() for insight in insights]
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(insights_dict, f, indent=2, ensure_ascii=False)
                
            elif format_type.lower() == 'csv':
                # Flatten insights for CSV export
                flattened_data = []
                for insight in insights:
                    row = {
                        'review_id': insight.metadata.review_id,
                        'date': insight.metadata.date,
                        'rating': insight.metadata.rating,
                        'sentiment': insight.sentiment.overall_sentiment,
                        'sentiment_confidence': insight.sentiment.confidence,
                        'primary_topics': ', '.join(insight.topics.primary_topics),
                        'problems_count': len(insight.problems.problems_found),
                        'severity_score': insight.problems.severity_score,
                        'requires_attention': insight.problems.requires_attention,
                        'priority_level': insight.solutions.priority_level,
                        'suggested_actions_count': len(insight.solutions.suggested_actions)
                    }
                    flattened_data.append(row)
                
                df = pd.DataFrame(flattened_data)
                df.to_csv(filename, index=False)
            
            logger.info(f"Insights exported to: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting insights: {str(e)}")
            return ""


def main():
    """Demo function showing how to use the ReviewInsightsEngine"""
    
    # Sample reviews for testing
    sample_reviews = [
        {
            "review_id": "R67890",
            "date": "2025-01-05",
            "rating": "★★★★★ (5 stars)",
            "text": "Delivery was super fast and the rider was polite. The food was hot and fresh. Best experience so far!"
        },
        {
            "review_id": "R24680", 
            "date": "2025-01-07",
            "rating": "★☆☆☆☆ (1 star)",
            "text": "My order arrived late and items were missing. I contacted support but got no response."
        },
        {
            "review_id": "R13579",
            "date": "2025-01-10", 
            "rating": "★★★☆☆ (3 stars)",
            "text": "The app UI looks good but is confusing to navigate. Please make the checkout process simpler."
        }
    ]
    
    # Initialize the engine
    engine = ReviewInsightsEngine()
    
    # Process reviews
    print("Processing sample reviews...\n")
    results = engine.process_batch(sample_reviews)
    
    # Display results
    for i, insights in enumerate(results):
        print(f"=== Review {i+1}: {insights.metadata.review_id} ===")
        print(f"Rating: {insights.metadata.rating}/5")
        print(f"Sentiment: {insights.sentiment.overall_sentiment} (confidence: {insights.sentiment.confidence})")
        print(f"Primary Topics: {', '.join(insights.topics.primary_topics)}")
        print(f"Problems Found: {len(insights.problems.problems_found)}")
        print(f"Requires Attention: {insights.problems.requires_attention}")
        print(f"Priority Level: {insights.solutions.priority_level}")
        print(f"Suggested Actions: {len(insights.solutions.suggested_actions)}")
        if insights.solutions.suggested_actions:
            for action in insights.solutions.suggested_actions[:2]:  # Show first 2 actions
                print(f"  - {action}")
        print(f"Actionable Insights: {', '.join(insights.solutions.actionable_insights)}")
        print("-" * 60)
    
    # Export results
    json_file = engine.export_insights(results, 'json')
    csv_file = engine.export_insights(results, 'csv')
    
    print(f"\nResults exported to:")
    print(f"JSON: {json_file}")
    print(f"CSV: {csv_file}")
    
    # Show sample JSON output
    if results:
        print(f"\n=== Sample JSON Structure ===")
        sample_json = results[0].to_dict()
        print(json.dumps(sample_json, indent=2)[:500] + "...")


if __name__ == "__main__":
    main()