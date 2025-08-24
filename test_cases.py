import unittest
import json
from typing import Dict, Any, List
from review_insights_system import ReviewInsightsEngine, ReviewInsights


class TestReviewInsightsEngine(unittest.TestCase):
    """Test cases for the ReviewInsightsEngine"""
    
    def setUp(self):
        """Set up test environment"""
        self.engine = ReviewInsightsEngine()
        
    def test_positive_review_extraction(self):
        """Test Case 1: Positive Review with Fast Delivery"""
        
        # Test input
        test_review = {
            "review_id": "R67890",
            "date": "2025-01-05", 
            "rating": "★★★★★ (5 stars)",
            "text": "Delivery was super fast and the rider was polite. The food was hot and fresh. Best experience so far!"
        }
        
        # Expected results
        expected = {
            "metadata": {
                "review_id": "R67890",
                "rating": 5,
                "is_valid": True
            },
            "sentiment": {
                "overall_sentiment": "positive",
                "confidence": "> 0.5"
            },
            "topics": {
                "primary_topics": ["delivery", "food_quality", "service"],
                "business_categories": {
                    "delivery": "> 0",
                    "food_quality": "> 0"
                }
            },
            "problems": {
                "problems_found": [],
                "requires_attention": False,
                "severity_score": "< 0.3"
            },
            "solutions": {
                "priority_level": "low",
                "actionable_insights": ["should include positive case study mention"]
            }
        }
        
        # Process review
        result = self.engine.process_review(test_review)
        
        # Assertions
        self.assertEqual(result.metadata.review_id, expected["metadata"]["review_id"])
        self.assertEqual(result.metadata.rating, expected["metadata"]["rating"])
        self.assertTrue(result.metadata.is_valid)
        
        self.assertEqual(result.sentiment.overall_sentiment, expected["sentiment"]["overall_sentiment"])
        self.assertGreater(result.sentiment.confidence, 0.5)
        
        self.assertIn("delivery", result.topics.business_categories)
        self.assertIn("food_quality", result.topics.business_categories)
        self.assertGreater(result.topics.business_categories.get("delivery", 0), 0)
        
        self.assertEqual(len(result.problems.problems_found), 0)
        self.assertFalse(result.problems.requires_attention)
        self.assertLess(result.problems.severity_score, 0.3)
        
        self.assertEqual(result.solutions.priority_level, "low")
        
        print("✅ Test Case 1 (Positive Review): PASSED")
        self._print_test_results("Positive Review Test", test_review, result, expected)
        
    def test_negative_review_extraction(self):
        """Test Case 2: Negative Review with Multiple Issues"""
        
        # Test input
        test_review = {
            "review_id": "R24680",
            "date": "2025-01-07",
            "rating": "★☆☆☆☆ (1 star)", 
            "text": "My order arrived late and items were missing. I contacted support but got no response."
        }
        
        # Expected results
        expected = {
            "metadata": {
                "review_id": "R24680",
                "rating": 1,
                "is_valid": True
            },
            "sentiment": {
                "overall_sentiment": "negative",
                "confidence": "> 0.3"
            },
            "topics": {
                "primary_topics": ["delivery", "service"],
                "business_categories": {
                    "delivery": "> 0",
                    "service": "> 0"
                }
            },
            "problems": {
                "problems_found": "> 0",
                "requires_attention": True,
                "severity_score": "> 0.5"
            },
            "solutions": {
                "priority_level": "high",
                "suggested_actions": [
                    "improve delivery time",
                    "enhance customer service"
                ],
                "actionable_insights": ["immediate follow-up recommended"]
            }
        }
        
        # Process review
        result = self.engine.process_review(test_review)
        
        # Assertions
        self.assertEqual(result.metadata.review_id, expected["metadata"]["review_id"])
        self.assertEqual(result.metadata.rating, expected["metadata"]["rating"])
        self.assertTrue(result.metadata.is_valid)
        
        self.assertEqual(result.sentiment.overall_sentiment, expected["sentiment"]["overall_sentiment"])
        self.assertGreater(result.sentiment.confidence, 0.3)
        
        self.assertIn("delivery", result.topics.business_categories)
        self.assertIn("service", result.topics.business_categories)
        
        self.assertGreater(len(result.problems.problems_found), 0)
        self.assertTrue(result.problems.requires_attention)
        self.assertGreater(result.problems.severity_score, 0.5)
        
        self.assertEqual(result.solutions.priority_level, "high")
        self.assertGreater(len(result.solutions.suggested_actions), 0)
        
        print("✅ Test Case 2 (Negative Review): PASSED")
        self._print_test_results("Negative Review Test", test_review, result, expected)
        
    def test_neutral_review_extraction(self):
        """Test Case 3: Neutral Review with UI/UX Feedback"""
        
        # Test input
        test_review = {
            "review_id": "R13579",
            "date": "2025-01-10",
            "rating": "★★★☆☆ (3 stars)",
            "text": "The app UI looks good but is confusing to navigate. Please make the checkout process simpler."
        }
        
        # Expected results
        expected = {
            "metadata": {
                "review_id": "R13579", 
                "rating": 3,
                "is_valid": True
            },
            "sentiment": {
                "overall_sentiment": "neutral or mixed",
                "confidence": "> 0.1"
            },
            "topics": {
                "primary_topics": ["app_ui"],
                "business_categories": {
                    "app_ui": "> 0"
                }
            },
            "problems": {
                "problems_found": "> 0",
                "requires_attention": True,
                "severity_score": "0.2 - 0.6"
            },
            "solutions": {
                "priority_level": "medium",
                "suggested_actions": [
                    "simplify user interface",
                    "ux testing"
                ],
                "improvement_areas": ["technology"]
            }
        }
        
        # Process review
        result = self.engine.process_review(test_review)
        
        # Assertions
        self.assertEqual(result.metadata.review_id, expected["metadata"]["review_id"])
        self.assertEqual(result.metadata.rating, expected["metadata"]["rating"])
        self.assertTrue(result.metadata.is_valid)
        
        self.assertIn(result.sentiment.overall_sentiment, ["neutral", "negative", "mixed"])
        self.assertGreater(result.sentiment.confidence, 0.1)
        
        self.assertIn("app_ui", result.topics.business_categories)
        self.assertGreater(result.topics.business_categories.get("app_ui", 0), 0)
        
        self.assertGreater(len(result.problems.problems_found), 0)
        self.assertTrue(result.problems.requires_attention)
        self.assertGreaterEqual(result.problems.severity_score, 0.2)
        self.assertLessEqual(result.problems.severity_score, 0.6)
        
        self.assertEqual(result.solutions.priority_level, "medium")
        self.assertGreater(len(result.solutions.suggested_actions), 0)
        self.assertIn("technology", result.solutions.improvement_areas)
        
        print("✅ Test Case 3 (Neutral/UI Review): PASSED")
        self._print_test_results("Neutral UI Review Test", test_review, result, expected)
        
    def test_invalid_review_handling(self):
        """Test Case 4: Invalid Review Data Handling"""
        
        # Test input with missing/invalid data
        test_review = {
            "review_id": "",
            "date": "invalid-date",
            "rating": "no stars",
            "text": ""
        }
        
        # Expected results
        expected = {
            "metadata": {
                "is_valid": False,
                "review_id": "UNKNOWN or ERROR"
            },
            "sentiment": {
                "overall_sentiment": "neutral",
                "confidence": "low"
            },
            "graceful_degradation": True
        }
        
        # Process review
        result = self.engine.process_review(test_review)
        
        # Assertions
        self.assertFalse(result.metadata.is_valid)
        self.assertIn(result.metadata.review_id, ["UNKNOWN", "ERROR", ""])
        self.assertEqual(result.sentiment.overall_sentiment, "neutral")
        
        print("✅ Test Case 4 (Invalid Data): PASSED")
        self._print_test_results("Invalid Data Test", test_review, result, expected)
        
    def test_batch_processing(self):
        """Test Case 5: Batch Processing Efficiency"""
        
        # Test input - multiple reviews
        test_reviews = [
            {
                "review_id": "R001",
                "date": "2025-01-01", 
                "rating": "★★★★★ (5 stars)",
                "text": "Excellent service and fast delivery!"
            },
            {
                "review_id": "R002",
                "date": "2025-01-02",
                "rating": "★★☆☆☆ (2 stars)", 
                "text": "Food was cold and delivery took too long."
            },
            {
                "review_id": "R003",
                "date": "2025-01-03",
                "rating": "★★★★☆ (4 stars)",
                "text": "Good food quality but app could be more user-friendly."
            }
        ]
        
        # Expected results
        expected = {
            "total_processed": 3,
            "all_have_metadata": True,
            "sentiment_distribution": {
                "positive": ">= 1",
                "negative": ">= 1", 
                "neutral_or_mixed": ">= 0"
            },
            "processing_time": "< 10 seconds"
        }
        
        # Process batch
        import time
        start_time = time.time()
        results = self.engine.process_batch(test_reviews)
        processing_time = time.time() - start_time
        
        # Assertions
        self.assertEqual(len(results), expected["total_processed"])
        self.assertTrue(all(r.metadata.review_id for r in results))
        
        sentiments = [r.sentiment.overall_sentiment for r in results]
        self.assertIn("positive", sentiments)
        self.assertIn("negative", sentiments)
        
        self.assertLess(processing_time, 10)
        
        print("✅ Test Case 5 (Batch Processing): PASSED")
        print(f"   Processed {len(results)} reviews in {processing_time:.2f} seconds")
        
    def _print_test_results(self, test_name: str, input_data: Dict[str, Any], 
                           actual_result: ReviewInsights, expected: Dict[str, Any]):
        """Helper method to print detailed test results"""
        
        print(f"\n{'='*50}")
        print(f"TEST: {test_name}")
        print(f"{'='*50}")
        
        print(f"\n📥 INPUT:")
        print(f"   Review ID: {input_data.get('review_id', 'N/A')}")
        print(f"   Rating: {input_data.get('rating', 'N/A')}")
        print(f"   Text: {input_data.get('text', 'N/A')[:60]}...")
        
        print(f"\n📤 EXPECTED vs 📊 ACTUAL:")
        
        # Metadata comparison
        print(f"   Review ID: {actual_result.metadata.review_id}")
        print(f"   Rating: {actual_result.metadata.rating}/5")
        print(f"   Valid: {actual_result.metadata.is_valid}")
        
        # Sentiment comparison
        print(f"   Sentiment: {actual_result.sentiment.overall_sentiment}")
        print(f"   Confidence: {actual_result.sentiment.confidence:.3f}")
        
        # Topics comparison
        print(f"   Primary Topics: {actual_result.topics.primary_topics}")
        print(f"   Business Categories: {list(actual_result.topics.business_categories.keys())}")
        
        # Problems comparison
        print(f"   Problems Found: {len(actual_result.problems.problems_found)}")
        print(f"   Severity Score: {actual_result.problems.severity_score:.3f}")
        print(f"   Requires Attention: {actual_result.problems.requires_attention}")
        
        # Solutions comparison
        print(f"   Priority Level: {actual_result.solutions.priority_level}")
        print(f"   Suggested Actions: {len(actual_result.solutions.suggested_actions)}")
        if actual_result.solutions.suggested_actions:
            for action in actual_result.solutions.suggested_actions[:2]:
                print(f"     • {action}")
        
        print(f"   Actionable Insights: {len(actual_result.solutions.actionable_insights)}")
        for insight in actual_result.solutions.actionable_insights:
            print(f"     • {insight}")
            
        print(f"\n✅ ACCURACY ASSESSMENT:")
        
        # Check metadata accuracy
        metadata_accuracy = (
            actual_result.metadata.review_id == input_data.get('review_id') and
            actual_result.metadata.rating > 0 and
            actual_result.metadata.is_valid
        )
        print(f"   Metadata Extraction: {'✅ ACCURATE' if metadata_accuracy else '❌ NEEDS IMPROVEMENT'}")
        
        # Check sentiment accuracy
        expected_sentiment = self._determine_expected_sentiment(input_data.get('text', ''), 
                                                              actual_result.metadata.rating)
        sentiment_accuracy = actual_result.sentiment.overall_sentiment == expected_sentiment
        print(f"   Sentiment Analysis: {'✅ ACCURATE' if sentiment_accuracy else '⚠️  ACCEPTABLE'}")
        
        # Check topic relevance
        text = input_data.get('text', '').lower()
        topic_accuracy = any(
            any(keyword in text for keyword in self.engine.business_categories.get(topic, []))
            for topic in actual_result.topics.primary_topics
        )
        print(f"   Topic Detection: {'✅ RELEVANT' if topic_accuracy else '⚠️  REVIEW NEEDED'}")
        
        # Check problem detection logic
        problem_accuracy = (
            (actual_result.metadata.rating <= 2 and actual_result.problems.requires_attention) or
            (actual_result.metadata.rating >= 4 and not actual_result.problems.requires_attention) or
            (actual_result.metadata.rating == 3)
        )
        print(f"   Problem Detection: {'✅ LOGICAL' if problem_accuracy else '❌ NEEDS REVIEW'}")
        
        print(f"\n{'='*50}\n")
    
    def _determine_expected_sentiment(self, text: str, rating: int) -> str:
        """Helper method to determine expected sentiment based on text and rating"""
        if rating >= 4:
            return "positive"
        elif rating <= 2:
            return "negative" 
        else:
            # For rating 3, analyze text for mixed signals
            positive_words = ['good', 'nice', 'great', 'excellent', 'amazing', 'love', 'best']
            negative_words = ['bad', 'terrible', 'awful', 'worst', 'hate', 'poor', 'disappointing']
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                return "positive"
            elif neg_count > pos_count:
                return "negative"
            else:
                return "neutral"


def run_accuracy_analysis():
    """Run comprehensive accuracy analysis with detailed reporting"""
    
    print("🚀 Starting Comprehensive Accuracy Analysis...")
    print("="*70)
    
    engine = ReviewInsightsEngine()
    
    # Comprehensive test dataset
    test_dataset = [
        {
            "review_id": "ACC001",
            "date": "2025-01-15",
            "rating": "★★★★★ (5 stars)", 
            "text": "Amazing food quality and lightning fast delivery! The rider was super friendly and professional.",
            "expected_sentiment": "positive",
            "expected_topics": ["delivery", "food_quality", "service"],
            "expected_attention": False
        },
        {
            "review_id": "ACC002", 
            "date": "2025-01-16",
            "rating": "★☆☆☆☆ (1 star)",
            "text": "Worst experience ever! Food was cold, delivery took 2 hours, and customer service was rude and unhelpful.",
            "expected_sentiment": "negative",
            "expected_topics": ["delivery", "food_quality", "service"],
            "expected_attention": True
        },
        {
            "review_id": "ACC003",
            "date": "2025-01-17", 
            "rating": "★★★☆☆ (3 stars)",
            "text": "Food was decent but the app interface is really confusing. Checkout process needs to be simplified.",
            "expected_sentiment": "neutral",
            "expected_topics": ["app_ui", "food_quality"],
            "expected_attention": True
        },
        {
            "review_id": "ACC004",
            "date": "2025-01-18",
            "rating": "★★★★☆ (4 stars)",
            "text": "Great service overall! Only issue was packaging could be better to prevent spills.",
            "expected_sentiment": "positive", 
            "expected_topics": ["service", "packaging"],
            "expected_attention": False
        },
        {
            "review_id": "ACC005",
            "date": "2025-01-19",
            "rating": "★★☆☆☆ (2 stars)",
            "text": "Order was missing items and the food that arrived was stale. Very disappointed with the quality.",
            "expected_sentiment": "negative",
            "expected_topics": ["food_quality"],
            "expected_attention": True
        }
    ]
    
    # Process all reviews
    results = []
    for review in test_dataset:
        result = engine.process_review(review)
        results.append((review, result))
    
    # Calculate accuracy metrics
    total_tests = len(test_dataset)
    sentiment_correct = 0
    topic_relevant = 0
    attention_correct = 0
    
    print(f"\n📊 DETAILED ACCURACY ANALYSIS:")
    print(f"{'='*70}")
    
    for i, (expected, actual) in enumerate(results):
        print(f"\nTest {i+1}: {expected['review_id']}")
        print(f"Text: {expected['text'][:50]}...")
        
        # Sentiment accuracy
        sentiment_match = actual.sentiment.overall_sentiment == expected['expected_sentiment']
        if sentiment_match:
            sentiment_correct += 1
        print(f"  Sentiment: Expected {expected['expected_sentiment']} | Got {actual.sentiment.overall_sentiment} | {'✅' if sentiment_match else '❌'}")
        
        # Topic relevance
        topic_match = any(topic in expected['expected_topics'] for topic in actual.topics.primary_topics)
        if topic_match:
            topic_relevant += 1
        print(f"  Topics: Expected {expected['expected_topics']} | Got {actual.topics.primary_topics} | {'✅' if topic_match else '❌'}")
        
        # Attention flag accuracy
        attention_match = actual.problems.requires_attention == expected['expected_attention']
        if attention_match:
            attention_correct += 1
        print(f"  Attention: Expected {expected['expected_attention']} | Got {actual.problems.requires_attention} | {'✅' if attention_match else '❌'}")
    
    # Final accuracy report
    print(f"\n🎯 FINAL ACCURACY SCORES:")
    print(f"{'='*70}")
    print(f"Sentiment Analysis: {sentiment_correct}/{total_tests} ({sentiment_correct/total_tests*100:.1f}%)")
    print(f"Topic Detection: {topic_relevant}/{total_tests} ({topic_relevant/total_tests*100:.1f}%)")
    print(f"Attention Flagging: {attention_correct}/{total_tests} ({attention_correct/total_tests*100:.1f}%)")
    print(f"Overall System Accuracy: {(sentiment_correct+topic_relevant+attention_correct)/(total_tests*3)*100:.1f}%")
    
    return results


if __name__ == "__main__":
    # Run unit tests
    print("🧪 Running Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run accuracy analysis
    print("\n" + "="*70)
    run_accuracy_analysis()