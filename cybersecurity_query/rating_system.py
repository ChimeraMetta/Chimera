"""Rating and metrics system for cybersecurity queries.

Stores ratings in JSON files and computes accuracy metrics.
Uses inquirer for interactive rating collection.
"""

import json
import os
from typing import Optional
from cybersecurity_query.models import QueryResult, QueryRating, QueryMetrics


_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_RATINGS_DIR = os.path.join(_WORKSPACE_ROOT, "query_results")
_RATINGS_FILE = os.path.join(_RATINGS_DIR, "ratings.json")
_IMPROVEMENT_LOG = os.path.join(_RATINGS_DIR, "improvement_log.json")


class RatingSystem:
    """Collects and stores user ratings for query responses."""

    def __init__(self):
        os.makedirs(_RATINGS_DIR, exist_ok=True)
        self.ratings = self._load_ratings()

    def collect_rating_interactive(self, result: QueryResult) -> Optional[QueryRating]:
        """Collect a rating from the user via inquirer prompt."""
        try:
            import inquirer
            from common.logging_utils import ChimeraTheme

            questions = [
                inquirer.List(
                    "rating",
                    message="Rate this response (1=poor, 5=excellent)",
                    choices=["5 - Excellent", "4 - Good", "3 - Average", "2 - Poor", "1 - Very Poor"],
                    default="3 - Average",
                ),
            ]
            theme = ChimeraTheme()
            answers = inquirer.prompt(questions, theme=theme)

            if not answers:
                return None

            rating_val = int(answers["rating"][0])
            rating = QueryRating(
                query_id=result.query_id,
                rating=rating_val,
                query_text=result.entities.get("_raw_query", ""),
                intent=result.intent.value,
            )

            self._save_rating(rating)

            # Log low-rated queries for improvement
            if rating_val <= 2:
                self._log_low_rating(rating, result)

            return rating

        except (ImportError, Exception):
            return None

    def add_rating(self, query_id: str, rating: int, query_text: str = "",
                   intent: str = "") -> QueryRating:
        """Programmatically add a rating."""
        r = QueryRating(
            query_id=query_id,
            rating=rating,
            query_text=query_text,
            intent=intent,
        )
        self._save_rating(r)
        return r

    def get_metrics(self, translation_accuracy: float = 0.0) -> QueryMetrics:
        """Compute aggregate metrics from stored ratings."""
        ratings = self.ratings
        if not ratings:
            return QueryMetrics(translation_accuracy=translation_accuracy)

        total = len(ratings)
        avg = sum(r["rating"] for r in ratings) / total
        low = sum(1 for r in ratings if r["rating"] <= 2)

        return QueryMetrics(
            total_queries=total,
            average_rating=round(avg, 2),
            rated_queries=total,
            low_rated_queries=low,
            retrieval_accuracy=round(avg / 5.0, 2),  # Normalize to 0-1
            translation_accuracy=translation_accuracy,
            intent_accuracy=round(
                sum(1 for r in ratings if r["rating"] >= 3) / total, 2
            ) if total > 0 else 0.0,
        )

    def _load_ratings(self) -> list:
        """Load ratings from JSON file."""
        if os.path.exists(_RATINGS_FILE):
            try:
                with open(_RATINGS_FILE, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save_rating(self, rating: QueryRating):
        """Append a rating to the JSON file."""
        self.ratings.append({
            "query_id": rating.query_id,
            "rating": rating.rating,
            "query_text": rating.query_text,
            "intent": rating.intent,
            "timestamp": rating.timestamp,
            "feedback": rating.feedback,
        })
        try:
            with open(_RATINGS_FILE, "w") as f:
                json.dump(self.ratings, f, indent=2)
        except IOError:
            pass

    def _log_low_rating(self, rating: QueryRating, result: QueryResult):
        """Log low-rated queries for improvement analysis."""
        log_entry = {
            "query_id": rating.query_id,
            "rating": rating.rating,
            "query_text": rating.query_text,
            "intent": rating.intent,
            "metta_queries": result.metta_queries,
            "confidence": result.confidence,
            "timestamp": rating.timestamp,
        }

        log_data = []
        if os.path.exists(_IMPROVEMENT_LOG):
            try:
                with open(_IMPROVEMENT_LOG, "r") as f:
                    log_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                log_data = []

        log_data.append(log_entry)

        try:
            with open(_IMPROVEMENT_LOG, "w") as f:
                json.dump(log_data, f, indent=2)
        except IOError:
            pass
