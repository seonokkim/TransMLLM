"""
Unit tests for evaluation metrics
"""

import pytest

from codes.evaluation.metrics import (
    compute_bleu,
    compute_rouge,
    compute_bertscore
)


def test_bleu_computation():
    """Test BLEU score computation"""
    predictions = ["This is a test", "Another example"]
    references = ["This is a test", "Another example"]
    
    score = compute_bleu(predictions, references)
    assert isinstance(score, float)
    assert 0 <= score <= 100


def test_rouge_computation():
    """Test ROUGE score computation"""
    predictions = ["This is a test", "Another example"]
    references = ["This is a test", "Another example"]
    
    scores = compute_rouge(predictions, references)
    assert "rouge1" in scores
    assert "rouge2" in scores
    assert "rougeL" in scores


def test_bertscore_computation():
    """Test BERTScore computation"""
    predictions = ["This is a test", "Another example"]
    references = ["This is a test", "Another example"]
    
    scores = compute_bertscore(predictions, references)
    if scores is not None:
        assert "precision" in scores
        assert "recall" in scores
        assert "f1" in scores

