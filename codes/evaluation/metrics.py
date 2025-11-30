"""
Evaluation metrics for TransMLLM

Implements BLEU, COMET, ROUGE, and BERTScore metrics for translation evaluation.
"""

from typing import List, Dict, Optional
import logging

import sacrebleu
from rouge_score import rouge_scorer

# Optional imports with fallback
try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    logging.warning("COMET not available. Install with: pip install unbabel-comet")

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    logging.warning("BERTScore not available. Install with: pip install bert-score")


def compute_bleu(
    predictions: List[str],
    references: List[str],
    tokenize: str = "13a"
) -> float:
    """
    Compute BLEU score.
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations
        tokenize: Tokenization method (13a, zh, ja, etc.)
    
    Returns:
        BLEU score
    """
    if len(predictions) != len(references):
        raise ValueError(f"Predictions ({len(predictions)}) and references ({len(references)}) must have same length")
    
    if not predictions:
        return 0.0
    
    bleu = sacrebleu.corpus_bleu(
        predictions,
        [references],
        tokenize=tokenize
    )
    return bleu.score


def compute_comet(
    predictions: List[str],
    references: List[str],
    sources: Optional[List[str]] = None
) -> Optional[float]:
    """
    Compute COMET score.
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations
        sources: List of source texts (required for COMET)
    
    Returns:
        COMET score or None if unavailable
    """
    if not COMET_AVAILABLE:
        logging.warning("COMET not available, skipping")
        return None
    
    if len(predictions) != len(references):
        raise ValueError(f"Predictions ({len(predictions)}) and references ({len(references)}) must have same length")
    
    if sources is None:
        logging.warning("Sources not provided for COMET, using empty strings")
        sources = [""] * len(predictions)
    
    if len(sources) != len(predictions):
        raise ValueError(f"Sources ({len(sources)}) must have same length as predictions ({len(predictions)})")
    
    if not predictions:
        return 0.0
    
    try:
        # Download model if needed
        model_path = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(model_path)
        
        data = [
            {"src": src, "mt": pred, "ref": ref}
            for src, pred, ref in zip(sources, predictions, references)
        ]
        
        scores, _ = model.predict(data, batch_size=8, gpus=1)
        return float(sum(scores) / len(scores))
    except Exception as e:
        logging.error(f"COMET computation failed: {e}")
        return None


def compute_rouge(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations
    
    Returns:
        Dictionary with rouge1, rouge2, rougeL scores
    """
    if len(predictions) != len(references):
        raise ValueError(f"Predictions ({len(predictions)}) and references ({len(references)}) must have same length")
    
    if not predictions:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )
    
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores["rouge1"].append(score["rouge1"].fmeasure)
        scores["rouge2"].append(score["rouge2"].fmeasure)
        scores["rougeL"].append(score["rougeL"].fmeasure)
    
    return {
        "rouge1": sum(scores["rouge1"]) / len(scores["rouge1"]),
        "rouge2": sum(scores["rouge2"]) / len(scores["rouge2"]),
        "rougeL": sum(scores["rougeL"]) / len(scores["rougeL"]),
    }


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    lang: str = "en"
) -> Optional[Dict[str, float]]:
    """
    Compute BERTScore.
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations
        lang: Language code for BERTScore
    
    Returns:
        Dictionary with precision, recall, f1 scores or None if unavailable
    """
    if not BERTSCORE_AVAILABLE:
        logging.warning("BERTScore not available, skipping")
        return None
    
    if len(predictions) != len(references):
        raise ValueError(f"Predictions ({len(predictions)}) and references ({len(references)}) must have same length")
    
    if not predictions:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    try:
        P, R, F1 = bert_score(
            predictions,
            references,
            lang=lang,
            verbose=False
        )
        return {
            "precision": float(P.mean().item()),
            "recall": float(R.mean().item()),
            "f1": float(F1.mean().item())
        }
    except Exception as e:
        logging.error(f"BERTScore computation failed: {e}")
        return None
