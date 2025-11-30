"""
Benchmark evaluation for PATIMT-Bench and FLOES200
"""

import argparse
from pathlib import Path
from codes.evaluation.evaluate import evaluate


def evaluate_benchmark(
    model_path: str,
    benchmark: str,
    test_data_path: str,
    output_dir: str,
    **kwargs
):
    """
    Evaluate on benchmark dataset
    
    Args:
        model_path: Path to trained model
        benchmark: Benchmark name ("patimt" or "floes200")
        test_data_path: Path to test data
        output_dir: Output directory
        **kwargs: Additional arguments
    """
    if benchmark.lower() == "patimt":
        metrics = ["bleu", "comet", "rouge", "bertscore"]
    elif benchmark.lower() == "floes200":
        metrics = ["bleu", "comet", "rouge", "bertscore"]
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    
    results = evaluate(
        model_path=model_path,
        test_data_path=test_data_path,
        output_dir=output_dir,
        metrics=metrics,
        **kwargs
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--benchmark", type=str, required=True, choices=["patimt", "floes200"],
                       help="Benchmark name")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to test data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    
    args = parser.parse_args()
    
    results = evaluate_benchmark(
        model_path=args.model_path,
        benchmark=args.benchmark,
        test_data_path=args.test_data_path,
        output_dir=args.output_dir,
    )
    
    print(f"Benchmark: {args.benchmark}")
    print("Results:")
    for metric, score in results.items():
        print(f"  {metric}: {score}")


if __name__ == "__main__":
    main()

