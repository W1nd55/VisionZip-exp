import sys
import os
import argparse
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real
from typing import Tuple, Dict, Any, List
from subprocess import Popen, PIPE, CalledProcessError
import subprocess
import yaml

sys.path.insert(0, os.path.abspath("."))

# Global variable to store the best results
BEST_RESULT = {"alpha": (0, 0, 0), "f1_score": -1.0}


def run_mme_evaluation(alpha_config: Tuple[float, float, float],
                       cfg_path: str,
                       mme_root: str,
                       subtask: str) -> float:
    """
    Runs MME evaluation and extracts the objective score (prioritizing acc_plus)
    from the generated mme_summary.csv. The score is the mean of the
    'mme_acc_plus' column across all rows (subtasks) in the summary.

    """
    alpha1, alpha2, alpha3 = alpha_config

    out_root = (
        "eval_results/mme_eval_results/"
        f"mme_eval_results_hybrid_attn_dif_hybrid_a{alpha1:.4f}b{alpha2:.4f}c{alpha3:.4f}"
    )
    os.makedirs(out_root, exist_ok=True)

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    if "model" not in cfg:
        cfg["model"] = {}
    cfg["model"]["vision_zip_alpha"] = [float(alpha1), float(alpha2), float(alpha3)]

    tmp_cfg_path = os.path.join(out_root, "config_alpha.yaml")
    with open(tmp_cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    command = [
        "python",
        "tools/mme_run_all.py",
        "--cfg", tmp_cfg_path,
        "--mme_root", mme_root,
        "--out_root", out_root,
    ]

    if subtask:
        command.extend(["--only", subtask])

    print(f"--- Running Evaluation for Alpha: {alpha_config} ---")
    print(f"cfg: {tmp_cfg_path}")
    print(f"Command: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )

        if result.stderr:
            print("\n--- Subprocess Stderr (Error Detail) ---")
            print(result.stderr)
            print("------------------------------------------\n")

        summary_path = os.path.join(out_root, "mme_summary.csv")
        if not os.path.exists(summary_path):
            print(f"ERROR: Summary file not generated at {summary_path}")
            return 0.0

        df = pd.read_csv(summary_path)
        if df.empty or 'mme_acc' not in df.columns or 'mme_acc_plus' not in df.columns:
            print("ERROR: CSV is empty or missing required columns.")
            return 0.0

        objective_score = df['mme_acc_plus'].mean()
        print(f"Objective Score (mme_acc_plus average across all rows): {objective_score:.6f}")
        return objective_score

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Evaluation script failed. Return Code: {e.returncode}")
        print(f"Stderr: {e.stderr}")
        return 0.0
    except Exception as e:
        print(f"Unexpected error during file handling or execution: {e}")
        return 0.0


def objective_function_wrapper(alpha_values: List[float], args: argparse.Namespace) -> float:
    """
    The objective function expected by Bayesian optimization, receives a list of parameters, and returns a score to be minimized.
    """
    # Convert list to tuple (alpha1, alpha2, alpha3)
    alpha_config = tuple(alpha_values)
    
    # Run evaluation, get F1 score
    f1_score = run_mme_evaluation(
        alpha_config,
        cfg_path=args.cfg,
        mme_root=args.mme_root,
        subtask=args.subtask
    )
    
    # Bayesian optimization minimizes by default, so we return negative F1 Score to achieve maximization of F1 Score
    return -f1_score


def main():
    parser = argparse.ArgumentParser(description="Bayesian Optimization for VisionZip Alpha Weights.")
    parser.add_argument("--cfg", type=str, required=True, help="Path to model YAML config.")
    parser.add_argument("--mme_root", type=str, required=True, help="Root directory of MME benchmark.")
    parser.add_argument("--subtask", type=str, default="", help="MME subtask to optimize against (e.g., OCR).")
    parser.add_argument("--n_calls", type=int, default=20, help="Number of total optimization calls.")
    parser.add_argument("--n_initial", type=int, default=5, help="Number of random exploration calls.")
    
    args = parser.parse_args()

    # Define the search space (assuming alpha weights are between [0.0, 2.0])
    space = [
        Real(0.0, 2.0, name='alpha1'), 
        Real(0.0, 2.0, name='alpha2'), 
        Real(0.0, 2.0, name='alpha3')
    ]
    
    print(f"Starting Bayesian Optimization for {args.subtask} F1 Score...")
    print(f"Total iterations: {args.n_calls}, Initial random calls: {args.n_initial}")

    # Execute Bayesian optimization
    result = gp_minimize(
        func=lambda x: objective_function_wrapper(x, args), # Pass the lambda wrapped function
        dimensions=space,
        n_calls=args.n_calls,
        random_state=42,
        n_initial_points=args.n_initial,
        acq_func="EI", # Expected Improvement (common acquisition function)
        verbose=True
    )

    # Output final results
    print("\n" + "="*50)
    print("Optimization Complete.")
    print(f"Best F1 Score Found: {-result.fun:.6f}")
    print(f"Optimal Alpha Weights (alpha1, alpha2, alpha3): {result.x}")
    print("="*50)


if __name__ == "__main__":
    main()