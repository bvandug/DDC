import argparse
from tune_bc import tune_hyperparameters  # Import your existing function

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--study_name", type=str, default="buck_converter_study")
    parser.add_argument("--storage", type=str, default="sqlite:///buck_converter_study.db")
    args = parser.parse_args()

    tune_hyperparameters(
        algo_name=args.algo,
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage=args.storage
    )
