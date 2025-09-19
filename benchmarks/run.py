import datetime
import os
import json
import csv
from copy import deepcopy
import numpy as np
from multiprocessing import Pool
from .dataset_handler import DatasetHandler
from .metrics import metrics
from .seeds import SEEDS
from iMCTS import Regressor, simplify_expression
from tqdm import tqdm

def run_benchmark(
    benchmark: str = "Nguyen",
    model_params: dict = None,
    start_case: int = 1,
    end_case: int = None,
    run_num: int = 100,
    output_dir: str = "results",
    n_processes: int = 10  # number of parallel processes
):
    """Run symbolic regression benchmark (parallel version)"""
    # load configuration file
    if benchmark == "BlackBox":
        benchmark_type = "blackbox"
    else:
        benchmark_type = "basic"
    config_path = os.path.join("benchmarks", f"{benchmark_type}_config.json")
    with open(config_path) as f:
        config = json.load(f)
    
    if benchmark_type == "blackbox":
        test_cases = config[benchmark]
    else:
        test_cases = config["function_groups"][benchmark]
    if end_case is None:
        total_cases = len(test_cases)
    else:
        total_cases = min(end_case, len(test_cases))
    test_ratio = 0.25 if benchmark == "BlackBox" else 0
    model_params = deepcopy(model_params) or {}

    os.makedirs(output_dir, exist_ok=True)

    for case_idx in range(start_case, total_cases + 1):
        print(f"Running {benchmark} case {case_idx}/{total_cases}...")
        case_info = test_cases[case_idx-1]
        case_dir = os.path.join(output_dir, benchmark)
        csv_path = os.path.join(case_dir, f"{case_idx}.csv")
        os.makedirs(case_dir, exist_ok=True)

        # initialize result file if not exists
        if not os.path.exists(csv_path):
            _init_result_file(csv_path, benchmark, case_idx, case_info, model_params, config)

        # BlackBox data preloading
        if benchmark == "BlackBox":
            data_handler = DatasetHandler(config)
            data = data_handler.generate_group(benchmark, "datasets", case_idx)[0]
            X_total, y_total = data["x"], data["y"]
            print(f"Loaded {X_total.shape[1]} samples with {X_total.shape[0]} features.")
        else:
            X_total = y_total = None

        # prepare arguments for parallel runs
        args_list = []
        for run_id in range(run_num):
            seed = SEEDS[run_id % len(SEEDS)]
            args = {
                "benchmark": benchmark,
                "case_idx": case_idx,
                "model_params": model_params,
                "test_ratio": test_ratio,
                "config": config,
                "X_total": X_total,
                "y_total": y_total,
                "seed": seed,
                "run_id": run_id
            }
            args_list.append(args)

        # execute runs in parallel
        print(f"Running {len(args_list)} runs in parallel, n_processes={n_processes}...")
        with Pool(processes=n_processes) as pool:
            # use tqdm to display progress bar, keep result order
            results = list(tqdm(
                pool.imap(_process_single_run, args_list),
                total=len(args_list),
                desc=f"Case {case_idx} Progress",
                unit="run"
            ))

        # write results
        print('total results:', len(results))
        for result in results:
            if result is not None:  # handle possible failures
                _write_result(csv_path, **result)

def _process_single_run(args):
    """Process a single run task"""
    try:
        np.random.seed(args["seed"])  # set random seed

        # prepare data
        if args["benchmark"] == "BlackBox":
            data_handler = DatasetHandler(args["config"])
            X_train, X_test, y_train, y_test = data_handler.split_data(
                args["X_total"].T, args["y_total"], args["test_ratio"], args["seed"]
            )
            X_train, X_test = X_train.T, X_test.T
        else:
            data_handler = DatasetHandler(args["config"])
            data = data_handler.generate_group(
                group_name=args["benchmark"], 
                case_index=args["case_idx"], 
                seed=args["seed"]
            )[0]
            X_train = X_test = data["x"]
            y_train = y_test = data["y"]

        # train model
        model = Regressor(X_train, y_train, **args["model_params"])
        start_time = datetime.datetime.now()
        exp_str, vec_exp_str, evals, path = model.fit(seed=args["seed"])
        elapsed = (datetime.datetime.now() - start_time).total_seconds()

        # compute metrics
        metric, sympy_expr = metrics(model, exp_str, vec_exp_str, X_test, y_test, X_train, y_train)
        exp_str = simplify_expression(exp_str)

        return {
            "expr": exp_str,
            "sympy_expr": sympy_expr,
            "evals": evals,
            "best_path": path,
            "time": elapsed,
            "metric": metric,
            "seed": args["seed"]
        }
    
    except Exception as e:
        print(f"Run {args['run_id']} failed: {str(e)}")
        return None

def _init_result_file(path, benchmark, case_idx, case_info, params, config):
    """Initialize result file and write metadata (adapted to config parameters)"""
    if benchmark == "BlackBox":
        function_name = case_info
    else:
        function_name = case_info["name"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([
            ["Experiment Metadata"],
            ["Benchmark", benchmark],
            ["Case Index", case_idx],
            ["Function Name", function_name],
            ["Run Date", datetime.datetime.now().isoformat()],
            [],
            ["Model Parameters"],
            *[[k, v] for k, v in _get_model_params(params).items()],
            [],
            ["Data Columns"],
            ["Original Expression", "Sympy Expression", 
             "Evaluations", "Path", "Time (s)", "Test R2", "Train R2", 
                "Simplicity", "Seed"],
        ])


def _get_model_params(params):
    """Get model parameters information"""
    return {k: str(v) for k, v in params.items()}

def _write_result(path, expr, sympy_expr, evals, best_path, time, metric, seed):
    """Write single run result (parameterized adjustment)"""
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            expr,
            sympy_expr,
            evals,
            best_path,
            f"{time:.8f}",
            metric["test_accuracy"],
            metric["train_accuracy"],
            metric["simplicity"],
            seed
        ])