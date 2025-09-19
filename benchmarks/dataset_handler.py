import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.model_selection import train_test_split

class DatasetHandler:
    def __init__(self, config):
        self.config = config
    
    def split_data(
        self, 
        X: np.ndarray,
        y: np.ndarray,
        test_ratio: float,
        random_seed: int,
    ) -> tuple:
        """Split dataset using sklearn (shape: samples, features) -> keep original dimensions"""
        return train_test_split(
            X, y,
            test_size=test_ratio,
            random_state=random_seed,
            shuffle=True
        )
    
    def read_file(
        self,
        filename: str,
        label: str = "target",
        sep: str = None,
        sheet_name: int = 0
    ) -> tuple:
        """Read a data file and return feature matrix, target array, and feature names"""
        try:
            # load data from Excel or CSV (with optional gzip compression)
            if filename.endswith((".xls", ".xlsx")):
                df = pd.read_excel(filename, sheet_name=sheet_name)
            else:
                compression = "gzip" if filename.endswith(".gz") else None
                df = pd.read_csv(filename, sep=sep, compression=compression, engine="python")
            
            # clean column names: strip whitespace and replace dots with underscores
            df.columns = df.columns.str.strip().str.replace(".", "_")
            
            # separate features and target
            feature_names = df.columns.drop(label).to_numpy()
            y = df[label].to_numpy()
            X = df.drop(columns=label).to_numpy()
            
            # return in shape (features, samples)
            return X.T, y, feature_names
        except Exception as e:
            # file read failed
            print(f"File read failed: {e}")
            return None, None, None
    
    def _generate_samples(self, case: Dict, seed : int) -> np.ndarray:
        """Generate synthetic samples based on configuration (shape: features, samples)"""
        method = case.get("sampling", "U")
        low, high = case["data_range"]
        var_num, samples = case["variables"], case["samples"]
        rng = np.random.default_rng(seed)
        
        if method == "U":
            # uniform sampling
            return rng.uniform(low, high, (var_num, samples))
        if method == "E":
            # evenly spaced grid sampling
            if var_num == 1:
                return np.linspace(low, high, samples).reshape(1, -1)
            per_dim = int(round(samples ** (1/var_num)))
            axes = [np.linspace(low, high, per_dim) for _ in range(var_num)]
            # meshgrid then reshape to (features, samples)
            return np.stack(np.meshgrid(*axes), axis=0).reshape(var_num, -1)
        # unsupported sampling method
        raise ValueError(f"Unsupported sampling method: {method}")
    
    def generate_group(
        self,
        group_name: str,
        dir: Optional[str] = None,
        case_index: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[Dict]:
        """Generate datasets, supporting both file-based and synthetic data"""
        # get list of cases from config
        case_source = self.config.get(group_name, []) if dir else \
            self.config.get("function_groups", {}).get(group_name, [])
        
        if not case_source:
            raise ValueError(f"Group '{group_name}' not found")
        
        # select specific case if index provided
        if case_index is not None:
            case_source = [case_source[case_index - 1]] if 1 <= case_index < len(case_source)+1 else []
        
        results = []
        for case in case_source:
            case_name = case if dir else case.get("name", "unknown case")
            try:
                # data generation logic
                if dir:
                    filename = f"{dir}/{case}/{case}.tsv.gz"
                    X_total, y_total, feature_names = self.read_file(filename)
                    metadata = {"source": filename}
                    variables = X_total.shape[0]
                    extra = {"feature_names": feature_names}
                else:
                    X_total = self._generate_samples(case, seed)
                    y_total = eval(case["expression"])(X_total)
                    metadata = case
                    variables = case["variables"]
                    extra = {"sampling": case.get("sampling", "U")}
                
                # collect result, with X in (features, samples) format
                results.append({
                    "group": group_name,
                    "name": case_name,
                    "variables": variables,
                    "x": X_total,
                    "y": y_total,
                    "metadata": metadata,
                    **extra
                })
                
            except Exception as e:
                # processing failed for this case
                print(f"Processing failed [{case_name}]: {e}")
        
        return results