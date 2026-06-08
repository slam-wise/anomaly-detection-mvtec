"""End-to-end fit / evaluate / benchmark pipelines."""

from anomaly_det.pipelines.benchmark import run_benchmark
from anomaly_det.pipelines.evaluate import eval_category
from anomaly_det.pipelines.fit import fit_category

__all__ = ["fit_category", "eval_category", "run_benchmark"]