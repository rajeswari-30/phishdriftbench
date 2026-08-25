"""Process-isolated baseline execution (see docs/threading-notes.md).

Two distinct cross-library conflicts were found empirically on this stack,
neither fixed by capping thread counts:
  - CatBoost fit -> XGBoost fit -> PyTorch train/infer in one process: hangs.
  - PyTorch (BERT) infer -> LightGBM fit in one process: segfaults.
Every baseline therefore runs in its own short-lived subprocess, and B5
specifically is split into two subprocesses internally (BERT embedding,
then LightGBM fit/predict) so torch and lightgbm never share a process.

IMPORTANT: any top-level script that calls into this module must guard its
entry point with `if __name__ == "__main__":`, since multiprocessing's
`spawn` start method re-imports the launching module in each child process.
"""
from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from phishdriftbench.models import baselines as _baselines


# --------------------------------------------------------------------------
# Single train / single test (kept for simple call sites, e.g. smoke tests)
# --------------------------------------------------------------------------

def _bert_embed_worker(urls, model_name, allow_weight_download):
    return _baselines._bert_embed(urls, model_name, allow_weight_download)


def _compute_bert_embeddings_isolated(urls, model_name, allow_weight_download):
    """Runs BERT embedding in its own subprocess (torch only — no lightgbm
    ever imported here), returning a plain numpy array."""
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
        fut = ex.submit(_bert_embed_worker, urls, model_name, allow_weight_download)
        return fut.result()


def _fit_predict_b5_worker(X_train, y_train, X_test, bert_feats_train, bert_feats_test,
                            model_name, fit_kwargs):
    fit_kwargs = fit_kwargs or {}
    model = _baselines.fit_b5(X_train, y_train, model_name=model_name,
                               bert_feats=bert_feats_train, **fit_kwargs)
    return _baselines.predict_b5(model, X_test, bert_feats=bert_feats_test)


def _fit_predict_worker(name, X_train, y_train, X_test, fit_kwargs=None):
    fit_kwargs = fit_kwargs or {}
    if name == "B5":
        raise RuntimeError("B5 must go through run_baseline_isolated, which handles its two-stage isolation.")
    fit_fn, predict_fn = _baselines.BASELINES[name]
    model = fit_fn(X_train, y_train, **fit_kwargs)
    return predict_fn(model, X_test)


def run_baseline_isolated(name: str, X_train, y_train, X_test, urls_train=None, urls_test=None,
                           allow_weight_download: bool = False, fit_kwargs: dict | None = None,
                           model_name: str = "distilbert-base-uncased"):
    """Fit+predict one baseline (B1..B5) in a fresh subprocess against ONE
    test set; returns the score array. For multiple test sets sharing one
    fit (Axis T's windows, Axis S's sources), use
    `run_baseline_isolated_multi` instead — it fits once, not once per test
    set."""
    if name == "B5":
        if urls_train is None or urls_test is None:
            raise ValueError("B5 requires urls_train and urls_test")
        bert_feats_train = _compute_bert_embeddings_isolated(urls_train, model_name, allow_weight_download)
        bert_feats_test = _compute_bert_embeddings_isolated(urls_test, model_name, allow_weight_download)
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
            fut = ex.submit(_fit_predict_b5_worker, X_train, y_train, X_test,
                             bert_feats_train, bert_feats_test, model_name, fit_kwargs)
            return fut.result()

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
        fut = ex.submit(_fit_predict_worker, name, X_train, y_train, X_test, fit_kwargs)
        return fut.result()


def run_all_baselines(names: list[str], X_train, y_train, X_test, urls_train=None, urls_test=None,
                       allow_weight_download: bool = False) -> dict:
    """Run each named baseline (in its own subprocess) against the same
    train/test split and return {name: scores}."""
    return {
        name: run_baseline_isolated(name, X_train, y_train, X_test, urls_train, urls_test,
                                     allow_weight_download)
        for name in names
    }


# --------------------------------------------------------------------------
# Fit-once / predict-many (Axis T's windows, Axis S's sources): avoids
# refitting the same model once per test set.
# --------------------------------------------------------------------------

def _fit_predict_multi_worker(name, X_train, y_train, X_tests: dict, fit_kwargs=None) -> dict:
    fit_kwargs = fit_kwargs or {}
    fit_fn, predict_fn = _baselines.BASELINES[name]
    model = fit_fn(X_train, y_train, **fit_kwargs)
    return {test_name: predict_fn(model, X_test) for test_name, X_test in X_tests.items()}


def _fit_predict_b5_multi_worker(X_train, y_train, X_tests: dict, bert_feats_train,
                                  bert_feats_tests: dict, model_name, fit_kwargs) -> dict:
    fit_kwargs = fit_kwargs or {}
    model = _baselines.fit_b5(X_train, y_train, model_name=model_name,
                               bert_feats=bert_feats_train, **fit_kwargs)
    return {
        test_name: _baselines.predict_b5(model, X_tests[test_name], bert_feats=bert_feats_tests[test_name])
        for test_name in X_tests
    }


def run_baseline_isolated_multi(name: str, X_train, y_train, X_tests: dict, urls_train=None,
                                 urls_tests: dict | None = None, allow_weight_download: bool = False,
                                 fit_kwargs: dict | None = None,
                                 model_name: str = "distilbert-base-uncased") -> dict:
    """Fit ONE model for `name` in a fresh subprocess, then score every
    named test set in `X_tests` (dict[str, DataFrame]) with that same
    model. Returns {test_name: scores}.

    For B5, embeddings for train + all test sets are computed together in a
    single BERT subprocess call (one concatenated forward pass, split back
    out by index) rather than one call per test set."""
    if name == "B5":
        if urls_train is None or urls_tests is None:
            raise ValueError("B5 requires urls_train and urls_tests")
        test_names = list(X_tests.keys())

        # Dedup by URL value before embedding: test sets that share a fixed
        # anchor population across windows/sources (e.g. Axis T's held-out
        # legitimate sample, reused identically in every window) would
        # otherwise be re-embedded once per window for no benefit.
        all_urls_flat = list(urls_train) + [u for tn in test_names for u in urls_tests[tn]]
        unique_urls = list(dict.fromkeys(all_urls_flat))
        idx_of = {u: i for i, u in enumerate(unique_urls)}

        unique_feats = _compute_bert_embeddings_isolated(unique_urls, model_name, allow_weight_download)
        bert_feats_train = unique_feats[[idx_of[u] for u in urls_train]]
        bert_feats_tests = {
            tn: unique_feats[[idx_of[u] for u in urls_tests[tn]]] for tn in test_names
        }

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
            fut = ex.submit(_fit_predict_b5_multi_worker, X_train, y_train, X_tests,
                             bert_feats_train, bert_feats_tests, model_name, fit_kwargs)
            return fut.result()

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
        fut = ex.submit(_fit_predict_multi_worker, name, X_train, y_train, X_tests, fit_kwargs)
        return fut.result()


def run_all_baselines_multi(names: list[str], X_train, y_train, X_tests: dict, urls_train=None,
                             urls_tests: dict | None = None, allow_weight_download: bool = False) -> dict:
    """{baseline_name: {test_name: scores}} for every baseline in `names`,
    each fit once against `X_train`/`y_train`."""
    return {
        name: run_baseline_isolated_multi(name, X_train, y_train, X_tests, urls_train, urls_tests,
                                           allow_weight_download)
        for name in names
    }
