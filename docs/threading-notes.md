# Threading note: cross-library OpenMP deadlock (macOS, Apple Silicon)

CatBoost, XGBoost, LightGBM and PyTorch each bundle their own OpenMP-style
threading runtime. Root-caused by bisection:

- CatBoost fit, then PyTorch train/predict, in the same process: **fine**.
- CatBoost fit, then XGBoost fit, then PyTorch train/predict, in the same
  process: **deadlocks** (near-zero CPU use — a genuine deadlock, not slow
  computation), reproduced even on tiny synthetic data (500x5) with 3
  epochs. Capping thread counts (`thread_count=4`, `n_jobs=4`,
  `torch.set_num_threads(4)`) does NOT prevent it — the deadlock is a
  cross-runtime interaction (three separate bundled OpenMP-style runtimes
  live in one address space once both GBM libraries have initialized
  theirs), not a resource-contention issue that thread caps would fix.

**Fix applied:** every baseline still caps its thread count (harmless, and
helps on machines with fewer cores), but the real fix is process isolation.
`eval/isolated_run.py` runs each baseline's fit+predict in its own
short-lived subprocess (`concurrent.futures.ProcessPoolExecutor` with the
`spawn` start method, one process per call). This is what `PhishDriftBench`'s
split engine and the smoke-test script use to run B1-B5 together — never
call more than one of {CatBoost, XGBoost, LightGBM, PyTorch-using-baseline}
back to back inside one long-lived interpreter.
