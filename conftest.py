# Root conftest: makes pytest put the repository root on sys.path so tests
# can import the root-level modules (verify_labels, train_smoke, ...)
# regardless of how pytest is invoked (`pytest` vs `python -m pytest`).
