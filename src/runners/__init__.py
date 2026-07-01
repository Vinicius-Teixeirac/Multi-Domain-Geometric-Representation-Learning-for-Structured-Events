"""Runners package: one module per pipeline stage (cleaning, entity
construction, splitting, tabular/text/graph representation, and model
training) that wires config, idempotency checks, and artifact I/O together
for `main.py` to orchestrate."""
