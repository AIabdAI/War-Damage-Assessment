# DVC vs MLflow — why this project uses both

There is a real overlap between the two tools, but they answer different questions.

## DVC doesn't really "track experiments" — it versions files, and metrics happen to be files

In `dvc.yaml` the train stage declares `reports/metrics.json` as a *metrics output*.
That file gets committed to git alongside `params.yaml` and `dvc.lock`. So in DVC's
world, **one experiment = one git commit**: the parameters that went in, the exact
data version (via the lock file's hashes), and the numbers that came out.

Commands like `dvc metrics diff master` don't consult any database — they just read
`metrics.json` at two points in git history and diff them.

## MLflow is a purpose-built run database

Every training run becomes a row in `mlflow.db` with its parameters, per-epoch
metric curves, and artifacts — independent of git. You get a UI (`mlflow ui`),
you can compare dozens of runs side by side, including runs you never committed
(failed ideas, quick tests).

## The practical difference

|                       | DVC metrics                                   | MLflow                          |
|-----------------------|-----------------------------------------------|---------------------------------|
| Unit of tracking      | git commit                                    | run in a database               |
| Best question         | "how does this PR change the numbers vs master?" | "which of my last 20 runs was best, and how did it converge?" |
| Where you see it      | `dvc metrics diff`, CML's PR comment          | `mlflow ui` at localhost:5000   |
| Tied to reproducibility | yes — lock file pins exact data + code      | no — a run is just a record     |

## Why both earn their place here

- **DVC metrics feed the CI story**: CML posts the git-diff of metrics on every PR —
  an auditable, reviewable record tied to exact data versions.
- **MLflow feeds the exploration story**: during a day of tuning you generate many
  runs, most never committed, and browse them in the UI. The ultralytics trainer
  logs to MLflow natively (steered by `train_smoke.py` into the
  `war-damage-smoke` experiment, beside the registered baselines).

Note: DVC has its own fuller experiment feature (`dvc exp run` / `dvc exp show`)
that runs variations without committing. This project doesn't use it — MLflow
covers that role better with its UI and the native ultralytics integration.
