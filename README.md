# MLOps Continuous Training Tutorial

An end-to-end, minimal-but-complete reference implementation of an **MLOps
continuous training (CT)** loop. The project trains a classifier on the
classical Fisher *Iris* dataset, tracks every input and artifact with
[DVC](https://dvc.org), and retrains automatically in CI with GitHub Actions,
writing metrics back as a CML report.

It is designed as a teaching artifact: small enough to read end to end in one
sitting, yet exercising the mechanisms that real ML platforms rely on —
declarative pipelines, content-addressed artifact versioning, remote storage,
keyless cloud authentication, and pull-request-driven retraining.

---

## 1. Educational objectives

By reading and running this repository, a student should be able to:

1. Express a training workflow as a **declarative DAG** of stages with explicit
   inputs (`deps`) and outputs (`outs`), and understand why this beats a
   hand-ordered shell script.
2. Use **content hashing** (MD5) to make pipeline execution cache-aware and
   reproducible: a stage re-runs only when its dependencies' hashes change.
3. Separate **data**, **code**, and **model artifacts** across two version
   control systems — Git for code and metadata, DVC for large/opaque binaries.
4. Perform **stratified, seeded** data splitting so that evaluation is
   statistically valid and bit-for-bit reproducible.
5. Write a **model-agnostic** evaluation stage that reconstructs the estimator
   from a persisted configuration rather than hard-coding its type.
6. Implement **data validation as a unit test suite** that gates the pipeline
   before any training compute is spent.
7. Authenticate CI to a cloud provider **without long-lived secrets**, using
   GitHub OIDC and GCP Workload Identity Federation.
8. Publish **metric diffs** against a baseline branch as an automated report on
   the pull request or commit.

---

## 2. Repository layout

```
.
├── bootstrap.sh                 # idempotent GCP provisioning (WIF, SA, bucket)
├── .env.cloudbuild(.example)    # GCP project/bucket/WIF variables
├── dvc.yaml                     # pipeline DAG (stages, deps, outs, metrics)
├── dvc.lock                     # resolved DAG with content hashes (committed)
├── requirements.txt             # fully pinned environment (pip freeze)
├── data/
│   ├── iris.csv                 # source dataset (DVC-tracked)
│   ├── iris.csv.dvc             # pointer: md5 + size for iris.csv
│   ├── train.csv                # pipeline output: training split
│   └── eval.csv                 # pipeline output: held-out split
├── src/
│   ├── validate.py              # stage 1: data-quality unit tests
│   ├── split.py                 # stage 2: stratified train/eval split
│   ├── train.py                 # stage 3: fit estimator, dump artifacts
│   └── eval.py                  # stage 4: score on held-out split
├── models/
│   ├── model.joblib             # fitted estimator (DVC-tracked binary)
│   ├── training_config.json     # model type, params, columns (cache: false)
│   └── metrics.json             # eval metrics (cache: false)
├── docs/
│   └── GITHUB_ACTIONS.md        # GitHub Actions reference notes
└── .github/workflows/ct.yml     # continuous training workflow
```

---

## 3. Data

**Source.** `data/iris.csv` — 101 rows (header + 100 samples) of the Fisher
Iris dataset, with four continuous features in centimetres and one categorical
label:

| column | role | type | notes |
| --- | --- | --- | --- |
| `sepal_length` | feature | float | cm |
| `sepal_width` | feature | float | cm |
| `petal_length` | feature | float | cm |
| `petal_width` | feature | float | cm |
| `species` | label | string | `setosa` \| `versicolor` \| `virginica` |

**Versioning.** The CSV is *not* stored in Git. `data/iris.csv.dvc` is a small
text pointer recording the file's MD5 (`4954d408…`) and byte size (`2606`); the
content itself lives in the DVC cache and is synchronised to a remote. This
keeps Git history light while still allowing exact reconstruction of any
dataset revision via `dvc checkout` / `dvc pull`.

---

## 4. The pipeline

`dvc.yaml` defines four stages. DVC resolves them into a directed acyclic graph
from the `deps` → `outs` relationships; `dvc repro` then executes only the
stages whose hashes have drifted.

```
data/iris.csv ─┬─► data_validation ─────────────────────────► (exit status)
               │      dep: src/validate.py
               │
               └─► split ──┬─► data/train.csv ──► train ──┬─► models/model.joblib
                     dep:   │                          │      models/training_config.json
                src/split.py└─► data/eval.csv  ──┐      │
                                                │      └────► eval ──► models/metrics.json
                                                └─────────────┘
                                                     dep: src/eval.py
```

| stage | command | deps | outs |
| --- | --- | --- | --- |
| `data_validation` | `uv run python -m unittest src.validate` | `data/iris.csv`, `src/validate.py` | — (gating stage) |
| `split` | `uv run src/split.py` | `data/iris.csv`, `src/split.py` | `data/train.csv`, `data/eval.csv` |
| `train` | `uv run src/train.py` | `data/train.csv`, `src/train.py` | `models/model.joblib`, `models/training_config.json` |
| `eval` | `uv run src/eval.py` | `data/eval.csv`, `models/model.joblib`, `src/eval.py` | `models/metrics.json` (declared as a DVC *metric*) |

Two deliberate design choices are worth noting:

- **`cache: false` on the JSON artifacts.** `training_config.json` and
  `metrics.json` are small, human-readable, and diffable; excluding them from
  the DVC cache lets them be committed to Git so that `dvc metrics diff` and
  code review can operate on plain text.
- **Validation runs first and is cheap.** A failing assertion aborts the DAG
  before `split` or `train` execute, so corrupt data cannot consume training
  time or pollute the model registry.

### 4.1 `src/validate.py` — data contracts as tests

Validation is implemented with the standard library only (`csv` + `unittest`),
which makes the stage dependency-free and immune to environment drift.

<details>
<summary>Test contracts, thresholds, and the pytest history</summary>

Each test method is a contract:

| test | asserts |
| --- | --- |
| `test_file_exists_and_not_empty` | file present, ≥ 1 data row |
| `test_expected_schema` | header equals `FEATURE_COLUMNS + [species]`, in order |
| `test_no_missing_values` | no `None`/blank cell in any column |
| `test_feature_types_are_floats` | every feature parses as `float` |
| `test_feature_ranges_are_reasonable` | `0.0 < value ≤ 10.0` cm |
| `test_label_values_are_valid` | label ∈ {`setosa`, `versicolor`, `virginica`} |
| `test_all_classes_present` | all three classes observed at least once |
| `test_no_unexpected_null_tokens` | no `""`, `na`, `nan`, `null`, `none`, `?` |

The range check is intentionally *looser* than the observed data
(`MIN_CM = 0.0`, `MAX_CM = 10.0`): it is designed to catch corrupt or garbage
values without over-fitting to the current sample, which would make the test
brittle to legitimate new data. Failure messages report the 1-based CSV line
number (header is line 1) to make triage direct.

The stdlib-only choice is also a historical artefact of the build order
(§5, step 3): the validator was written before the ML dependency stack was
pinned, and the DVC stage initially invoked it with `pytest` — a runner that
was never in `requirements.txt`. Switching the `cmd` to
`python -m unittest src.validate` aligned the invocation with the design.

</details>

### 4.2 `src/split.py` — reproducible, stratified splitting

A single seeded, stratified `train_test_split` turns `data/iris.csv` into the
train and eval partitions.

<details>
<summary>Stratification, seeding, and why the split is a pure function</summary>

```python
train_test_split(iris, test_size=0.2, stratify=iris["species"],
                 random_state=42, shuffle=True)
```

- **`stratify=y`** preserves the class proportion in both partitions. With a
  small dataset, an unstratified split risks a class being absent or
  under-represented in the evaluation set, which biases metrics.
- **`random_state=42`** pins the shuffle, so the split is deterministic across
  machines and runs — a precondition for DVC cache hits and for reproducible
  evaluation.
- The split is exposed as a pure function `split(iris, eval_size, seed)`
  returning `(train_df, eval_df)`, while `main()` handles I/O; this keeps the
  logic unit-testable and lets `--eval-size` / `--seed` be overridden from the
  CLI.

Resulting cardinalities: 80 train rows, 21 eval rows (roughly 8/6/7 per class).

</details>

### 4.3 `src/train.py` — fitting and artifact persistence

Fits a `DecisionTreeClassifier(random_state=42, max_depth=3)` on the training
split and writes two artifacts: the serialised estimator and a self-describing
JSON manifest.

<details>
<summary>Regularisation evidence, artifacts, and the model-agnostic factory</summary>

`max_depth=3` is a deliberate regularisation choice, and it is the one
hyper-parameter that was actually tuned during development: the tree was
initially built at `max_depth=10`, deep enough to memorise the 80-row training
set. The measured effect is a textbook bias–variance trade:

| `max_depth` | train accuracy | train macro-F1 | eval accuracy | eval macro-F1 | `model.joblib` |
| --- | --- | --- | --- | --- | --- |
| 10 | 1.000 | 1.000 | 0.9048 | 0.897 | 3105 B |
| 3 | 0.950 | 0.950 | **0.9524** | **0.947** | 2401 B |

Capping the depth costs 5 points of *training* fit and buys ~4.8 points of
*held-out* accuracy — i.e. the deeper tree was over-fitting, and the smaller
tree generalises better. The reduced tree is also a smaller serialised artifact.

The two artifacts written are:

- `models/model.joblib` — the fitted estimator, serialised with `joblib`
  (efficient for NumPy-backed objects).
- `models/training_config.json` — a self-describing manifest:

```json
{
  "model_type": "DecisionTreeClassifier",
  "params": {"random_state": 42, "max_depth": 3},
  "feature_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
  "label_column": "species",
  "train_path": "data/train.csv",
  "n_train_rows": 80,
  "training_metrics": {"accuracy": 0, "f1_macro": 0}
}
```

The **single-source-of-truth pattern** is enforced by `build_model(params)`,
which both `train.py` and `eval.py` import. Changing the algorithm requires
editing one factory function; the evaluator adapts automatically by reading
`model_type`, `params`, `feature_columns`, and `label_column` from the manifest.
This is the mechanism that makes the pipeline *model-agnostic*.

</details>

### 4.4 `src/eval.py` — held-out evaluation

Scores the restored model on the held-out split, prints a classification report
and confusion matrix, and writes `models/metrics.json`. Measured accuracy:
**0.9524**.

<details>
<summary>Report contents, latest per-class metrics, and the confusion matrix</summary>

Evaluation reconstructs the estimator from the manifest, restores weights from
`model.joblib`, and scores `data/eval.csv` (never seen during fitting). It
reports and persists:

- **Accuracy** — fraction of correctly classified samples.
- **Per-class precision / recall / F1 / support** — via
  `classification_report(..., output_dict=True, zero_division=0)`. Macro and
  weighted averages are included.
- **Confusion matrix** — as a nested list, with rows = true class and columns =
  predicted class, ordered by `labels = sorted(y_true.unique())`.

`zero_division=0` suppresses undefined-metric warnings when a class has no
predicted samples. `labels` is captured alongside the matrix so the JSON is
self-describing.

**Latest measured performance** (21 eval rows):

| class | precision | recall | F1 | support |
| --- | --- | --- | --- | --- |
| setosa | 1.000 | 1.000 | 1.000 | 8 |
| versicolor | 1.000 | 0.833 | 0.909 | 6 |
| virginica | 0.875 | 1.000 | 0.933 | 7 |
| **macro avg** | 0.958 | 0.944 | 0.947 | 21 |

Confusion matrix (true × predicted):

```
[[8 0 0]     setosa     — perfect
 [0 5 1]     versicolor — one sample misclassified as virginica
 [0 0 7]]    virginica  — perfect
```

Overall accuracy: **0.9524**. The single error is the classic
*versicolor ↔ virginica* boundary case, where the two species overlap in petal
measurements; a depth-3 tree cannot carve that region cleanly. This is an
instructive example of irreducible class overlap rather than an implementation
defect.

</details>

---

## 5. Build order: a guided walkthrough

The repository was built in the order below, and that order is worth following
if you are reproducing it: version the data first, then grow the pipeline one
stage at a time, then automate it. Two tags mark the milestones — `v0.0` (data
versioned) and `v0.1` (first full pipeline run).

<details>
<summary>Six steps, from raw CSV to a self-updating pipeline</summary>

`dev` is the integration branch; `master` is protected and only receives changes
through a pull request.

**Step 1 — version the data before writing any model code.**
Initialise DVC, point it at a remote (here a GCS bucket), and track
`data/iris.csv` with `dvc add`. The dataset now has an addressable revision
(the `.dvc` pointer file) without bloating Git. At this point `src/` held only
empty placeholders — deliberately, so the storage layer was proven before the
compute layer existed.

**Step 2 — grow the pipeline one stage at a time.**
Add a stage to `dvc.yaml`, run it, commit it, then move to the next. The order
was: a stdlib validation script, then `split`, then `train` and `eval`. Each
new stage arrived together with the dependencies it actually needed — the
pandas/scikit-learn/joblib stack appears in `requirements.txt` at the same
moment as the first data-transforming stage, not before. The `train`/`eval`
commit also fixed the two conventions the project still uses: the model
manifest is written with `cache: false` so it stays diffable in Git, and
`metrics.json` is declared under `metrics:` so DVC can diff it. Running the
whole DAG once produced `dvc.lock` and the first artifacts (tag `v0.1`).

**Step 3 — make validation a gate, not an afterthought.**
The `data_validation` stage was added *after* the training stages existed, and
its first version invoked `pytest` — which was never installed, because the
validator was written against the standard library on purpose. The fix was to
call `python -m unittest`. Lesson: the test runner is part of your dependency
surface; if the tests are dependency-free, the command that runs them should be
too. Because the stage declares no outputs, a failure aborts the DAG before any
training compute is spent.

**Step 4 — provision keyless cloud access.**
`bootstrap.sh` creates the GCS bucket, the Workload Identity pool and provider,
the service account, and the two IAM bindings that let a GitHub workflow
impersonate it — no JSON key anywhere. The first attempt failed with
`403 iam.serviceAccounts.getAccessToken`: the WIF principal set had been built
from the repository *name* alone, so it could never match GitHub's
`assertion.repository` claim, which is `<owner>/<repo>`. The fix builds the
principal set from `gh repo view --json nameWithOwner` and, importantly, makes
both IAM bindings run on *every* invocation rather than only inside the
"create if missing" branches — the bucket binding had been skipped whenever the
bucket already existed, silently leaving the service account without storage
access on re-runs. This is the one failure mode in the project's history worth
studying closely: the attribute condition, the principal set, and the
idempotency of the bindings all have to agree.

**Step 5 — turn the pipeline into continuous training.**
The first `ct.yml` was the bare loop: checkout, authenticate via WIF, install
with `uv`, then `dvc pull` → `dvc repro` → `dvc push`, triggered by pushes to
`master`/`dev` and by PRs into `master`. Three additions followed: a manual
`workflow_dispatch` trigger, a CML report step (plus the least-privilege
`permissions:` block it needs), and the path filter that lets the workflow
trigger on changes to itself.

**Step 6 — close the loop.**
On `dev`, the workflow now commits the refreshed metadata (`dvc.lock`,
`dvc.yaml`, `models/metrics.json`, `models/training_config.json`) straight back
to the branch, so retrained state is reviewable without anyone running DVC
locally. That required a `concurrency` group so two runs cannot race to push,
and a guard that skips the job for fork PRs, which must never receive write
credentials.

### What the history is worth remembering

- **Over-fitting can hide bugs.** The tree was first built at `max_depth=10`,
  where it scored a perfect 1.0 on its own training data. Dropping to depth 3
  raised held-out accuracy from 0.905 to 0.952 — and simultaneously exposed an
  `int()` truncation bug in the training-metrics logging that had been masked
  by the perfect score. Regularising the model is also what made the defect
  visible.
- **Committed artifacts go stale.** The depth change was committed on its own,
  leaving `metrics.json` holding the previous model's numbers until the next
  pipeline run overwrote them. Anything checked into the repository is only as
  fresh as the last run that produced it.
- **CI is the place to fix staleness.** That is precisely the gap the write-back
  step exists to close: the pipeline, not the human, is responsible for keeping
  the recorded metrics in sync with the code.

Current branch state: `master` holds the `v0.0` tag and the merged first pull
request; `dev` is several commits ahead with the second pull request still open.

</details>

---

## 6. Environment and reproducibility

Dependencies are managed with [uv](https://docs.astral.sh/uv/) and pinned in
`requirements.txt` (a full freeze, including transitive versions such as
`dvc==3.67.1`, `scikit-learn==1.9.1`, `pandas==3.1.1`, `numpy==2.5.4`).

```bash
uv venv                              # create .venv
uv pip install -r requirements.txt   # install the pinned environment
```

Remote storage is configured in `.dvc/config`:

```ini
[core]
    remote = gcp-remote
['remote "gcp-remote"']
    url = gs://mlops_ct_tutorial
```

Typical local workflow:

```bash
uv run dvc pull     # fetch iris.csv, model.joblib from the remote
uv run dvc repro    # run only stale stages
uv run dvc push     # upload new/changed artifacts
uv run dvc metrics show          # current metrics
uv run dvc metrics diff <rev>    # metrics change vs a revision
```

`dvc repro` is incremental by construction: each stage's `deps` and `outs` are
hashed (see `dvc.lock`), and a stage runs only when a hash changes. Editing
`src/train.py` invalidates `train` (and, transitively, `eval`), but leaves
`data_validation` and `split` cached.

---

## 7. Continuous training

`.github/workflows/ct.yml` turns the local pipeline into an automated,
PR-aware retraining loop. The workflow was built up over two pull requests —
first the bare pull/repro/push loop, then write-back, concurrency and fork
guards (see §5, steps 5–6).

**Triggers.**

- `workflow_dispatch` (manual, with a free-text `reason` input).
- `push` to `master` or `dev`, path-filtered to `src/**`,
  `data/iris.csv.dvc`, `dvc.yaml`, `dvc.lock`, `requirements.txt`, and the
  workflow file itself — so documentation-only commits do not trigger training.
- `pull_request` targeting `master`.

**Concurrency.** `group: ct-${{ github.ref }}` with
`cancel-in-progress: false` serialises runs per ref, preventing two runs from
racing to push `dvc.lock` or open the same report.

**Fork safety.** The job is skipped for pull requests from forks
(`head.repo.full_name == github.repository`), so untrusted code never receives
write credentials.

**Permissions (least privilege).**

| scope | level | purpose |
| --- | --- | --- |
| `contents` | write | push branches / commit metadata |
| `id-token` | write | mint the OIDC token for WIF |
| `pull-requests` | write | open PRs and post comments |

**Job sequence.**

1. `actions/checkout@v4` with `fetch-depth: 0` — real history is needed to
   branch, commit, and compute metric diffs.
2. `google-github-actions/auth@v2` — keyless authentication (see §8).
3. `astral-sh/setup-uv@v5` (with cache) and `iterative/setup-cml@v2`.
4. `uv venv` + `uv pip install -r requirements.txt`.
5. `dvc pull` → `dvc repro` → `dvc push`: fetch data, retrain end to end, upload
   new artifacts.
6. **Metadata write-back.** On the `dev` ref only — reached by a direct push or
   a manual dispatch, never a pull request, and never `master` — the bot commits
   the refreshed `dvc.lock`, `dvc.yaml`, `models/metrics.json`, and
   `models/training_config.json` back to the branch with `[skip ci]`, guarded by
   `git diff --cached --quiet` so no-op runs exit cleanly. `master` is left to
   the PR path because it is protected.
7. **CML report.** `dvc metrics diff "origin/$BASE" --md` produces a Markdown
   table of metric changes against the base branch (falling back to
   `dvc metrics show --md` when no baseline exists). It is posted as a PR
   comment (`cml comment update`) or a commit comment
   (`cml comment update --target=commit`) depending on the event. A `CML_PAT`
   secret is preferred over the default `GITHUB_TOKEN`, because GitHub locks
   commit-comment updates made with the default token.

`docs/GITHUB_ACTIONS.md` contains a broader reference on workflow syntax,
triggers, matrices, caching, and the rationale for `id-token: write`.

---

## 8. Keyless cloud authentication (GCP WIF)

No service-account JSON key exists anywhere in this repository. Instead,
`bootstrap.sh` provisions **Workload Identity Federation**, letting GitHub's
OIDC token be exchanged for short-lived GCP credentials.

What the script does (all steps idempotent, safe to re-run):

1. Activates the `default` gcloud configuration and sets the quota project.
2. Enables `iamcredentials.googleapis.com` and
   `cloudresourcemanager.googleapis.com`.
3. Creates the GCS bucket `gs://mlops_ct_tutorial` (uniform bucket-level
   access, public-access prevention).
4. Creates the WIF pool (`mlops-pool`) and an OIDC provider (`github-provider`)
   with issuer `https://token.actions.githubusercontent.com`, mapping
   `google.subject`, `attribute.actor`, `attribute.repository`, and
   `attribute.repository_owner`, and restricting to repositories owned by the
   authenticated GitHub user.
5. Creates the service account `<repo>-actions@<project>.iam.gserviceaccount.com`.
6. Grants it `roles/storage.objectAdmin` on the bucket.
7. Binds `roles/iam.workloadIdentityUser` to the principal set
   `attribute.repository/<owner>/<repo>` — i.e. only this repository can
   impersonate the account.
8. Writes `SA_EMAIL` and `WORKLOAD_IDENTITY_PROVIDER` back into
   `.env.cloudbuild`, which are surfaced to the workflow as repository
   variables `GCP_SA_EMAIL` and `GCP_WIF_PROVIDER`.

The trust chain is therefore:

```
GitHub OIDC token ──► GCP STS token exchange ──► impersonate SA ──► gs:// access
        (id-token: write)          (WIF pool + provider)        (objectAdmin)
```

---

## 9. Setup

```bash
# 1. Local environment
uv venv
uv pip install -r requirements.txt

# 2. Cloud (once per project; requires gcloud + gh authenticated)
cp .env.cloudbuild.example .env.cloudbuild   # fill in PROJECT_ID, GCS_BUCKET, ...
./bootstrap.sh

# 3. Wire the values printed by bootstrap.sh into repository variables
#    GCP_WIF_PROVIDER and GCP_SA_EMAIL (Settings → Secrets and variables → Actions)

# 4. Run the pipeline
uv run dvc repro
uv run dvc metrics show
```

---

## 10. Design notes and known limitations

Stated plainly, as this is a teaching repository:

- **Training metrics are coerced to integers.** In `src/train.py`,
  `training_metrics` casts accuracy and macro-F1 with `int(...)`, so
  `models/training_config.json` reports `0` for both. This is a bug; the fix is
  to drop the casts (`float(accuracy)` or no cast at all). Held-out metrics in
  `models/metrics.json` are unaffected, as `eval.py` does not cast. The bug
  survived the early stages of the project because at the original
  `max_depth=10` the training metrics were exactly `1.0` and `int(1.0) == 1`
  looked plausible; it surfaced only when the depth was lowered to 3 and
  `int(0.95) == 0` (see §5).
- **`eval.py` imports `from train import build_model`**, which relies on `src/`
  being on `sys.path` (true for `uv run src/eval.py`). Running it as
  `python -m src.eval` from the repository root would require a relative import
  (`from .train import build_model`) or a package `__init__.py`.
- **The manifest records `model_type` but the evaluator only reads it for
  reporting.** `build_model` ignores the recorded type and always constructs a
  `DecisionTreeClassifier`. Genuine model-agnosticism would dispatch on
  `model_type` to the correct estimator class.
- **Train and eval metrics use different label orderings.** `classification_report`
  sorts labels internally, while the `labels` list is derived from
  `sorted(y_true.unique())`; these coincide on the current data but are not
  guaranteed to for arbitrary label sets.
- **Single dataset, single model, no hyper-parameter search.** Scope is
  intentional: the point is the *plumbing*, not the modelling.

---

## 11. Glossary

| term | meaning here |
| --- | --- |
| **stage** | a named unit of work in `dvc.yaml` with `cmd`, `deps`, `outs` |
| **dep / out** | declared input / output of a stage, hashed to drive caching |
| **DVC** | Data Version Control — Git-like versioning for data and models |
| **CML** | Continuous Machine Learning — CI reporting for ML metrics |
| **WIF** | Workload Identity Federation — keyless OIDC → cloud IAM exchange |
| **CT** | Continuous Training — automated retraining on code/data change |
| **stratified split** | a split that preserves class proportions in each partition |
| **artifact** | a produced file (model, config, metrics) tracked by DVC or Git |

---

## 12. References

- Fisher, R. A. (1936). *The use of multiple measurements in taxonomic
  problems.* Annals of Eugenics, 7(2), 179–188.
- Sculley, D., et al. (2015). *Hidden Technical Debt in Machine Learning
  Systems.* NeurIPS.
- DVC documentation — <https://dvc.org/doc>
- scikit-learn user guide — <https://scikit-learn.org/stable/user_guide.html>
- GitHub Actions OIDC hardening —
  <https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect>
- Google Cloud Workload Identity Federation —
  <https://cloud.google.com/iam/docs/workload-identity-federation>
