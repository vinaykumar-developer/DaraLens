# DataLens

**DataLens** is a Streamlit web app that automates the first pass of exploratory
data analysis (EDA) on any CSV file. Upload a dataset, and it profiles every
column, runs a set of deterministic checks against it, and shows you a
prioritized list of plain-English insights — no manual `df.describe()` /
`df.isna().sum()` digging required.

## What it does

1. **Upload** a CSV.
2. **Profile** every column — dtype, % missing, unique value count, and a
   simplified type classification (numeric / categorical / datetime /
   constant / text), plus skew and standard deviation for numeric columns.
3. **Run insight rules** against that profile — missing values, skewed
   distributions, highly correlated column pairs, outliers (IQR method),
   constant columns, high-cardinality categoricals, and possible class
   imbalance in a likely target column.
4. **Display findings**, sorted by severity (🔴 critical → 🟠 warning →
   🔵 info), each with a short human-readable explanation containing the
   real computed numbers (percentages, correlation values, etc).

## Architecture

```
datalens/
├── app.py            # Streamlit UI — entry point
├── profiling.py       # Column profiling logic
├── insight_rules.py    # All insight rule functions + rule registry
├── engine.py            # Runs all rules, collects + sorts findings
├── requirements.txt
└── README.md
```

The pipeline flows in one direction:

```
CSV upload → profiling.py (profile_columns) → insight_rules.py (INSIGHT_RULES)
           → engine.py (generate_insights)  → app.py (renders findings)
```

- **`profiling.py`** turns a raw DataFrame into a structured dictionary
  describing each column. This is the single source of truth every rule
  reads from — rules never re-inspect raw dtypes themselves.
- **`insight_rules.py`** contains one small function per check. Every
  function has the same signature — `(df, profile) -> list[finding]` — and
  is registered in the `INSIGHT_RULES` list. This is a **registry pattern**:
  the engine doesn't know or care what a rule does internally, it just calls
  every function in the list and collects whatever findings come back.
- **`engine.py`** is the orchestrator: profile the data, run every rule
  (each wrapped in its own `try/except` so one bad rule can't crash the
  app), flatten the results, and sort by severity.
- **`app.py`** is a thin UI layer — it never contains analysis logic itself,
  it just calls `generate_insights()` and `profile_columns()` and renders
  the results.

### Why rule-based, not ML/NLP

This project deliberately avoids machine learning or LLM-generated text for
the insight messages. Every message is built from an f-string template
filled in with real, computed values (e.g. `"skew = 1.42"`, `"38.0% missing"`).
This makes the system **fully explainable and deterministic** — the same
CSV always produces the same insights, and every finding can be traced back
to a specific, auditable rule and threshold. That traceability is the whole
point of the design, and is easy to defend in a viva.

## How to run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## How to add a new insight rule

This is the main extensibility point of the project — adding a rule takes
exactly two steps and touches no other file:

1. **Write the rule function** in `insight_rules.py`, following the standard
   signature:

   ```python
   def check_something(df, profile):
       findings = []
       for col, info in profile.items():
           if <some condition on info>:
               findings.append({
                   "column": col,
                   "message": f"Column '{col}' ...",
                   "severity": "warning",  # or "critical" / "info"
               })
       return findings
   ```

2. **Register it** by adding one line to the `INSIGHT_RULES` list at the
   bottom of `insight_rules.py`:

   ```python
   INSIGHT_RULES = [
       ...,
       {"name": "Something Check", "function": check_something},
   ]
   ```

`engine.py` and `app.py` require **no changes** — the new rule is picked up
automatically the next time insights are generated.

## Error handling

- Non-CSV or corrupted uploads are caught and shown as a friendly error
  instead of crashing.
- Empty files, or CSVs with 0 rows/columns, are detected and reported.
- The correlation-pairs rule automatically returns no findings if there are
  fewer than 2 numeric columns (instead of throwing an error).
- Every rule call in `engine.py` is wrapped in `try/except`, so a single
  rule failing on an unusual dataset never takes down the whole app.
