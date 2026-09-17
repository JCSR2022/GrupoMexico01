# Grupo Mexico Automatic ETL

An AI-assisted ETL and data-quality project for preparing mining drill-bit
data for analysis and sourcing decisions. The project started as the
`AutomaticETL.ipynb` notebook and has been modularized into reusable Python
components suitable for local execution and future production deployment.

## Project scope

The workflow currently supports:

- Loading an Excel dataset into a controlled ETL context.
- Inspecting dataset shape, columns, data types, missing values, unique values,
  duplicates, memory usage, and sample values.
- Detecting missing values, duplicate rows, numeric outliers, inconsistent
  string formatting, and mixed data types.
- Applying controlled transformations:
  - Fill or drop missing values.
  - Normalize string columns.
  - Convert column data types.
  - Remove duplicate rows.
- Exporting the cleaned dataset as Excel or CSV.
- Maintaining a data-quality and process log for every detected issue and
  transformation.
- Generating a quality report and process report.
- Optionally using a Groq/LangChain agent to choose and execute the available
  ETL tools through natural-language interaction.

The current implementation is a data-quality and preparation layer. It does
not yet define business-specific rules for supplier scoring, drill-bit
performance, cost optimization, or automated approval of questionable data.
Those rules should be added explicitly and validated with the business owners
before production use.

## Architecture

```text
Input Excel file
       |
       v
main.py --------------> ETLContext
       |                    |
       |                    v
       +--------------> tools.py
                            |
                            +--> detections
                            +--> transformations
                            +--> quality/process log
                            +--> Excel/CSV exports
       |
       +--------------> agent.py (optional Groq/LangChain interface)
```

### Main files

| File | Responsibility |
| --- | --- |
| `main.py` | Interactive command-line entry point and conversation loop. |
| `tools.py` | ETL state, data-quality tools, transformations, reports, and exports. |
| `agent.py` | Optional Groq/LangChain agent configuration. |
| `AutomaticETL.ipynb` | Original exploratory notebook and reference workflow. |
| `DrillingBits.xlsx` | Example input dataset. |
| `requirements.txt` | Python dependencies. |
| `.env.example` | Environment-variable template. |
| `env` | Compatibility module that reads `GROQ_API_KEY` from the environment. |

## Requirements

- Python 3.12 or a compatible recent Python version.
- A virtual environment is recommended.
- An Excel input file.
- A Groq API key if the interactive AI agent is used.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Configuration

Set the Groq credentials in the process environment or in a local `.env`
file managed by your deployment system. The application loads `.env`
automatically when it starts. Do not commit secrets.

Linux/macOS:

```bash
export GROQ_API_KEY="your-groq-api-key"
export GROQ_MODEL="openai/gpt-oss-20b"
```

Windows PowerShell:

```powershell
$env:GROQ_API_KEY = "your-groq-api-key"
$env:GROQ_MODEL = "openai/gpt-oss-20b"
```

In production, configure secrets through the platform's secret manager or
environment configuration instead of committing a `.env` file.
The API key that was previously exposed in local configuration should be
revoked and regenerated.

## Interactive usage

Start an agent session with the sample workbook:

```bash
python main.py --input DrillingBits.xlsx --output-dir output
```

The application displays:

```text
ETL agent ready. Type 'quit', 'exit', or 'q' to finish.
You:
```

Enter natural-language requests, for example:

```text
Inspect the dataset.
```

```text
Detect missing values, duplicates, outliers, and mixed data types.
```

```text
Normalize the MINA column using uppercase.
```

```text
Fill missing values in Precio using the median.
```

```text
Remove duplicate rows and generate a quality report.
```

```text
Export the cleaned dataset as `clean_drilling_bits.csv`.
```

Finish the session with any of:

```text
quit
exit
q
```

The application always attempts to save the current dataset and data-quality
log when the session ends:

```text
output/clean_dataset.xlsx
output/dq_log.xlsx
```

The output directory is created automatically if it does not exist.

## Available ETL tools

The tools are implemented in `tools.py` and are exposed to the agent through
`create_tools(context)`.

| Tool | Purpose |
| --- | --- |
| `inspect_dataset` | Profile the current dataset. |
| `detect_missing_values` | Find and log missing values. |
| `detect_duplicates` | Find and log duplicate rows. |
| `detect_outliers` | Detect numeric values outside the 1.5 IQR bounds. |
| `detect_Inconsistent_formatting` | Find whitespace and case variations in string columns. |
| `detect_Mixed_data_types` | Detect multiple Python value types in a column. |
| `normalize_column` | Apply `strip`, `lower`, `upper`, or `title` to string values. |
| `convert_datatype` | Convert a column to integer, float, string, Boolean, or datetime. |
| `remove_duplicates` | Remove duplicate rows while keeping the first occurrence. |
| `export_clean_dataset` | Export the current data as Excel or CSV. |
| `solve_Inconsistent_formatting` | Resolve formatting issues in one column. |
| `solve_Mixed_data_types` | Resolve mixed types through explicit conversion. |
| `solve_quality_issues` | Dispatch a supported quality fix. |
| `generate_quality_report` | Return dataset profile, summary, and DQ log. |
| `generate_log_process` | Return transformation and export records. |
| `save_df` | Save the dataset as an Excel workbook. |
| `save_etl_context_log` | Save the DQ log as an Excel workbook. |

## Data-quality log

Every detection or transformation may add an entry to `ETLContext.dq_log`.
Entries include:

- Sequential log ID.
- UTC timestamp.
- Record type: `issue` or `process`.
- Tool that created the entry.
- Affected column, when applicable.
- Issue type and description.
- Number of affected rows.
- Proposed or executed action.
- Current status.

The log is the authoritative record of the quality issues identified during
the current run. It is exported to `dq_log.xlsx` when the session ends.

## Python usage without the agent

The tools can also be used deterministically from Python:

```python
from pathlib import Path

import pandas as pd

from tools import (
    ETLContext,
    detect_missing_values,
    detect_outliers,
    inspect_dataset,
    save_df,
    save_etl_context_log,
)

context = ETLContext(
    pd.read_excel("DrillingBits.xlsx"),
    output_dir=Path("output"),
)

profile = inspect_dataset(context)
missing = detect_missing_values(context)
outliers = detect_outliers(context)
save_df(context)
save_etl_context_log(context)
```

This deterministic mode is recommended for scheduled jobs and production
pipelines where transformations should be predefined and reproducible.

## Production considerations

Before deploying to production:

1. Store API keys in a secret manager and rotate any key previously exposed in
   source files.
2. Pin tested dependency versions for reproducible deployments.
3. Add automated tests for each transformation and business rule.
4. Define acceptable thresholds for missing values, duplicates, and outliers.
5. Validate date formats and domain-specific categorical values.
6. Decide whether transformations should fail the job or quarantine invalid
   rows.
7. Add structured application logging and monitoring.
8. Use a unique run identifier for each ETL execution.
9. Avoid allowing an AI agent to make irreversible production changes without
   validation or approval.
10. Define input and output storage locations appropriate for the deployment
    environment.

## Limitations

- The current command-line workflow expects an Excel input file.
- Date parsing and domain validation are not yet specialized for every source
  format.
- Outlier detection is statistical and does not replace business review.
- The AI agent requires a valid Groq API key and network access.
- Agent behavior depends on the selected model; deterministic tool execution
  should be preferred for scheduled production jobs.

## License and ownership

Add the applicable organizational license, data classification, and ownership
information before publishing or deploying this project outside its intended
environment.
