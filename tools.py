"""Reusable tools for the data-quality and ETL workflow.

The functions in this module operate on an :class:`ETLContext` instead of a
module-level global DataFrame.  This makes a run isolated, testable, and safe
to use in a long-running process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import StructuredTool


@dataclass
class ETLContext:
    """State for one ETL execution."""

    df: pd.DataFrame
    output_dir: Path = field(default_factory=lambda: Path.cwd())
    dq_log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        cleaned_columns = [str(column).strip() for column in self.df.columns]
        if len(cleaned_columns) != len(set(cleaned_columns)):
            raise ValueError("Input dataset has duplicate column names after trimming whitespace.")
        self.df = self.df.copy()
        self.df.columns = cleaned_columns
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_issue(
        self,
        *,
        record_type: str,
        tool: str,
        column: str | None,
        issue_type: str,
        description: str,
        affected_rows: int | None,
        action: str,
        status: str = "Identified",
    ) -> None:
        self.dq_log.append(
            {
                "id": len(self.dq_log) + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "record_type": record_type,
                "tool": tool,
                "column": column,
                "issue_type": issue_type,
                "description": description,
                "affected_rows": affected_rows,
                "action": action,
                "status": status,
            }
        )


def _clean_column_reference(column: str) -> str:
    """Remove presentation markup and surrounding quotes from a column name."""
    cleaned = str(column).strip().strip("*").strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()
    cleaned = cleaned.strip("*").strip()
    return cleaned


def _resolve_column(context: ETLContext, column: str) -> str:
    """Resolve a user/model column reference against canonical column names."""
    requested = _clean_column_reference(column)
    exact_matches = [name for name in context.df.columns if str(name) == requested]
    if exact_matches:
        return exact_matches[0]
    normalized_matches = [
        name for name in context.df.columns if str(name).strip().casefold() == requested.casefold()
    ]
    if len(normalized_matches) == 1:
        return normalized_matches[0]
    if not normalized_matches:
        raise ValueError(f"Column '{column}' does not exist.")
    raise ValueError(f"Column reference '{column}' is ambiguous: {normalized_matches}.")


def _output_path(context: ETLContext, filename: str) -> Path:
    """Return a path inside the configured output directory."""
    name = Path(filename).name
    if not name:
        raise ValueError("Output filename cannot be empty.")
    path = context.output_dir / name
    if path.suffix.lower() != ".xlsx":
        path = path.with_suffix(".xlsx")
    return path


def inspect_dataset(context: ETLContext) -> dict[str, Any]:
    """Return a compact, JSON-safe profile of the current dataset."""
    df = context.df
    columns = []
    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        sample_size = min(5, len(non_null))
        sample_values = non_null.head(sample_size).tolist()
        columns.append(
            {
                "name": str(column),
                "dtype": str(series.dtype),
                "missing_values": int(series.isna().sum()),
                "missing_percentage": round(float(series.isna().mean() * 100), 2),
                "unique_values": int(series.nunique(dropna=True)),
                "sample_values": [str(value) for value in sample_values],
            }
        )
    return {
        "tool": "inspect_dataset",
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": columns,
        "duplicates": {"count": int(df.duplicated().sum())},
        "memory_usage_mb": round(float(df.memory_usage(deep=True).sum() / 1024**2), 2),
    }


def detect_missing_values(context: ETLContext) -> dict[str, Any]:
    """Find missing values and register each affected column in the DQ log."""
    df = context.df
    missing = df.isna().sum()
    issues = []
    for column, count in missing[missing > 0].items():
        percentage = round(float(count / len(df) * 100), 2) if len(df) else 0.0
        issue = {
            "column": str(column),
            "missing_values": int(count),
            "missing_percentage": percentage,
        }
        issues.append(issue)
        context.log_issue(
            record_type="issue",
            tool="detect_missing_values",
            column=str(column),
            issue_type="Missing values",
            description=f"Column contains {int(count)} missing values ({percentage}%).",
            affected_rows=int(count),
            action="Investigate and apply an appropriate missing-value treatment.",
        )
    return {
        "tool": "detect_missing_values",
        "columns_with_missing_values": issues,
        "message": "No missing values detected." if not issues else None,
    }


def detect_duplicates(context: ETLContext) -> dict[str, Any]:
    """Detect duplicate rows and register the issue."""
    duplicate_mask = context.df.duplicated(keep="first")
    count = int(duplicate_mask.sum())
    if count:
        context.log_issue(
            record_type="issue",
            tool="detect_duplicates",
            column=None,
            issue_type="Duplicate rows",
            description=f"Detected {count} duplicate rows.",
            affected_rows=count,
            action="Review and remove duplicate rows when they are not valid repeats.",
        )
    return {
        "tool": "detect_duplicates",
        "duplicate_rows": count,
        "message": "No duplicate rows detected." if count == 0 else None,
    }


def detect_outliers(context: ETLContext, columns: list[str] | None = None) -> dict[str, Any]:
    """Detect numeric outliers using the 1.5 IQR rule."""
    df = context.df
    selected = columns or [
        str(column) for column in df.select_dtypes(include="number").columns
    ]
    issues = []
    for column in selected:
        column = _resolve_column(context, column)
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        values = df[column].dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = df[column].notna() & ((df[column] < lower) | (df[column] > upper))
        count = int(mask.sum())
        if count:
            issue = {
                "column": column,
                "outlier_rows": count,
                "lower_bound": float(lower),
                "upper_bound": float(upper),
            }
            issues.append(issue)
            context.log_issue(
                record_type="issue",
                tool="detect_outliers",
                column=column,
                issue_type="Outliers",
                description=f"Detected {count} values outside the 1.5 IQR bounds.",
                affected_rows=count,
                action="Review outliers against the relevant business rules.",
            )
    return {"tool": "detect_outliers", "columns_with_outliers": issues}


def detect_Inconsistent_formatting(
    context: ETLContext, columns: list[str] | None = None
) -> dict[str, Any]:
    """Find string values that differ only by whitespace or letter case."""
    df = context.df
    selected = columns or [str(c) for c in df.select_dtypes(include="object").columns]
    issues = []
    for column in selected:
        column = _resolve_column(context, column)
        series = df[column].dropna().astype(str)
        normalized = series.str.strip().str.casefold()
        inconsistent = int((series != series.str.strip()).sum())
        case_groups = int(
            (pd.DataFrame({"raw": series, "normalized": normalized})
             .groupby("normalized", dropna=False)["raw"]
             .nunique() > 1)
            .sum()
        )
        if inconsistent or case_groups:
            issue = {
                "column": column,
                "whitespace_variations": inconsistent,
                "case_variations": max(case_groups, 0),
            }
            issues.append(issue)
            context.log_issue(
                record_type="issue",
                tool="detect_Inconsistent_formatting",
                column=column,
                issue_type="Inconsistent formatting",
                description="Values contain whitespace or case variations.",
                affected_rows=inconsistent,
                action="Normalize whitespace and letter casing where appropriate.",
            )
    return {"tool": "detect_Inconsistent_formatting", "issues": issues}


def detect_Mixed_data_types(
    context: ETLContext, columns: list[str] | None = None
) -> dict[str, Any]:
    """Detect object columns whose non-null values infer to multiple types."""
    df = context.df
    selected = columns or [str(c) for c in df.columns]
    issues = []
    for column in selected:
        column = _resolve_column(context, column)
        series = df[column].dropna()
        if series.empty or not (series.dtype == "object" or series.dtype.name == "string"):
            continue
        type_counts = series.map(lambda value: type(value).__name__).value_counts()
        if len(type_counts) > 1:
            issue = {"column": column, "types": type_counts.astype(int).to_dict()}
            issues.append(issue)
            context.log_issue(
                record_type="issue",
                tool="detect_Mixed_data_types",
                column=column,
                issue_type="Mixed data types",
                description=f"Values contain multiple Python types: {issue['types']}.",
                affected_rows=int(len(series)),
                action="Convert the column to one explicitly selected data type.",
            )
    return {"tool": "detect_Mixed_data_types", "issues": issues}


def normalize_column(
    context: ETLContext,
    column: str,
    operation: str = "strip",
) -> dict[str, Any]:
    """Normalize a string column using strip, lower, upper, or title."""
    column = _resolve_column(context, column)
    if not pd.api.types.is_string_dtype(context.df[column]):
        raise ValueError(f"Column '{column}' is not a string column.")
    operation = operation.strip().lower()
    if operation not in {"strip", "lower", "upper", "title"}:
        raise ValueError("Operation must be one of: strip, lower, upper, title.")
    values = context.df[column].astype("string")
    normalized = values.str.strip()
    if operation == "lower":
        normalized = normalized.str.lower()
    elif operation == "upper":
        normalized = normalized.str.upper()
    elif operation == "title":
        normalized = normalized.str.title()
    context.df[column] = normalized
    context.log_issue(
        record_type="process",
        tool="normalize_column",
        column=column,
        issue_type="Formatting",
        description=f"Applied '{operation}' normalization.",
        affected_rows=int(len(context.df)),
        action=f"Normalize column '{column}' using '{operation}'.",
        status="Completed",
    )
    return {"tool": "normalize_column", "column": column, "operation": operation}


def convert_datatype(
    context: ETLContext,
    column: str,
    datatype: str,
) -> dict[str, Any]:
    """Convert a column to numeric, string, boolean, or datetime."""
    column = _resolve_column(context, column)
    datatype = datatype.strip().lower()
    if datatype in {"int", "int64", "integer"}:
        converted = pd.to_numeric(context.df[column], errors="raise").astype("int64")
    elif datatype in {"float", "float64", "number", "numeric"}:
        converted = pd.to_numeric(context.df[column], errors="raise").astype("float64")
    elif datatype in {"str", "string", "object"}:
        converted = context.df[column].astype("string")
    elif datatype in {"bool", "boolean"}:
        converted = context.df[column].astype("boolean")
    elif datatype in {"datetime", "datetime64"}:
        converted = pd.to_datetime(context.df[column], errors="raise")
    else:
        raise ValueError("Datatype must be int, float, string, bool, or datetime.")
    context.df[column] = converted
    context.log_issue(
        record_type="process",
        tool="convert_datatype",
        column=column,
        issue_type="Data type",
        description=f"Converted column to '{datatype}'.",
        affected_rows=int(len(context.df)),
        action=f"Convert '{column}' to '{datatype}'.",
        status="Completed",
    )
    return {"tool": "convert_datatype", "column": column, "datatype": datatype}


def remove_duplicates(context: ETLContext, subset: list[str] | None = None) -> dict[str, Any]:
    """Remove duplicate rows and preserve the first occurrence."""
    before = len(context.df)
    if subset:
        subset = [_resolve_column(context, column) for column in subset]
    context.df = context.df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    removed = before - len(context.df)
    context.log_issue(
        record_type="process",
        tool="remove_duplicates",
        column=None,
        issue_type="Duplicate rows",
        description=f"Removed {removed} duplicate rows.",
        affected_rows=removed,
        action="Keep the first occurrence of each duplicate row.",
        status="Completed",
    )
    return {"tool": "remove_duplicates", "removed_rows": removed, "remaining_rows": len(context.df)}


def export_clean_dataset(
    context: ETLContext, filename: str = "clean_dataset.xlsx", file_format: str = "xlsx"
) -> dict[str, Any]:
    """Export the current dataset as Excel or CSV."""
    file_format = file_format.strip().lower()
    if file_format not in {"xlsx", "csv"}:
        raise ValueError("file_format must be 'xlsx' or 'csv'.")
    path = context.output_dir / Path(filename).name
    path = path.with_suffix(f".{file_format}")
    if file_format == "xlsx":
        context.df.to_excel(path, index=False)
    else:
        context.df.to_csv(path, index=False)
    context.log_issue(
        record_type="process",
        tool="export_clean_dataset",
        column=None,
        issue_type="Export",
        description=f"Dataset exported to '{path.name}'.",
        affected_rows=len(context.df),
        action=f"Export clean dataset as {file_format}.",
        status="Completed",
    )
    return {"tool": "export_clean_dataset", "status": "Completed", "path": str(path)}


def solve_Inconsistent_formatting(
    context: ETLContext, column: str, operation: str = "strip"
) -> dict[str, Any]:
    """Resolve formatting issues by normalizing one column."""
    result = normalize_column(context, column, operation)
    result["tool"] = "solve_Inconsistent_formatting"
    return result


def solve_Mixed_data_types(
    context: ETLContext, column: str, datatype: str
) -> dict[str, Any]:
    """Resolve mixed types by converting one column explicitly."""
    result = convert_datatype(context, column, datatype)
    result["tool"] = "solve_Mixed_data_types"
    return result


def solve_quality_issues(
    context: ETLContext,
    issue_type: str,
    column: str | None = None,
    strategy: str | None = None,
) -> dict[str, Any]:
    """Dispatch a supported quality fix through one controlled interface."""
    normalized = issue_type.strip().lower().replace("_", " ")
    if normalized in {"missing", "missing value", "missing values"}:
        if not column or not strategy:
            raise ValueError("column and strategy are required for missing values.")
        return solve_missing_values(context, column, strategy)
    if normalized in {"duplicate", "duplicates", "duplicate rows"}:
        return remove_duplicates(context)
    if normalized in {"formatting", "inconsistent formatting"}:
        if not column:
            raise ValueError("column is required for formatting issues.")
        return solve_Inconsistent_formatting(context, column, strategy or "strip")
    if normalized in {"mixed types", "mixed data types"}:
        if not column or not strategy:
            raise ValueError("column and strategy/datatype are required for mixed types.")
        return solve_Mixed_data_types(context, column, strategy)
    raise ValueError(f"Unsupported issue type '{issue_type}'.")


def generate_quality_report(context: ETLContext) -> dict[str, Any]:
    """Generate a JSON-safe quality report from the current state and log."""
    profile = inspect_dataset(context)
    issue_records = [entry for entry in context.dq_log if entry["record_type"] == "issue"]
    process_records = [entry for entry in context.dq_log if entry["record_type"] == "process"]
    return {
        "tool": "generate_quality_report",
        "dataset": profile,
        "summary": {
            "issues_identified": len(issue_records),
            "processes_executed": len(process_records),
            "log_entries": len(context.dq_log),
        },
        "data_quality_log": list(context.dq_log),
    }


def generate_log_process(context: ETLContext) -> dict[str, Any]:
    """Return only transformation and export records from the ETL log."""
    return {
        "tool": "generate_log_process",
        "process_log": [
            entry for entry in context.dq_log if entry["record_type"] == "process"
        ],
    }


def solve_missing_values(
    context: ETLContext, column: str, strategy: str
) -> dict[str, Any]:
    """Resolve missing values in one column using a controlled strategy."""
    column = _resolve_column(context, column)

    strategy = strategy.strip().lower()
    supported = {"drop", "mean", "median", "mode"}
    if strategy not in supported:
        raise ValueError(
            f"Unsupported strategy '{strategy}'. Use: {', '.join(sorted(supported))}."
        )

    before = int(context.df[column].isna().sum())
    if before == 0:
        return {
            "tool": "solve_missing_values",
            "column": column,
            "strategy": strategy,
            "missing_values_before": 0,
            "missing_values_after": 0,
            "status": "No action required",
        }

    if strategy == "drop":
        context.df = context.df.dropna(subset=[column]).reset_index(drop=True)
        action = f"Removed rows with missing values in '{column}'."
    else:
        if strategy in {"mean", "median"} and not pd.api.types.is_numeric_dtype(
            context.df[column]
        ):
            raise ValueError(
                f"Strategy '{strategy}' cannot be applied to non-numeric column '{column}'."
            )
        if strategy == "mean":
            value = context.df[column].mean()
        elif strategy == "median":
            value = context.df[column].median()
        else:
            mode = context.df[column].mode()
            if mode.empty:
                raise ValueError(f"Could not determine mode for column '{column}'.")
            value = mode.iloc[0]
        context.df[column] = context.df[column].fillna(value)
        action = f"Filled missing values in '{column}' using the {strategy}."

    after = int(context.df[column].isna().sum())
    status = "Resolved" if after == 0 else "Partially Resolved"
    context.log_issue(
        record_type="process",
        tool="solve_missing_values",
        column=column,
        issue_type="Missing values",
        description=f"Applied '{strategy}' to {before} missing values.",
        affected_rows=before,
        action=action,
        status=status,
    )
    return {
        "tool": "solve_missing_values",
        "column": column,
        "strategy": strategy,
        "missing_values_before": before,
        "missing_values_after": after,
        "status": status,
    }


def save_df(context: ETLContext, filename: str = "clean_dataset.xlsx") -> dict[str, Any]:
    """Export the current dataset as an Excel workbook."""
    path = _output_path(context, filename)
    context.df.to_excel(path, index=False)
    context.log_issue(
        record_type="process",
        tool="save_df",
        column=None,
        issue_type="Export",
        description=f"Clean dataset exported to '{path.name}'.",
        affected_rows=len(context.df),
        action=f"Save dataset as Excel file '{path.name}'.",
        status="Completed",
    )
    return {
        "tool": "save_df",
        "status": "Completed",
        "file": path.name,
        "path": str(path),
        "rows": int(len(context.df)),
        "columns": int(len(context.df.columns)),
    }


def save_etl_context_log(
    context: ETLContext, filename: str = "dq_log.xlsx"
) -> dict[str, Any]:
    """Export the data-quality log as an Excel workbook."""
    path = _output_path(context, filename)
    pd.DataFrame(context.dq_log).to_excel(path, index=False)
    return {
        "tool": "save_etl_context_log",
        "status": "Completed",
        "file": path.name,
        "path": str(path),
        "entries": int(len(context.dq_log)),
    }


def create_tools(context: ETLContext) -> list[StructuredTool]:
    """Create LangChain tools bound to one ETL context."""
    return [
        StructuredTool.from_function(
            lambda: inspect_dataset(context),
            name="inspect_dataset",
            description="Inspect the current dataset without modifying it.",
        ),
        StructuredTool.from_function(
            lambda: detect_missing_values(context),
            name="detect_missing_values",
            description="Detect and log missing values in the current dataset.",
        ),
        StructuredTool.from_function(
            lambda: detect_duplicates(context),
            name="detect_duplicates",
            description="Detect and log duplicate rows.",
        ),
        StructuredTool.from_function(
            lambda columns=None: detect_outliers(context, columns),
            name="detect_outliers",
            description="Detect numeric outliers using the 1.5 IQR rule.",
        ),
        StructuredTool.from_function(
            lambda columns=None: detect_Inconsistent_formatting(context, columns),
            name="detect_Inconsistent_formatting",
            description="Detect whitespace and letter-case inconsistencies.",
        ),
        StructuredTool.from_function(
            lambda columns=None: detect_Mixed_data_types(context, columns),
            name="detect_Mixed_data_types",
            description="Detect columns containing multiple value types.",
        ),
        StructuredTool.from_function(
            lambda column, operation="strip": normalize_column(context, column, operation),
            name="normalize_column",
            description="Normalize a string column using strip, lower, upper, or title.",
        ),
        StructuredTool.from_function(
            lambda column, datatype: convert_datatype(context, column, datatype),
            name="convert_datatype",
            description="Convert a column to an explicit data type.",
        ),
        StructuredTool.from_function(
            lambda subset=None: remove_duplicates(context, subset),
            name="remove_duplicates",
            description="Remove duplicate rows, keeping the first occurrence.",
        ),
        StructuredTool.from_function(
            lambda filename="clean_dataset.xlsx", file_format="xlsx": export_clean_dataset(
                context, filename, file_format
            ),
            name="export_clean_dataset",
            description="Export the dataset as Excel or CSV.",
        ),
        StructuredTool.from_function(
            lambda column, operation="strip": solve_Inconsistent_formatting(
                context, column, operation
            ),
            name="solve_Inconsistent_formatting",
            description="Resolve formatting inconsistencies in a column.",
        ),
        StructuredTool.from_function(
            lambda column, datatype: solve_Mixed_data_types(context, column, datatype),
            name="solve_Mixed_data_types",
            description="Resolve mixed data types by converting a column.",
        ),
        StructuredTool.from_function(
            lambda issue_type, column=None, strategy=None: solve_quality_issues(
                context, issue_type, column, strategy
            ),
            name="solve_quality_issues",
            description="Apply a supported quality fix.",
        ),
        StructuredTool.from_function(
            lambda: generate_quality_report(context),
            name="generate_quality_report",
            description="Generate a quality report from the current dataset and log.",
        ),
        StructuredTool.from_function(
            lambda: generate_log_process(context),
            name="generate_log_process",
            description="Return the ETL transformation and export process log.",
        ),
        StructuredTool.from_function(
            lambda filename="clean_dataset.xlsx": save_df(context, filename),
            name="save_df",
            description="Export the current dataset to an Excel file.",
        ),
        StructuredTool.from_function(
            lambda filename="dq_log.xlsx": save_etl_context_log(context, filename),
            name="save_etl_context_log",
            description="Export the ETL data-quality log to an Excel file.",
        ),
    ]


__all__ = [
    "ETLContext",
    "create_tools",
    "convert_datatype",
    "detect_Inconsistent_formatting",
    "detect_Mixed_data_types",
    "detect_duplicates",
    "detect_missing_values",
    "detect_outliers",
    "export_clean_dataset",
    "generate_log_process",
    "generate_quality_report",
    "inspect_dataset",
    "normalize_column",
    "remove_duplicates",
    "save_df",
    "save_etl_context_log",
    "solve_Inconsistent_formatting",
    "solve_Mixed_data_types",
    "solve_missing_values",
    "solve_quality_issues",
]
