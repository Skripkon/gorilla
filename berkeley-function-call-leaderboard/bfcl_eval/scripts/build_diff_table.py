#!/usr/bin/env python3
"""
Script to build a comparison table between two models from BFCL evaluation results.
Generates an HTML table showing performance differences between old and new models.
"""

import argparse
import sys
import pandas as pd
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_bfcl_root = _script_dir.parent.parent
if str(_bfcl_root) not in sys.path:
    sys.path.insert(0, str(_bfcl_root))

from bfcl_eval.utils import load_dataset_entry


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build comparison table between two models from BFCL evaluation results"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="benches/gorilla/berkeley-function-call-leaderboard/score/data_overall.csv",
        help="Path to the CSV file containing evaluation results"
    )
    
    parser.add_argument(
        "--model-old",
        type=str,
        default="GigaChat 2 MAX (PROD)",
        help="Name of the old model in the results CSV"
    )
    
    parser.add_argument(
        "--model-new",
        type=str,
        default="GigaChat 2 MAX (IFT)",
        help="Name of the new model in the results CSV"
    )
    
    parser.add_argument(
        "--display-old",
        type=str,
        default="Giga 28.2",
        help="Display name for the old model in the output table"
    )
    
    parser.add_argument(
        "--display-new",
        type=str,
        default="Giga 30.1",
        help="Display name for the new model in the output table"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="bfcl_v4.html",
        help="Output HTML file name"
    )
    
    return parser.parse_args()


def load_and_filter_results(results_path: str, model_old: str, model_new: str) -> pd.DataFrame:
    """Load results from CSV and filter for the specified models."""
    results = pd.read_csv(results_path)
    results = results[results["Model"].isin([model_old, model_new])]
    return results


def get_evaluation_columns() -> list[str]:
    return [
        "Non-Live Simple AST",
        "Non-Live Multiple AST",
        "Live Simple AST",
        "Live Multiple AST",
        "Relevance Detection",
        "Irrelevance Detection",
        "Multi Turn Base",
        "Multi Turn Miss Func",
        "Multi Turn Miss Param",
        "Multi Turn Long Context",
        "Memory KV",
        "Memory Vector",
        "Memory Recursive Summarization"
    ]


COLUMN_TO_BFCL_CATEGORIES: dict[str, list[str]] = {
    "Non-Live Simple AST": ["simple_python", "simple_java", "simple_javascript"],
    "Non-Live Multiple AST": ["multiple"],
    "Live Simple AST": ["live_simple"],
    "Live Multiple AST": ["live_multiple"],
    "Relevance Detection": ["live_relevance"],
    "Irrelevance Detection": ["irrelevance", "live_irrelevance"],
    "Multi Turn Base": ["multi_turn_base"],
    "Multi Turn Miss Func": ["multi_turn_miss_func"],
    "Multi Turn Miss Param": ["multi_turn_miss_param"],
    "Multi Turn Long Context": ["multi_turn_long_context"],
    "Memory KV": ["memory_kv"],
    "Memory Vector": ["memory_vector"],
    "Memory Recursive Summarization": ["memory_rec_sum"],
}


def get_subset_sizes(column_names: list[str]) -> dict[str, int]:
    sizes = {}
    for col in column_names:
        categories = COLUMN_TO_BFCL_CATEGORIES[col]
        total = 0
        for cat in categories:
            entries = load_dataset_entry(
                cat, include_prereq=False, include_language_specific_hint=False
            )
            total += len(entries)
        sizes[col] = total
    return sizes


def process_results(results: pd.DataFrame, column_names: list[str]) -> pd.DataFrame:
    """Process results by extracting relevant columns and converting percentages to floats."""
    main_results = results[column_names]
    # Convert percentage strings to float
    main_results = main_results.applymap(lambda x: float(str(x).strip("%")))
    return main_results


def create_comparison_table(
    results: pd.DataFrame,
    main_results: pd.DataFrame,
    model_old: str,
    model_new: str,
    display_old: str,
    display_new: str,
    subset_sizes: dict[str, int],
) -> pd.DataFrame:
    model_to_display = {model_old: display_old, model_new: display_new}
    main_results_named = main_results.copy()
    main_results_named.index = results["Model"].map(model_to_display)

    old_data = main_results_named.loc[display_old]
    new_data = main_results_named.loc[display_new]

    df_display = pd.DataFrame({
        display_old: old_data,
        display_new: new_data
    })
    df_display["Diff"] = df_display[display_new] - df_display[display_old]

    total_size = sum(subset_sizes.values())
    macro_avg_row = pd.Series({
        display_old: df_display[display_old].mean(),
        display_new: df_display[display_new].mean(),
        "Diff": df_display["Diff"].mean()
    }, name="Macro Average")
    micro_old = sum(df_display[display_old] * df_display.index.map(subset_sizes)) / total_size
    micro_new = sum(df_display[display_new] * df_display.index.map(subset_sizes)) / total_size
    micro_avg_row = pd.Series({
        display_old: micro_old,
        display_new: micro_new,
        "Diff": micro_new - micro_old
    }, name="Micro Average")

    df_display = pd.concat([df_display, macro_avg_row.to_frame().T])
    df_display = pd.concat([df_display, micro_avg_row.to_frame().T])
    return df_display


def color_diff(val: float) -> str:
    """Apply color formatting based on difference value."""
    color = 'green' if val > 0 else 'red' if val < 0 else 'black'
    return f'color: {color}'


def style_table(df_display: pd.DataFrame, display_old: str, display_new: str) -> pd.DataFrame.style:
    """Apply styling to the comparison table."""
    styled = df_display.style.format({
        display_old: "{:.2f}",
        display_new: "{:.2f}",
        "Diff": "{:+.2f}"
    }).applymap(color_diff, subset=["Diff"]) \
     .set_table_styles([
         {'selector': 'th.row_heading', 'props': [('text-align', 'left')]},
         {'selector': 'td, th', 'props': [('border', '1px solid #888')]},
         {'selector': 'table', 'props': [('border-collapse', 'collapse')]},
         {'selector': 'tr:nth-last-child(-n+2)', 'props': [('font-weight', 'bold'), ('background-color', '#f0f0f0')]}
     ], axis=1)
    
    return styled


def save_html_table(styled_table: pd.DataFrame.style, output_path: str) -> None:
    """Save the styled table as an HTML file."""
    html_str = styled_table.to_html()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_str)
    print(f"Comparison table saved to: {output_path}")


def main():
    """Main function to orchestrate the table generation process."""
    args = parse_arguments()
    
    # Validate input file exists
    if not Path(args.results_dir).exists():
        raise FileNotFoundError(f"Results file not found: {args.results_dir}")
    
    # Load and filter results
    results = load_and_filter_results(args.results_dir, args.model_old, args.model_new)
    
    # Get evaluation columns
    column_names = get_evaluation_columns()
    
    # Process results
    main_results = process_results(results, column_names)
    subset_sizes = get_subset_sizes(column_names)

    df_display = create_comparison_table(
        results, main_results,
        args.model_old, args.model_new,
        args.display_old, args.display_new,
        subset_sizes,
    )
    
    # Style the table
    styled_table = style_table(df_display, args.display_old, args.display_new)
    
    # Save to HTML
    save_html_table(styled_table, args.output)


if __name__ == "__main__":
    main()