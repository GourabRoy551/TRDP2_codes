"""Generate LaTeX tables directly from the verified modular SHAP run.

The report never depends on copied numerical values. Re-running this script
updates every table and summary macro from the experiment CSV/JSON artifacts.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


REPORT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = REPORT_DIR.parent
RUN_NAME = "modular_s1_s10"
VALUES_DIR = PROJECT_DIR / "outputs" / "values" / RUN_NAME
EVALUATION_DIR = PROJECT_DIR / "outputs" / "evaluation" / RUN_NAME
TABLE_DIR = REPORT_DIR / "tables"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def latex_escape(value: object) -> str:
    """Escape text placed in ordinary LaTeX table cells."""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def format_probability(row: dict[str, str]) -> float:
    key = "positive_probability" if row["prediction"] == "POS" else "negative_probability"
    return float(row[key])


def write_summary_table(summary: list[dict[str, str]]) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Sentence-level classification and SHAP additivity results.}",
        r"\label{tab:classification}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccrrrc}",
        r"\toprule",
        r"ID & Gold & Pred. & Correct & $p(\hat{y}\mid x)$ & $f(x)$ & $\phi_0$ & $\sum_i\phi_i$ & $\epsilon_{\mathrm{add}}$ \\",
        r"\midrule",
    ]
    for row in summary:
        lines.append(
            f"{row['sentence_id']} & {row['gold_label']} & {row['prediction']} & "
            f"{int(float(row['correct']))} & {format_probability(row):.6f} & "
            f"{float(row['margin']):.4f} & {float(row['base_value']):.4f} & "
            f"{float(row['sum_shap_values']):.4f} & "
            f"{float(row['additivity_residual']):.1e} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""])
    (TABLE_DIR / "classification_results.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_top_words_table(summary: list[dict[str, str]]) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Largest positive and negative word-level attributions in each sentence.}",
        r"\label{tab:topwords}",
        r"\begin{tabular}{lllll}",
        r"\toprule",
        r"ID & Gold & Prediction & Largest positive attribution & Largest negative attribution \\",
        r"\midrule",
    ]
    for row in summary:
        lines.append(
            f"{row['sentence_id']} & {row['gold_label']} & {row['prediction']} & "
            f"{latex_escape(row['top_positive_word'])} & "
            f"{latex_escape(row['top_negative_word'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (TABLE_DIR / "top_words.tex").write_text("\n".join(lines), encoding="utf-8")


def write_sentence_table(summary: list[dict[str, str]]) -> None:
    lines = [
        r"\begin{longtable}{@{}llp{0.73\textwidth}@{}}",
        r"\caption{Complete qualitative sentence set used in the experiment.}\label{tab:sentences}\\",
        r"\toprule",
        r"ID & Gold & Sentence \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{3}{l}{\tablename\ \thetable{} -- continued}\\",
        r"\toprule",
        r"ID & Gold & Sentence \\",
        r"\midrule",
        r"\endhead",
        r"\midrule\multicolumn{3}{r}{Continued on next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for row in summary:
        lines.append(
            f"{row['sentence_id']} & {row['gold_label']} & "
            f"{latex_escape(row['text'])} \\\\"
        )
    lines.extend([r"\end{longtable}", ""])
    (TABLE_DIR / "sentences.tex").write_text("\n".join(lines), encoding="utf-8")


def write_faithfulness_tables(faithfulness: list[dict[str, str]]) -> None:
    by_fraction: dict[float, list[dict[str, str]]] = defaultdict(list)
    by_sentence: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in faithfulness:
        by_fraction[float(row["fraction"])].append(row)
        by_sentence[row["sentence_id"]].append(row)

    aggregate = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Faithfulness results averaged over the ten sentences. Standard deviations are population standard deviations.}",
        r"\label{tab:faithfulness-aggregate}",
        r"\small",
        r"\begin{tabular}{crrrrr}",
        r"\toprule",
        r"Fraction $\alpha$ & Mean selected words & Mean comp. & SD comp. & Mean suff. gap & SD suff. gap \\",
        r"\midrule",
    ]
    for fraction in sorted(by_fraction):
        rows = by_fraction[fraction]
        selected = [float(row["selected_count"]) for row in rows]
        comp = [float(row["comprehensiveness_drop"]) for row in rows]
        suff = [float(row["sufficiency_gap"]) for row in rows]
        aggregate.append(
            f"{fraction:.1f} & {mean(selected):.2f} & {mean(comp):.4f} & "
            f"{pstdev(comp):.4f} & {mean(suff):.4f} & {pstdev(suff):.4f} \\\\"
        )
    aggregate.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (TABLE_DIR / "faithfulness_aggregate.tex").write_text(
        "\n".join(aggregate), encoding="utf-8"
    )

    sentence_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Mean faithfulness scores by sentence across $\alpha\in\{0.1,0.2,0.3,0.5\}$.}",
        r"\label{tab:faithfulness-sentence}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"ID & Mean comprehensiveness drop & Mean sufficiency gap \\",
        r"\midrule",
    ]
    for sentence_id in sorted(by_sentence, key=lambda value: int(value[1:])):
        rows = by_sentence[sentence_id]
        sentence_lines.append(
            f"{sentence_id} & "
            f"{mean(float(row['comprehensiveness_drop']) for row in rows):.4f} & "
            f"{mean(float(row['sufficiency_gap']) for row in rows):.4f} \\\\"
        )
    sentence_lines.extend(
        [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    )
    (TABLE_DIR / "faithfulness_by_sentence.tex").write_text(
        "\n".join(sentence_lines), encoding="utf-8"
    )

    detail = [
        r"\begin{longtable}{@{}lrrp{0.34\textwidth}rr@{}}",
        r"\caption{Complete faithfulness evaluation. Comp. is the comprehensiveness drop and suff. is the sufficiency gap.}\label{tab:faithfulness-detail}\\",
        r"\toprule",
        r"ID & $\alpha$ & $k$ & Selected words & Comp. & Suff. \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{6}{l}{\tablename\ \thetable{} -- continued}\\",
        r"\toprule",
        r"ID & $\alpha$ & $k$ & Selected words & Comp. & Suff. \\",
        r"\midrule",
        r"\endhead",
        r"\midrule\multicolumn{6}{r}{Continued on next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for row in faithfulness:
        detail.append(
            f"{row['sentence_id']} & {float(row['fraction']):.1f} & "
            f"{row['selected_count']} & {latex_escape(row['selected_features'])} & "
            f"{float(row['comprehensiveness_drop']):.4f} & "
            f"{float(row['sufficiency_gap']):.4f} \\\\"
        )
    detail.extend([r"\end{longtable}", ""])
    (TABLE_DIR / "faithfulness_detail.tex").write_text(
        "\n".join(detail), encoding="utf-8"
    )


def write_word_values_table(words: list[dict[str, str]]) -> None:
    lines = [
        r"\begin{longtable}{@{}lrlrrr@{}}",
        r"\caption{Complete word-level SHAP results. Support is signed in the direction of the predicted class.}\label{tab:word-values}\\",
        r"\toprule",
        r"ID & Index & Word & $\Phi_w$ & $|\Phi_w|$ & Prediction support \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{6}{l}{\tablename\ \thetable{} -- continued}\\",
        r"\toprule",
        r"ID & Index & Word & $\Phi_w$ & $|\Phi_w|$ & Prediction support \\",
        r"\midrule",
        r"\endhead",
        r"\midrule\multicolumn{6}{r}{Continued on next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for row in words:
        lines.append(
            f"{row['sentence_id']} & {row['word_index']} & "
            f"{latex_escape(row['word'])} & {float(row['shap_value']):.6f} & "
            f"{float(row['absolute_shap_value']):.6f} & "
            f"{float(row['support_for_prediction']):.6f} \\\\"
        )
    lines.extend([r"\end{longtable}", ""])
    (TABLE_DIR / "word_values.tex").write_text("\n".join(lines), encoding="utf-8")


def write_macros(
    summary: list[dict[str, str]],
    faithfulness: list[dict[str, str]],
    config: dict[str, object],
) -> None:
    correct = sum(int(float(row["correct"])) for row in summary)
    max_residual = max(abs(float(row["additivity_residual"])) for row in summary)
    mean_comp = mean(float(row["comprehensiveness_drop"]) for row in faithfulness)
    mean_suff = mean(float(row["sufficiency_gap"]) for row in faithfulness)
    lines = [
        f"\\newcommand{{\\ResultSentenceCount}}{{{len(summary)}}}",
        f"\\newcommand{{\\ResultCorrectCount}}{{{correct}}}",
        f"\\newcommand{{\\ResultMaxResidual}}{{{max_residual:.2e}}}",
        f"\\newcommand{{\\ResultMeanComp}}{{{mean_comp:.4f}}}",
        f"\\newcommand{{\\ResultMeanSuff}}{{{mean_suff:.4f}}}",
        f"\\newcommand{{\\ResultRuntime}}{{{float(config['total_runtime_seconds']):.2f}}}",
        f"\\newcommand{{\\ResultModelEvaluations}}{{{int(config['total_model_text_evaluations']):,}}}",
        f"\\newcommand{{\\ResultForwardBatches}}{{{int(config['total_forward_batches']):,}}}",
    ]
    (TABLE_DIR / "result_macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary = read_csv(VALUES_DIR / "sentence_summary.csv")
    faithfulness = read_csv(EVALUATION_DIR / "faithfulness.csv")
    words = read_csv(VALUES_DIR / "word_shap_values.csv")
    config = json.loads((VALUES_DIR / "config.json").read_text(encoding="utf-8"))

    write_summary_table(summary)
    write_top_words_table(summary)
    write_sentence_table(summary)
    write_faithfulness_tables(faithfulness)
    write_word_values_table(words)
    write_macros(summary, faithfulness, config)
    print(
        f"Generated tables for {len(summary)} sentences, "
        f"{len(words)} words, and {len(faithfulness)} faithfulness rows."
    )


if __name__ == "__main__":
    main()
