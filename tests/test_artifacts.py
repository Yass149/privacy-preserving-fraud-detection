"""Integrity of the reported results.

These tests treat the dissertation's numbers as something that can regress. Two
separate incidents in this project's history motivate them: a dashboard panel
that read a superseded results file and displayed numbers from a model it was
not describing, and a set of reported figures that turned out not to be
reproducible because nothing was seeded.

Every test here is skipped cleanly if the artefacts are absent, so the suite
still runs on a fresh clone with no trained models.
"""

import json
import re

import pytest

pytestmark = pytest.mark.requires_artifacts


def load(repo_root, name):
    import sys
    sys.path.insert(0, str(repo_root))
    try:
        import paths as project_paths
    finally:
        sys.path.pop(0)
    path = project_paths.artefact(name)
    if not path.exists():
        pytest.skip(f"{name} not present")
    return json.loads(path.read_text())


def fraud(metrics):
    return metrics["classification_report"]["Fraud"]


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

def test_every_training_run_records_its_seed(repo_root):
    """An unseeded run cannot be reproduced, and the dissertation claims all
    results are reproducible. Any metrics file without a seed is either a stale
    pre-seeding artefact or evidence the seeding regressed."""
    unseeded = []
    for path in sorted(repo_root.glob("results/metrics/training_metrics*.json")):
        data = json.loads(path.read_text())
        if data.get("seed") is None:
            unseeded.append(path.name)
    assert not unseeded, (
        f"these runs record no seed and are therefore not reproducible: {unseeded}"
    )


# Metrics files that predate tiebreak logging. Each of these runs DID use the
# hash ordering (it is the loader default) but did not record the fact, so
# their ordering cannot be verified from the artefact alone. The list is pinned
# so it can shrink when a run is regenerated but never silently grow.
LEGACY_UNRECORDED_TIEBREAK = {"training_metrics_baseline.json"}


def test_elliptic_runs_use_the_decorrelated_row_ordering(repo_root):
    """Ordering tied rows by txId was shown to leak: a model with no
    minority-class signal still scored 0.49 recall under it, versus exactly
    0.000 under a decorrelated ordering. Any Elliptic run reporting
    tiebreak='txid' is contaminated and must not be cited.

    A missing tiebreak is treated as a failure too, not a pass. The earlier
    version of this test checked only for the string 'txid', so a file with no
    tiebreak field at all sailed through -- which is precisely what happened to
    the non-private baseline whose recall anchors the headline comparison.
    """
    contaminated, unrecorded = [], []
    for path in sorted(repo_root.glob("results/metrics/training_metrics*.json")):
        if "paysim" in path.name:
            continue  # PaySim uses window_size=1 and is structurally immune
        data = json.loads(path.read_text())
        tiebreak = data.get("tiebreak")
        if tiebreak == "txid":
            contaminated.append(path.name)
        elif tiebreak != "hash" and path.name not in LEGACY_UNRECORDED_TIEBREAK:
            unrecorded.append(f"{path.name} (tiebreak={tiebreak!r})")

    assert not contaminated, (
        f"these runs used the contaminated txId ordering: {contaminated}. "
        f"See thesis_source.md Section 3.6.6."
    )
    assert not unrecorded, (
        f"these runs do not record which row ordering they used, so the "
        f"ordering cannot be verified from the artefact: {unrecorded}. "
        f"Every training script now accepts and logs --tiebreak; regenerate "
        f"these runs, or add them to LEGACY_UNRECORDED_TIEBREAK with a reason."
    )


def test_legacy_tiebreak_allowlist_does_not_grow(repo_root):
    """The allowlist above exists to pin a known gap, not to absorb new ones.

    Every entry must still be a real file lacking a tiebreak record. When a
    legacy run is regenerated with a tiebreak, its entry should be deleted --
    this test fails if it is left behind, so the allowlist shrinks to empty
    rather than quietly outliving the problem.
    """
    for name in sorted(LEGACY_UNRECORDED_TIEBREAK):
        path = repo_root / "results" / "metrics" / name
        if not path.exists():
            pytest.skip(f"{name} not present")
        data = json.loads(path.read_text())
        assert data.get("tiebreak") is None, (
            f"{name} now records tiebreak={data.get('tiebreak')!r}, so it no "
            f"longer belongs in LEGACY_UNRECORDED_TIEBREAK -- remove it from "
            f"the allowlist in tests/test_artifacts.py."
        )


# --------------------------------------------------------------------------
# The core finding must survive
# --------------------------------------------------------------------------

def test_reweighting_collapses_and_oversampling_recovers(repo_root):
    """The dissertation's central empirical claim, asserted as a property
    rather than as exact numbers so it survives legitimate retraining.
    """
    baseline = load(repo_root, "training_metrics_baseline.json")
    collapsed = load(repo_root, "training_metrics.json")
    oversampled = load(repo_root, "training_metrics_oversampled.json")

    assert fraud(baseline)["recall"] > 0.6, "non-private baseline should detect most fraud"
    assert fraud(collapsed)["recall"] < 0.01, (
        "DP-SGD with loss reweighting should collapse to near-zero fraud recall; "
        "if this now passes, the headline finding has changed and Chapter 4 needs revisiting"
    )
    assert fraud(oversampled)["recall"] > 0.6, (
        "oversampling should restore fraud recall to roughly baseline parity"
    )


def test_reported_epsilon_falls_but_group_privacy_correction_is_documented(repo_root):
    """The per-record epsilon does fall under oversampling -- but that is NOT a
    privacy improvement, and this test guards the distinction rather than the
    original claim.

    An earlier version of this test asserted only that the oversampled run
    reaches a lower epsilon, and described it as "the STRONGER formal
    guarantee". That locked a conclusion into the suite instead of a
    measurement. Opacus accounts per record; oversampling duplicates each
    minority window five times, so removing one fraud transaction removes five
    records. Group privacy then gives roughly 5 x 1.828 = 9.1 for a duplicated
    transaction against 2.274 for the reweighted run -- about four times WEAKER
    for exactly the individuals the system protects.

    So the arithmetic fact is asserted, and separately the thesis is required to
    carry the correction. If someone removes that explanation, this fails.
    """
    collapsed = load(repo_root, "training_metrics.json")
    oversampled = load(repo_root, "training_metrics_oversampled.json")

    # The arithmetic fact, stated as what it is: a per-record quantity.
    assert oversampled["final_epsilon"] < collapsed["final_epsilon"], (
        "the oversampled run should report a lower PER-RECORD epsilon than the "
        "reweighted run, because its larger dataset lowers the sampling rate"
    )

    # The correction must be present in the write-up, or the number misleads.
    thesis = repo_root / "thesis" / "thesis_source.md"
    if not thesis.exists():
        pytest.skip("thesis_source.md not present")
    text = thesis.read_text()

    k = int(oversampled.get("minority_oversample_factor", 5))
    corrected = k * oversampled["final_epsilon"]
    required = {
        "group privacy": "group-privacy",
        "the corrected value": f"{corrected:.1f}",
        "the per-record caveat": "per record",
    }
    absent = {label: token for label, token in required.items() if token not in text}
    assert not absent, (
        "the thesis reports a lower epsilon under oversampling without the "
        f"group-privacy correction that makes it interpretable. Missing: {absent}. "
        f"A duplicated fraud transaction's effective budget is {k} x "
        f"{oversampled['final_epsilon']:.3f} = {corrected:.1f}, not "
        f"{oversampled['final_epsilon']:.3f}."
    )


def test_privacy_risk_is_redistributed_not_uniformly_reduced(repo_root):
    """The class-stratified membership inference finding: DP-SGD protects the
    class that was most exposed while increasing exposure for the class that
    was least exposed."""
    strat = load(repo_root, "mia_results_stratified.json")
    base = next(e for e in strat if e["tag"].startswith("baseline"))
    dp = next(e for e in strat if e["tag"].startswith("dp_sgd"))

    b_fraud = base["by_class"]["Fraud"]["attack_auc"]
    b_safe = base["by_class"]["Safe"]["attack_auc"]
    d_fraud = dp["by_class"]["Fraud"]["attack_auc"]
    d_safe = dp["by_class"]["Safe"]["attack_auc"]

    assert b_fraud > b_safe, "baseline should leak more on the minority (Fraud) class"
    assert d_fraud < b_fraud, "DP-SGD should reduce Fraud-class exposure"
    assert d_safe > b_safe, "DP-SGD should increase Safe-class exposure"


def test_attack_auc_values_are_in_range(repo_root):
    """An AUC outside [0, 1] means the attack computation is broken. Below 0.5
    on aggregate would mean the attack is anti-correlated, which would indicate
    a sign error rather than a privacy result."""
    for name in ("mia_results.json", "mia_results_stratified.json"):
        for entry in load(repo_root, name):
            agg = entry.get("aggregate", entry)
            assert 0.0 <= agg["attack_auc"] <= 1.0
            for stats in entry.get("by_class", {}).values():
                assert 0.0 <= stats["attack_auc"] <= 1.0


# --------------------------------------------------------------------------
# The thesis text must match the artefacts
# --------------------------------------------------------------------------

def test_thesis_headline_numbers_match_the_source_json(repo_root):
    """Guards the exact failure that already happened once: a displayed number
    drifting from the run it claims to describe. Each headline figure is
    recomputed from JSON and searched for in the dissertation text."""
    thesis = (repo_root / "thesis" / "thesis_source.md")
    if not thesis.exists():
        pytest.skip("thesis_source.md not present")
    text = thesis.read_text()

    checks = {
        "oversampled fraud recall":
            f"{fraud(load(repo_root, 'training_metrics_oversampled.json'))['recall']:.3f}",
        "oversampled epsilon":
            f"{load(repo_root, 'training_metrics_oversampled.json')['final_epsilon']:.3f}",
        "baseline fraud recall":
            f"{fraud(load(repo_root, 'training_metrics_baseline.json'))['recall']:.3f}",
        "GPU speedup":
            f"{load(repo_root, 'scalability_metrics.json')['speedup']:.2f}",
    }
    missing = {k: v for k, v in checks.items() if v not in text}
    assert not missing, (
        f"these values are in the results JSON but absent from thesis_source.md, "
        f"so the write-up has drifted from the artefacts: {missing}"
    )


def test_thesis_word_count_meets_the_requirement(repo_root):
    """The University template requires 10,000-15,000 words for Chapter 1
    through the end of Conclusions."""
    thesis = repo_root / "thesis" / "thesis_source.md"
    if not thesis.exists():
        pytest.skip("thesis_source.md not present")
    lines = thesis.read_text().split("\n")

    try:
        start = next(i for i, l in enumerate(lines) if l.startswith("# Chapter 1"))
        end = next(i for i, l in enumerate(lines) if l.startswith("# Chapter 7"))
    except StopIteration:
        pytest.skip("chapter markers not found")

    body = [l for l in lines[start:end]
            if not l.startswith("|") and not l.startswith("![") and not l.startswith(": ")]
    words = sum(len(l.split()) for l in body)
    assert 10_000 <= words <= 15_000, (
        f"Chapters 1-6 contain {words:,} words, outside the required 10,000-15,000"
    )

    stated = re.search(r"Total word count:\s*([\d,]+)", thesis.read_text())
    if stated:
        claimed = int(stated.group(1).replace(",", ""))
        assert abs(claimed - words) < 400, (
            f"the abstract claims {claimed:,} words but the document contains "
            f"{words:,} -- recount before submitting"
        )


def test_every_figure_referenced_by_the_thesis_exists(repo_root):
    """A missing figure renders as broken alt text in the PDF."""
    thesis = repo_root / "thesis" / "thesis_source.md"
    if not thesis.exists():
        pytest.skip("thesis_source.md not present")
    referenced = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", thesis.read_text())
    missing = [f for f in referenced if not (repo_root / "thesis" / f).exists()]
    assert not missing, f"thesis references figures that do not exist: {missing}"


# --------------------------------------------------------------------------
# Demo cases powering the dashboard
# --------------------------------------------------------------------------

def test_demo_cases_are_internally_consistent(repo_root):
    """Each exported case declares an outcome (true_positive, false_negative,
    ...). That label must agree with its own true_label and predicted
    probability, or the dashboard's case browser is mislabelling real evidence.
    """
    demo = load(repo_root, "demo_cases.json")
    threshold = demo["decision_threshold"]
    expected = {
        "true_positive": (1, 1), "false_negative": (1, 0),
        "false_positive": (0, 1), "true_negative": (0, 0),
    }
    for case in demo["cases"]:
        predicted = 1 if case["fraud_probability"] >= threshold else 0
        assert (case["true_label"], predicted) == expected[case["outcome"]], (
            f"case {case['id']} is labelled {case['outcome']} but has "
            f"true_label={case['true_label']} and predicted={predicted}"
        )


def test_demo_case_counts_reproduce_the_reported_metrics(repo_root):
    """The exported confusion counts must reproduce the recall and precision
    reported for the same checkpoint -- otherwise the demo and the dissertation
    are describing different models."""
    demo = load(repo_root, "demo_cases.json")
    metrics = load(repo_root, "training_metrics_oversampled.json")
    counts = demo["outcome_counts_full_test_set"]

    tp, fn, fp = counts["true_positive"], counts["false_negative"], counts["false_positive"]
    assert tp + fn + fp + counts["true_negative"] == demo["n_test_windows_total"]

    assert abs(tp / (tp + fn) - fraud(metrics)["recall"]) < 0.005, "recall mismatch"
    assert abs(tp / (tp + fp) - fraud(metrics)["precision"]) < 0.005, "precision mismatch"


def test_demo_cases_carry_no_raw_dataset_values(repo_root):
    """The export is deliberately limited to derived values so no Elliptic
    feature rows are redistributed. This asserts that boundary holds."""
    demo = load(repo_root, "demo_cases.json")
    for case in demo["cases"]:
        assert set(case.keys()) <= {
            "id", "outcome", "true_label", "fraud_probability", "top_attributions"
        }, f"case {case['id']} carries unexpected keys: {sorted(case.keys())}"
        for attr in case["top_attributions"]:
            assert set(attr.keys()) == {"feature_index", "attribution"}


# --------------------------------------------------------------------------
# Cell-by-cell table verification
#
# The substring guard above can detect a MISSING number but never a WRONG one,
# which is exactly how every cell of the PaySim results table drifted from its
# source JSON while the suite stayed green. These tests parse the actual table
# rows and compare each cell positionally.
# --------------------------------------------------------------------------

def _parse_markdown_table(text, header_fragment):
    """Return the data rows of the first markdown table whose header contains
    `header_fragment`, as lists of stripped cell strings."""
    rows, in_table = [], False
    for line in text.split("\n"):
        stripped = line.strip()
        if not in_table:
            if stripped.startswith("|") and header_fragment in stripped:
                in_table = True
            continue
        if not stripped.startswith("|"):
            break
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            continue          # the |---|---| separator row
        rows.append(cells)
    return rows


def _cell_value(cell):
    """Parse a table cell to a float. Returns None for 'not applicable' dashes.
    Tolerates trailing annotations such as '0.000 (degenerate)'."""
    cell = cell.replace("**", "").strip()
    if cell in {"—", "-", "--", "n/a", ""}:
        return None
    token = cell.split()[0]
    try:
        return float(token)
    except ValueError:
        return None


def _assert_cell(label, column, actual, claimed_cell, tol=0.0006):
    claimed = _cell_value(claimed_cell)
    assert claimed is not None, f"{label} / {column}: could not parse cell {claimed_cell!r}"
    assert abs(claimed - actual) <= tol, (
        f"{label} / {column}: thesis states {claimed}, source JSON gives {actual:.4f} "
        f"(difference {abs(claimed - actual):.4f}). The write-up has drifted from the artefacts."
    )


def test_elliptic_results_table_matches_json_cell_by_cell(repo_root):
    """Table 4.2 — every cell against its run's metrics file."""
    thesis = repo_root / "thesis" / "thesis_source.md"
    if not thesis.exists():
        pytest.skip("thesis_source.md not present")
    rows = _parse_markdown_table(thesis.read_text(), "Fraud Recall")
    assert rows, "could not locate the Elliptic results table"

    sources = {
        "Non-private baseline": "training_metrics_baseline.json",
        "DP-SGD, 10x loss reweighting": "training_metrics.json",
        "DP-SGD, 5x minority oversampling": "training_metrics_oversampled.json",
    }
    checked = 0
    for row in rows:
        label = row[0].replace("**", "").strip()
        if label not in sources:
            continue
        m = load(repo_root, sources[label])
        f, macro = fraud(m), m["classification_report"]["macro avg"]
        # Columns: Configuration | Recall | Precision | Fraud F1 | Macro F1 | eps | delta
        _assert_cell(label, "Fraud Recall", f["recall"], row[1])
        _assert_cell(label, "Fraud Precision", f["precision"], row[2])
        _assert_cell(label, "Fraud F1", f["f1-score"], row[3])
        _assert_cell(label, "Macro F1", macro["f1-score"], row[4])
        if m.get("final_epsilon") is not None and _cell_value(row[5]) is not None:
            _assert_cell(label, "epsilon", m["final_epsilon"], row[5])
        checked += 1
    assert checked == 3, f"expected to check 3 Elliptic rows, checked {checked}"


def test_paysim_results_table_matches_json_cell_by_cell(repo_root):
    """Table 4.5 — every cell against its run's metrics file.

    This is the test that was missing. Every cell of this table drifted from
    its source and the suite stayed green, because the only prior guard was a
    substring search over four Elliptic values.
    """
    thesis = repo_root / "thesis" / "thesis_source.md"
    if not thesis.exists():
        pytest.skip("thesis_source.md not present")
    rows = _parse_markdown_table(thesis.read_text(), "PaySim Configuration")
    assert rows, "could not locate the PaySim results table"

    sources = {
        "Non-private baseline": "training_metrics_paysim_baseline_v2.json",
        "DP-SGD, naive (10x reweight)": "training_metrics_paysim_dp_os1_v1.json",
        "DP-SGD, oversampled (200x)": "training_metrics_paysim_dp_os200_v1.json",
    }
    checked = 0
    for row in rows:
        label = row[0].replace("**", "").strip()
        if label not in sources:
            continue
        m = load(repo_root, sources[label])
        f, macro = fraud(m), m["classification_report"]["macro avg"]
        # Columns: PaySim Configuration | Recall | Precision | Macro F1 | eps
        _assert_cell(label, "Fraud Recall", f["recall"], row[1])
        _assert_cell(label, "Fraud Precision", f["precision"], row[2])
        _assert_cell(label, "Macro F1", macro["f1-score"], row[3])
        if m.get("final_epsilon") is not None and _cell_value(row[4]) is not None:
            _assert_cell(label, "epsilon", m["final_epsilon"], row[4])
        checked += 1
    assert checked == 3, f"expected to check 3 PaySim rows, checked {checked}"


def test_epsilon_sweep_table_and_figure_plot_the_same_points(repo_root):
    """The sweep table in Chapter 4.4 and fig_2b must cover the same runs.

    They came apart once: training_metrics_nm1.1.json was removed as a stale
    duplicate (the nm=1.1 run IS the default oversampled configuration), and
    app.py was updated to inject the real run as the centre point while
    visualize.py was not. The table then showed six points and the generated
    figure five. Both consumers now assemble the same set; this asserts every
    row of the table resolves to a real source file.
    """
    thesis = repo_root / "thesis" / "thesis_source.md"
    if not thesis.exists():
        pytest.skip("thesis_source.md not present")
    rows = _parse_markdown_table(thesis.read_text(), "noise_multiplier")
    assert rows, "could not locate the noise_multiplier sweep table"

    table_eps = sorted(_cell_value(r[1]) for r in rows if _cell_value(r[1]) is not None)
    assert len(table_eps) == 6, f"expected 6 sweep rows, found {len(table_eps)}"

    # Reproduce exactly what visualize.py and app.py assemble.
    sources = sorted(repo_root.glob("results/metrics/training_metrics_nm*.json"))
    centre = repo_root / "results" / "metrics" / "training_metrics_oversampled.json"
    if centre.exists():
        sources.append(centre)
    plotted_eps = sorted(
        json.loads(p.read_text())["final_epsilon"] for p in sources
    )

    assert len(plotted_eps) == len(table_eps), (
        f"the sweep table has {len(table_eps)} rows but the figure would plot "
        f"{len(plotted_eps)} points. If an nm file was removed, the consumers "
        f"in visualize.py and app.py must inject the replacement."
    )
    for claimed, actual in zip(table_eps, plotted_eps):
        assert abs(claimed - actual) <= 0.001, (
            f"sweep table epsilon {claimed} has no matching source run "
            f"(nearest is {actual:.4f})"
        )


def test_threshold_independent_metrics_are_valid_when_present(repo_root):
    """Average precision and ROC-AUC are recorded from the next training run
    onward. Runs predating that change simply lack the fields and are skipped.

    The examiner's criticism these answer: recall, precision and F1 all describe
    a single 0.5 cutoff, so comparing the baseline (recall 0.727, precision
    0.323) against the DP model (0.743, 0.232) compares two arbitrary points on
    two different curves and cannot support a claim of 'parity'. Average
    precision summarises the whole curve.
    """
    checked = 0
    for path in sorted(repo_root.glob("results/metrics/training_metrics*.json")):
        data = json.loads(path.read_text())
        for field in ("average_precision", "roc_auc"):
            if field not in data:
                continue
            value = data[field]
            assert isinstance(value, (int, float)), f"{path.name}: {field} is not numeric"
            assert 0.0 <= value <= 1.0, f"{path.name}: {field}={value} outside [0, 1]"
            checked += 1
        # Average precision below the class prevalence would mean the model
        # ranks worse than random, which indicates a sign or ordering error
        # rather than a weak model.
        if "average_precision" in data:
            report = data.get("classification_report", {})
            fraud_support = report.get("Fraud", {}).get("support")
            total = sum(report.get(c, {}).get("support", 0) for c in ("Safe", "Fraud"))
            if fraud_support and total:
                prevalence = fraud_support / total
                assert data["average_precision"] >= prevalence * 0.5, (
                    f"{path.name}: average precision {data['average_precision']:.4f} is far "
                    f"below the class prevalence {prevalence:.4f}, suggesting the scores are "
                    f"inverted or misaligned with the labels rather than merely weak"
                )
    if checked == 0:
        pytest.skip("no run has recorded threshold-independent metrics yet")


def test_mia_values_appear_in_the_thesis_correctly_rounded(repo_root):
    """The membership-inference figures are quoted in prose and in a table
    across several chapters, so a table-cell check does not cover them.

    This caught a real error: the Safe-class DP value is 0.68647, which rounds
    to 0.686, but the thesis carried 0.687 in five places -- the rounding had
    been done by hand rather than from the source.
    """
    thesis = repo_root / "thesis" / "thesis_source.md"
    if not thesis.exists():
        pytest.skip("thesis_source.md not present")
    text = thesis.read_text()
    strat = load(repo_root, "mia_results_stratified.json")

    wrong = []
    for entry in strat:
        tag = "baseline" if entry["tag"].startswith("baseline") else "DP-SGD"
        for cls, stats in entry["by_class"].items():
            correct = f"{stats['attack_auc']:.3f}"
            if correct not in text:
                wrong.append(f"{tag}/{cls}: expected {correct} "
                             f"(exact {stats['attack_auc']:.5f}) — not found in the thesis")
    assert not wrong, (
        "membership-inference values in the thesis do not match the source "
        "rounded to three decimals:\n  " + "\n  ".join(wrong)
    )


def test_latex_thesis_is_in_sync_with_its_markdown_source(repo_root):
    """thesis.tex is derived from thesis_source.md, and used to drift.

    The LaTeX file was maintained by hand-running pandoc and hand-patching the
    result. That is invisible drift waiting to happen: a correction made in the
    Markdown does not reach the .tex, and the .tex is what compiles into the
    submitted document. It had already happened once -- the .tex still claimed
    oversampling reached a "tighter privacy budget" after the source had been
    corrected to say the opposite.

    scripts/build_thesis.py now performs the conversion reproducibly, and this
    asserts the committed .tex is what that script produces from the current
    source. Skips where pandoc is unavailable (the CI image has no pandoc), so
    the guard is advisory there rather than absent everywhere.
    """
    import shutil
    import subprocess
    import sys

    tex = repo_root / "thesis" / "latex" / "thesis.tex"
    builder = repo_root / "scripts" / "build_thesis.py"
    if not tex.exists() or not builder.exists():
        pytest.skip("latex/thesis.tex or its build script is not present")
    if shutil.which("pandoc") is None:
        pytest.skip("pandoc is not installed on this machine")

    sys.path.insert(0, str(builder.parent))
    try:
        import build_thesis
    finally:
        sys.path.pop(0)

    expected = build_thesis.build(
        (repo_root / "thesis" / "thesis_source.md").read_text(),
        (repo_root / "thesis" / "latex" / "preamble.tex").read_text(),
    )
    actual = tex.read_text()
    if expected == actual:
        return

    import difflib
    diff = list(difflib.unified_diff(
        actual.split("\n"), expected.split("\n"),
        "committed thesis.tex", "rebuilt from thesis_source.md", lineterm=""))
    pytest.fail(
        "latex/thesis.tex does not match what thesis_source.md produces. Run "
        "`python scripts/build_thesis.py --docx` to regenerate it.\n"
        + "\n".join(diff[:40])
    )


def test_front_matter_headings_match_the_official_templates_exact_styling(repo_root):
    """Declaration, Abstract and Acknowledgements are NOT styled identically.

    Ground truth is the official template's own source files, not a rendered
    PDF and not "consistency" among the three: chapters/00_i_abstract.tex and
    00_ii_acknowledgments.tex both use "\\chapter*{\\center \\Large ...}", but
    the master file's own Declaration (CS_report_template.tex:83) uses
    "\\chapter*{\\Large Declaration}" -- \\Large only, left-aligned, no
    \\center. An earlier version of this build centred Declaration too,
    reasoning the asymmetry looked like an oversight in the department's own
    file; a direct comparison against the rendered official PDF showed that
    reasoning was wrong. Don't "fix" a heading style the template
    deliberately uses.
    """
    tex_path = repo_root / "thesis" / "latex" / "thesis.tex"
    if not tex_path.exists():
        pytest.skip("thesis/latex/thesis.tex not present")
    tex = tex_path.read_text()
    assert r"\chapter*{\Large Declaration}" in tex
    assert r"\chapter*{\center\Large Declaration}" not in tex
    assert r"\chapter*{\center\Large Abstract}" in tex
    assert r"\chapter*{\center\Large Acknowledgements}" in tex


def test_the_disproven_privacy_claim_has_not_come_back(repo_root):
    """A specific claim this project made, and retracted.

    The thesis reported that minority oversampling reached a *tighter* privacy
    budget than loss reweighting -- better utility and better privacy at once.
    It does report a lower epsilon, but only because Opacus charges per record
    and oversampling makes five records out of each fraud transaction. Corrected
    for group privacy the guarantee is about four times weaker.

    That claim was written into six documents and a test. This makes sure it
    does not creep back in through a doc regenerated from an old template or a
    passage restored from an earlier draft.
    """
    # Only phrases that can *only* be the assertion. Words like "tighter" have a
    # legitimate use elsewhere (a smaller epsilon budget is a stricter one), and
    # the retraction itself quotes the old wording, so a bare keyword scan would
    # fire on correct text.
    banned = (
        "better utility and stronger privacy",
        "at a tighter privacy budget",
        "at a *tighter* privacy budget",
        "costs no extra privacy",
        "tighter \u03b5 than",
        "tighter epsilon",
    )
    offenders = []
    for path in sorted(sorted(repo_root.glob("docs/*.md")) + list(repo_root.glob("*.md")) + list(repo_root.glob("thesis/*.md"))) + sorted(repo_root.glob("thesis/latex/*.tex")):
        if path.name.startswith("_") or ".remember" in str(path):
            continue
        lowered = path.read_text().lower()
        for phrase in banned:
            if phrase in lowered:
                offenders.append(f"{path.name}: {phrase!r}")
    assert not offenders, (
        "the retracted 'oversampling buys better privacy' claim is back:\n  "
        + "\n  ".join(offenders)
        + "\nEpsilon falls per record because the dataset grew; per transaction "
          "it rises by the oversampling factor."
    )

    # A phrase list only catches wording someone happened to reuse. The claim can
    # also come back reworded, so guard it structurally instead: any document
    # that puts the two epsilons side by side is making the comparison, and a
    # comparison without the correction is the misleading one.
    oversampled = load(repo_root, "training_metrics_oversampled.json")
    collapsed = load(repo_root, "training_metrics.json")
    low, high = f"{oversampled['final_epsilon']:.3f}", f"{collapsed['final_epsilon']:.3f}"
    k = int(oversampled.get("minority_oversample_factor", 5))
    corrected = f"{k * oversampled['final_epsilon']:.1f}"

    uncorrected = []
    for path in sorted(sorted(repo_root.glob("docs/*.md")) + list(repo_root.glob("*.md")) + list(repo_root.glob("thesis/*.md"))) + sorted(repo_root.glob("thesis/latex/*.tex")):
        if path.name.startswith("_") or path.name == "MEMORY.md":
            continue
        text = path.read_text()
        if low in text and high in text and corrected not in text:
            uncorrected.append(path.name)
    assert not uncorrected, (
        f"these documents compare \u03b5 = {low} against {high} without the "
        f"group-privacy correction ({corrected}), which is the comparison that "
        f"misleads: {uncorrected}"
    )


def test_mia_transitions_quoted_anywhere_use_the_right_dp_value(repo_root):
    """The same rounding error, guarded across every document rather than one.

    The membership-inference results are usually written as a transition --
    "0.608 -> 0.686" -- and that shape is what makes them checkable. Whenever a
    document quotes a baseline attack AUC on the left of an arrow, the value on
    the right has to be that same class's DP figure.

    The narrower version of this test only read thesis_source.md, and the README
    was carrying 0.687 where the source says 0.68647. One document being right
    is not the same as the claim being right.
    """
    strat = load(repo_root, "mia_results_stratified.json")
    expected = {}
    for entry in strat:
        which = "baseline" if entry["tag"].startswith("baseline") else "dp"
        for cls, stats in entry["by_class"].items():
            expected.setdefault(cls, {})[which] = stats["attack_auc"]

    # baseline value (3dp) -> the DP value that must follow it
    transitions = {
        f"{v['baseline']:.3f}": (cls, f"{v['dp']:.3f}", v["dp"])
        for cls, v in expected.items()
        if "baseline" in v and "dp" in v
    }

    # Two complementary checks. The first catches a transition whose right-hand
    # side drifted. The second catches a value quoted next to "AUC" that belongs
    # to no recorded run at all -- which is what PROJECT_EXPLAINED.md was doing
    # with "AUC 0.583 / AUC 0.848", a stale pair from a superseded run where
    # *both* numbers were wrong, so the left-hand side matched nothing to check.
    #
    # Known limit: a prose transition with both sides stale and no nearby "AUC"
    # token is not caught. Proximity heuristics were tried and produced false
    # positives on recall transitions sharing a paragraph with the audit prose,
    # which is worse than the gap.
    allowed = set()
    for source in ("mia_results_stratified.json", "mia_results.json",
               "mia_results_collapsed_model_excluded.json"):
        path = repo_root / "results" / "attacks" / source
        if not path.exists():
            continue
        for entry in json.loads(path.read_text()):
            if "aggregate" in entry:
                allowed.add(f"{entry['aggregate']['attack_auc']:.3f}")
            elif "attack_auc" in entry:      # round-1 file is flat
                allowed.add(f"{entry['attack_auc']:.3f}")
            for stats in entry.get("by_class", {}).values():
                allowed.add(f"{stats['attack_auc']:.3f}")

    arrow = re.compile(r"(0\.\d{3})\s*(?:->|→|--+>)\s*(0\.\d{3})")
    labelled_auc = re.compile(r"(?<!ROC-)\bAUC\b[^0-9\n]{0,12}(0\.\d{3})(?!\d)", re.IGNORECASE)
    wrong = []
    for path in (sorted(sorted(repo_root.glob("docs/*.md")) + list(repo_root.glob("*.md")) + list(repo_root.glob("thesis/*.md"))) + sorted(repo_root.glob("thesis/latex/*.tex"))
                 + sorted(list(repo_root.glob("*.py")) + list(repo_root.glob("scripts/**/*.py")) + list(repo_root.glob("src/*.py")))):
        if path.name.startswith("_") or path.name == "MEMORY.md":
            continue
        text = path.read_text()
        for left, right in arrow.findall(text):
            if left in transitions:
                cls, correct, exact = transitions[left]
                if right != correct:
                    wrong.append(
                        f"{path.name}: {cls} class written as {left} -> {right}, "
                        f"but the DP value is {exact:.5f}, i.e. {correct}"
                    )
        for value in labelled_auc.findall(text):
            if value not in allowed and value != "0.500":
                wrong.append(
                    f"{path.name}: quotes \"AUC {value}\", which matches no recorded "
                    f"attack AUC {sorted(allowed)}"
                )
    assert not wrong, "membership-inference transitions are misquoted:\n  " + "\n  ".join(wrong)


def _thesis(repo_root):
    path = repo_root / "thesis" / "thesis_source.md"
    if not path.exists():
        pytest.skip("thesis_source.md not present")
    return path.read_text()


def test_control_1_table_matches_json_cell_by_cell(repo_root):
    """Control 1 isolates the class weight: same seed, same clipping, weight on
    or off. It is the load-bearing evidence for Section 5.1's mechanism, so its
    cells are checked against the runs rather than trusted."""
    text = _thesis(repo_root)
    rows = _parse_markdown_table(text, "Final train loss")
    assert len(rows) == 2, f"expected 2 rows in the Control 1 table, found {len(rows)}"

    for row, name in zip(rows, ["ctrl_dp_unweighted", "ctrl_rw_mgn1.0"]):
        m = load(repo_root, f"training_metrics_{name}.json")
        label = row[0]
        _assert_cell(label, "final train loss", m["epoch_history"][-1]["train_loss"], row[2], tol=0.0006)
        _assert_cell(label, "fraud recall", fraud(m)["recall"], row[3])
        _assert_cell(label, "average precision", m["average_precision"], row[4])
        _assert_cell(label, "ROC-AUC", m["roc_auc"], row[5])


def test_control_3_clipping_sweep_table_matches_json_cell_by_cell(repo_root):
    """Control 3 is the most direct confirmation in the thesis: recall against
    the clipping norm, everything else fixed. Its shape is the argument, so a
    single wrong cell would misrepresent it."""
    text = _thesis(repo_root)
    rows = _parse_markdown_table(text, "Clipping norm")
    assert len(rows) == 3, f"expected 3 rows in the clipping sweep table, found {len(rows)}"

    for row, name in zip(rows, ["ctrl_rw_mgn1.0", "ctrl_rw_mgn10.0", "ctrl_rw_mgn100.0"]):
        m = load(repo_root, f"training_metrics_{name}.json")
        report = m["classification_report"]
        label = f"max_grad_norm {row[0]}"
        _assert_cell(label, "max_grad_norm", m["max_grad_norm"], row[0], tol=0.0006)
        _assert_cell(label, "fraud recall", report["Fraud"]["recall"], row[1])
        _assert_cell(label, "fraud precision", report["Fraud"]["precision"], row[2])
        _assert_cell(label, "macro F1", report["macro avg"]["f1-score"], row[3])
        _assert_cell(label, "average precision", m["average_precision"], row[4])
        _assert_cell(label, "ROC-AUC", m["roc_auc"], row[5])
        _assert_cell(label, "safe recall", report["Safe"]["recall"], row[6])


def test_control_3_recall_rises_monotonically_with_the_clipping_norm(repo_root):
    """The direction is the finding, not the individual numbers.

    If loosening the clip did not restore the class weight's effect, the
    mechanism in Section 5.1 would be wrong -- clipping would not be what
    cancels the weighting. This asserts the direction independently of how the
    thesis happens to describe it.
    """
    runs = [load(repo_root, f"training_metrics_ctrl_rw_mgn{n}.json") for n in ("1.0", "10.0", "100.0")]
    recalls = [fraud(m)["recall"] for m in runs]
    assert recalls == sorted(recalls) and recalls[0] < recalls[-1], (
        f"fraud recall must rise as max_grad_norm is loosened, got {recalls}. If this "
        f"direction ever reverses, Section 5.1's mechanism needs rewriting, not the test."
    )

    # And the counterpoint the discussion depends on: it is not a better model,
    # only a differently biased one.
    aps = [m["average_precision"] for m in runs]
    assert aps[0] > aps[-1], (
        f"average precision should fall as the clip is loosened, got {aps}; Section 5.1 "
        f"argues the recovered recall is an operating-point shift, not restored skill"
    )


def test_collapsed_models_still_rank_far_above_chance(repo_root):
    """The refinement the controls forced: a recall of 0.000 is not a model that
    learned nothing.

    Every reweighted DP-SGD run predicts zero fraud, but their average precision
    sits near the non-private baseline's and far above the 0.065 prevalence
    floor. The collapse is in where the decision boundary falls, not in what the
    network learned. Section 5.1 says so, and this keeps that claim tied to the
    artefacts.
    """
    prevalence = 1081 / 16666
    collapsed = [load(repo_root, f"training_metrics_{n}.json")
                 for n in ("ctrl_rw_mgn1.0", "seed1_reweighted", "seed2_reweighted")]

    for m in collapsed:
        assert fraud(m)["recall"] == 0.0, "this run is not one of the collapsed ones"
        assert m["average_precision"] > 4 * prevalence, (
            f"a collapsed run scores AP {m['average_precision']:.3f}, barely above the "
            f"{prevalence:.3f} prevalence floor -- the 'ranks well, decides badly' "
            f"reading in Section 5.1 would not hold"
        )


def test_every_table_and_figure_is_captioned_and_referenced(repo_root):
    """Presentation marks the write-up was silently losing.

    Ten tables carried no caption and no number at all, while the prose referred
    to "Table 4.2" and "Table 4.5" -- labels that existed nowhere in the output.
    Four of the five figures were never mentioned in the text.

    LaTeX numbers both counters itself (\\numberwithin ... {section} in the
    preamble), so what has to be checked here is that every table has a caption
    line and that every number the counters will produce is pointed at from the
    prose.
    """
    text = _thesis(repo_root)
    lines = text.split("\n")

    tables, figures = 0, len(re.findall(r"^!\[", text, re.M))
    captions = 0
    for i, line in enumerate(lines):
        nxt = lines[i + 1].replace("|", "").strip() if i + 1 < len(lines) else ""
        if line.startswith("|") and "-" in nxt and set(nxt) <= set("-: "):
            tables += 1
            # the caption follows the table body, after a blank line
            j = i + 1
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].startswith(": "):
                captions += 1
    assert tables and captions == tables, (
        f"{tables - captions} of {tables} tables have no caption. Pandoc turns a "
        f'": Caption text" line after the table into \\caption{{}}; without it the '
        f"table appears unnumbered and cannot be referenced."
    )

    # Hand-written "Figure N:" prefixes inside the alt text would double up with
    # LaTeX's own counter, producing "Figure 4.1: Figure 1: ...".
    assert not re.search(r"^!\[(Figure|Table) \d", text, re.M), (
        "a figure caption carries its own hand-written number; LaTeX numbers "
        "figures itself, so this renders as 'Figure 4.1: Figure 1: ...'"
    )

    unreferenced = []
    for n in range(1, tables + 1):
        if not re.search(rf"Tables? 4\.{n}\b(?!\d)", text):
            unreferenced.append(f"Table 4.{n}")
    for n in range(1, figures + 1):
        if not (re.search(rf"Figures? 4\.{n}\b", text)
                or re.search(rf"Figures 4\.\d+ and 4\.{n}\b", text)):
            unreferenced.append(f"Figure 4.{n}")
    assert not unreferenced, (
        f"these are captioned but never pointed at from the prose: {unreferenced}. "
        f"An unreferenced float reads as decoration."
    )


def test_the_latex_contains_no_character_pdflatex_cannot_typeset(repo_root):
    """The Overleaf build was failing on four characters.

    thesis.tex carried U+2212 (MINUS SIGN) and U+2248 (ALMOST EQUAL TO), both
    hard LaTeX errors under T1/inputenc -- introduced by the group-privacy
    passages, which write things like "(k-1)e" and "~ 9.1". A local pdflatex run
    was the only thing that revealed it, because the Markdown and the .docx both
    render them fine.

    scripts/build_thesis.py now maps them and raises on anything unmapped. This
    checks the committed artefact directly, so the guard holds even if someone
    edits the .tex by hand.
    """
    tex = repo_root / "thesis" / "latex" / "thesis.tex"
    if not tex.exists():
        pytest.skip("latex/thesis.tex not present")

    import unicodedata
    builder = repo_root / "scripts" / "build_thesis.py"
    if not builder.exists():
        pytest.skip("build script not present")
    import sys
    sys.path.insert(0, str(builder.parent))
    try:
        import build_thesis
    finally:
        sys.path.pop(0)

    offenders = {c for c in tex.read_text() if ord(c) > 127} - build_thesis.TEX_SAFE_NON_ASCII
    assert not offenders, (
        "latex/thesis.tex contains characters with no tested pdfTeX mapping: "
        + ", ".join(f"U+{ord(c):04X} ({unicodedata.name(c, '?')})" for c in sorted(offenders))
        + ". Rebuild with scripts/build_thesis.py, which maps them."
    )


def test_every_citation_has_a_reference_and_every_reference_is_cited(repo_root):
    """Nothing was wrong here -- 24 in-text citations against 24 entries -- and
    this keeps it that way through further editing. A dangling citation or an
    orphaned reference entry is one of the cheapest things to lose marks on."""
    text = _thesis(repo_root)
    if "# References" not in text:
        pytest.skip("no reference section")
    body, rest = text.split("# References", 1)
    refs = rest.split("# Appendices")[0]

    cited = set()
    for pattern in (r"([A-Z][A-Za-z\-']+)(?:\s+(?:and|&)\s+[A-Z][A-Za-z\-']+)?(?:\s+et al\.?)?\s*\((\d{4}[a-z]?)\)",
                    r"\(([A-Z][A-Za-z\-']+)(?:\s+et al\.?)?,\s*(\d{4}[a-z]?)\)"):
        cited |= {(m.group(1), m.group(2)) for m in re.finditer(pattern, body)}

    listed = set()
    for line in refs.split("\n"):
        surname = re.match(r"([A-Z][A-Za-z\-']+)[,.]", line.strip())
        year = re.search(r"\((\d{4}[a-z]?)\)", line)
        if surname and year:
            listed.add((surname.group(1), year.group(1)))

    missing = sorted(cited - listed)
    orphaned = sorted(listed - cited)
    assert not missing, f"cited in the text but absent from the reference list: {missing}"
    assert not orphaned, f"listed in the references but never cited: {orphaned}"


def test_no_figure_asserts_an_uncited_external_standard(repo_root):
    """A figure once carried a horizontal line labelled "Strict Enterprise Limit
    (epsilon <= 3.0)" -- a hardcoded axhline with no citation and no such
    standard behind it. It manufactured authority the work does not have, on the
    figure a reader looks at first.

    Any threshold drawn on a figure has to come from the data or from a cited
    source. This keeps invented benchmarks out of the generators.
    """
    invented = ("enterprise limit", "industry standard", "regulatory limit",
                "acceptable threshold", "compliance limit", "recommended maximum")
    offenders = []
    for path in sorted(repo_root.glob("scripts/figures/visualize*.py")) + [repo_root / "app.py"]:
        if not path.exists():
            continue
        lowered = path.read_text().lower()
        for phrase in invented:
            if phrase in lowered:
                offenders.append(f"{path.name}: {phrase!r}")
    assert not offenders, (
        "a figure or panel asserts an external standard with no citation:\n  "
        + "\n  ".join(offenders)
        + "\nDraw thresholds from the data, or cite the source in the caption."
    )


def test_dashboard_does_not_repeat_the_retracted_privacy_claim(repo_root):
    """The structural epsilon guard globs *.md and latex/*.tex, so app.py was
    outside it -- and the PaySim panel was still captioned "ε improves too",
    which is the retracted claim in the one place a panel member would see it.

    Any dashboard text putting the two epsilons together has to carry the
    correction, exactly as the written documents must.
    """
    app = repo_root / "app.py"
    if not app.exists():
        pytest.skip("app.py not present")
    text = app.read_text()

    banned = ("improves too", "better utility and stronger privacy",
              "costs no extra privacy", "tighter budget")
    offenders = [phrase for phrase in banned if phrase in text.lower()]
    assert not offenders, (
        f"the dashboard states the retracted privacy claim: {offenders}. Epsilon falls "
        f"per record because the dataset grew; per transaction it rises by the "
        f"oversampling factor."
    )

    # And the correction has to actually be present where epsilon is discussed.
    for token in ("group-privacy", "per record"):
        assert token in text.lower(), (
            f"the dashboard reports epsilon without the {token!r} framing that makes "
            f"it interpretable"
        )


# --------------------------------------------------------------------------
# Repository layout
# --------------------------------------------------------------------------

def test_paths_module_resolves_every_artefact_family(repo_root):
    """paths.py is the single place that knows where results live. If a
    directory moves and this is not updated, the dashboard silently shows
    "not available yet" instead of failing, which is the worst outcome: the
    panel looks fine and the data is gone.
    """
    import sys
    sys.path.insert(0, str(repo_root))
    try:
        import paths as project_paths
    finally:
        sys.path.pop(0)

    # FIGURES and LATEX live under thesis/, and DOCS is docs/. Both are kept
    # locally but not tracked: the dissertation is submitted separately, and
    # docs/ holds working notes that are not part of the deliverable. Assert
    # those only when the directory is actually present.
    always = (project_paths.METRICS, project_paths.ATTACKS,
              project_paths.CHECKPOINTS, project_paths.SRC, project_paths.HPC)
    optional = (project_paths.FIGURES, project_paths.LATEX, project_paths.DOCS)
    for directory in always:
        assert directory.is_dir(), f"paths.py points at a missing directory: {directory}"
    for directory in optional:
        if directory.exists():
            assert directory.is_dir(), f"paths.py points at a non-directory: {directory}"

    # One real file per naming convention, so the prefix table stays honest.
    for name in ("training_metrics_oversampled.json", "mia_results_stratified.json",
                 "fraud_detector_oversampled.pth", "demo_cases.json"):
        resolved = project_paths.artefact(name)
        assert resolved.exists(), (
            f"paths.artefact({name!r}) resolved to {resolved}, which does not exist -- "
            f"the prefix table in paths.py is out of date with the layout"
        )

    # An already-resolved path must survive unchanged, since callers pass both.
    already = project_paths.METRICS / "training_metrics.json"
    assert project_paths.artefact(already) == already


def test_no_source_file_references_a_path_that_moved(repo_root):
    """The reorganisation moved roughly eighty files. A missed reference does
    not raise at import time -- it fails later, when a panel renders empty or a
    figure regenerates from nothing. This checks the references directly.

    Only paths containing a directory separator are checked. Bare filenames are
    resolved through paths.artefact at runtime and are covered by the test
    above; hpc/ is excluded because those scripts run on the cluster from their
    own working directory and name files with no repository layout in mind.
    """
    sources = (list(repo_root.glob("*.py")) + list(repo_root.glob("src/*.py"))
               + list(repo_root.glob("scripts/**/*.py")) + list(repo_root.glob("tests/*.py")))
    pattern = re.compile(r"[\"']([A-Za-z0-9_][A-Za-z0-9_./-]*/[A-Za-z0-9_./-]+\.(?:py|json|pth|md|tex|png|jpg|txt|ini|yml))[\"']")

    broken = []
    for path in sources:
        for ref in set(pattern.findall(path.read_text())):
            if ref.startswith(("hpc/", "http", "/", "data/")) or "*" in ref:
                continue
            if not (repo_root / ref).exists():
                broken.append(f"{path.relative_to(repo_root)} -> {ref}")
    assert not broken, "source files reference paths that do not exist:\n  " + "\n  ".join(sorted(broken))


def test_appendices_use_the_official_lettered_environment(repo_root):
    """A book-class defect found only by actually compiling the PDF: Appendix
    A/B/C were rendering as "7.1", "7.2", "7.3" -- numbered sub-sections of
    Chapter 7 (Reflection) -- instead of their own lettered chapters, because
    the level-2 "## Appendix A: ..." headings under "# Appendices" were never
    told apart from an ordinary numbered section.

    No cell-by-cell check catches this kind of thing; it only shows up in the
    rendered PDF. This guards the LaTeX structure that produces the correct
    rendering instead: exactly one balanced \\begin{appendices}...\\end{appendices}
    wrapping exactly three chapter-level headings.
    """
    tex = repo_root / "thesis" / "latex" / "thesis.tex"
    if not tex.exists():
        pytest.skip("thesis/latex/thesis.tex not present")
    text = tex.read_text()

    assert text.count(r"\begin{appendices}") == 1, (
        "expected exactly one \\begin{appendices} -- Appendix A/B/C must be the "
        "only thing inside it, or the lettering (A, B, C) will be wrong"
    )
    assert text.count(r"\end{appendices}") == 1
    assert text.index(r"\begin{appendices}") < text.index(r"\end{appendices}"), (
        "\\begin{appendices} must open before it closes"
    )

    inside = text[text.index(r"\begin{appendices}"):text.index(r"\end{appendices}")]
    chapters = re.findall(r"\\chapter\{([^}]*)\}", inside)
    assert len(chapters) == 3, (
        f"expected 3 chapters inside the appendices environment (A, B, C), found "
        f"{len(chapters)}: {chapters}. If this is meant to change, update this test "
        f"deliberately -- it exists to catch an appendix silently losing its "
        f"lettering, not to block adding a fourth appendix."
    )

    builder = repo_root / "scripts" / "build_thesis.py"
    if builder.exists():
        import sys
        sys.path.insert(0, str(builder.parent))
        try:
            import build_thesis
        finally:
            sys.path.pop(0)
        assert build_thesis.APPENDICES_MARKER not in text, (
            "the internal marker used to build the appendices environment leaked "
            "into the committed thesis.tex -- the substitution in build() did not run"
        )


def test_overleaf_upload_bundle_is_complete(repo_root):
    """thesis.tex depends on two files that are easy to forget when copying a
    handful of figures into a new Overleaf project: the department's own
    CS_report.sty (without it the document does not compile at all) and the
    university crest image the title page displays (without it the title page
    silently loses its crest rather than failing to build). Both must exist
    and both must be tracked, or Yassine's next fresh Overleaf paste is missing
    a file with no obvious symptom until he tries it.
    """
    tex = repo_root / "thesis" / "latex" / "thesis.tex"
    if not tex.exists():
        pytest.skip("thesis/latex/thesis.tex not present")
    text = tex.read_text()

    if r"\usepackage{CS_report}" in text:
        sty = repo_root / "thesis" / "latex" / "CS_report.sty"
        assert sty.exists(), (
            "thesis.tex requires CS_report.sty but it is not at thesis/latex/CS_report.sty"
        )

    for match in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+\.png)\}", text):
        if match in ("figures/", ):
            continue
        candidate = repo_root / "thesis" / "figures" / match
        assert candidate.exists(), (
            f"thesis.tex references image {match!r}, which is not in thesis/figures/ "
            f"-- it will be missing from a fresh Overleaf upload"
        )
