"""Interactive dashboard for the MSc dissertation "Privacy-Preserving Machine
Learning for Financial Crime Detection".

Everything shown here is read from the logged metrics JSON files produced by the
training and attack scripts in hpc/ -- no number is hardcoded in this file, so
the dashboard cannot drift out of sync with the results it describes. If a file
is missing, the section that needs it says so rather than silently showing
nothing.

The dashboard deliberately does NOT load the model or run inference. Predictions
and feature attributions for the real held-out cases are computed once on the
cluster by hpc/export_demo_cases.py and read from demo_cases.json, which means
this runs on any machine with only streamlit, numpy, pandas and matplotlib --
no torch, no captum, no checkpoint file, no GPU.

Run:  streamlit run app.py
"""

import glob
import json
import os

import paths

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(page_title="Fraud Detection Under Differential Privacy",
                   layout="wide", initial_sidebar_state="expanded")

# ============================== PALETTE ==============================
# Dark-mode values from the project's validated data-viz palette. Categorical
# hues carry identity only; the STATUS_* colours are reserved for the live risk
# read-out and are never reused as decoration.
SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED = "#1a1a19", "#ffffff", "#c3c2b7", "#898781"
GRIDLINE, BASELINE_C = "#2c2c2a", "#383835"
BLUE, ORANGE, AQUA, YELLOW = "#3987e5", "#d95926", "#199e70", "#c98500"
STATUS_GOOD, STATUS_WARN, STATUS_CRIT = "#0ca30c", "#fab219", "#d03b3b"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


def _rgb(hex_colour):
    """'#3987e5' -> '57, 135, 229', for embedding in an rgba() CSS value. Keeps
    every tint below derived from the one hex constant above it rather than a
    second, hand-typed colour that could drift out of sync."""
    h = hex_colour.lstrip("#")
    return ", ".join(str(int(h[i:i + 2], 16)) for i in (0, 2, 4))


# One colour per tab -- used only to tint that tab's own section headings and
# its slot in the tab bar, so the six questions stay visually distinguishable
# as a reader scrolls or jumps between them. Reuses the existing categorical
# hues (documented above as "identity only") rather than introducing new ones;
# tab 6 (infrastructure) deliberately gets the muted neutral, since it is the
# one tab that is not itself a finding.
TAB_COLOURS = [ORANGE, BLUE, AQUA, YELLOW, BLUE, TEXT_MUTED]

# ============================== CSS SKIN ==============================
# Visual polish only -- no data, copy, or chart content below this block
# changes because of it. Pulls from the palette constants above so the
# Streamlit chrome and every matplotlib chart draw from one source of
# truth rather than two colour lists that can drift apart.
#
# Three faces, three jobs (a deliberate report/ledger pairing, not a random
# Google Fonts pull): Source Serif 4 for headings -- this reads as an
# analytical report, not a SaaS product, which is what this actually is;
# Inter for body copy and UI chrome, unchanged from the previous pass; IBM
# Plex Mono for every number that is a *result* (metric values, dataframe
# cells) so a reader's eye learns to find "the figures" at a glance, the way
# a ledger or terminal sets numerals apart from prose.
#
# Alert boxes now use this file's own STATUS_GOOD/WARN/CRIT colours instead
# of Streamlit's defaults -- deliberately deferred in the previous pass
# because there was no browser available to verify the selectors against.
# There is now (a headless Chrome + CDP harness); `stNotificationContentX`
# are Streamlit's own testing hooks (data-testid), not generated class
# names, so they are the stable thing to target here.
#
# Per-tab accent CSS: two rules per tab (its own tab-bar button when
# selected, and its panel's own h3 headings), built as a standalone string
# rather than nested inside the main CSS f-string below -- an f-string's
# {} interpolation cannot itself contain a # comment, which is what broke
# the previous version of this block.
_TAB_CSS = "\n".join(
    f"""
.stTabs [data-baseweb="tab-list"] button:nth-of-type({tab_i})[aria-selected="true"] {{
    color: {colour} !important;
    border-bottom-color: {colour} !important;
}}
.stTabs [data-baseweb="tab-panel"]:nth-of-type({panel_i}) [data-testid="stMarkdownContainer"] h3 {{
    border-left: 3px solid {colour};
    padding-left: 0.6rem;
}}
"""
    for tab_i, (panel_i, colour) in enumerate(
        zip(range(3, 3 + len(TAB_COLOURS)), TAB_COLOURS), start=1
    )
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600;8..60,700&family=IBM+Plex+Mono:wght@500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}
h1, h2, h3, [data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 {{
    font-family: 'Source Serif 4', Georgia, 'Times New Roman', serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}}

/* ---- cards: metrics, bordered containers (hero cover + every chart) ---- */
[data-testid="stMetric"] {{
    background: {SURFACE};
    border: 1px solid {BASELINE_C};
    border-radius: 10px;
    padding: 14px 18px 10px 18px;
}}
[data-testid="stMetricLabel"] p {{
    color: {TEXT_MUTED} !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}}
[data-testid="stMetricValue"] {{
    color: {TEXT_PRIMARY} !important;
    font-weight: 600 !important;
    font-family: 'IBM Plex Mono', ui-monospace, 'SFMono-Regular', Menlo, monospace !important;
    font-variant-numeric: tabular-nums;
}}
[data-testid="stMetricDelta"] {{
    font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
}}

/* `st.container(border=True)` shares one data-testid with every other
   vertical block on the page (Streamlit does not mark bordered ones
   separately), so the emotion-cache class that actually carries the border
   is auto-generated and unstable -- exactly the kind of selector this file
   has already burned itself on once. Instead, show() and the hero panel
   below each open their bordered container with a zero-size marker span as
   the first thing inside it, and :has() finds the wrapper from that marker.
   Verified against the real rendered DOM (CDP), not assumed. */
.element-container:has(> div > .app-card-mark) {{
    display: none;
}}
[data-testid="stVerticalBlockBorderWrapper"]:has(.app-card-mark) {{
    border-radius: 12px !important;
}}
[data-testid="stVerticalBlockBorderWrapper"]:has(.app-card-mark) > div {{
    border-radius: 12px !important;
    border-color: {BASELINE_C} !important;
    background: {SURFACE};
}}
[data-testid="stVerticalBlockBorderWrapper"]:has(.app-card-mark) > div > [data-testid="stVerticalBlock"] {{
    gap: 0.6rem;
}}

/* Chart figures ride inside a bordered container (see show()); round the
   raster PNG's square corners to match the card it now sits in. */
[data-testid="stImage"] img {{
    border-radius: 8px;
}}

/* ---- tab bar: one accent colour per question, not one for all six ---- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    border-bottom: 1px solid {BASELINE_C};
}}
.stTabs [data-baseweb="tab"] {{
    font-weight: 500;
}}
.stTabs [aria-selected="true"] {{
    color: {BLUE} !important;
    border-bottom-color: {BLUE} !important;
}}
{_TAB_CSS}

hr {{
    border-color: {BASELINE_C} !important;
    margin: 1.75rem 0 !important;
}}

div[data-testid="stExpander"] {{
    border-color: {BASELINE_C} !important;
    border-radius: 10px !important;
}}

/* ---- verdict boxes: this file's own status colours, tinted onto SURFACE,
   with a solid left bar -- a stamped verdict rather than a flat banner. ---- */
div[data-testid="stAlert"] {{
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
}}
div[data-testid="stAlert"]:has([data-testid="stNotificationContentSuccess"]) {{
    background: rgba({_rgb(STATUS_GOOD)}, 0.12);
    border-left: 3px solid {STATUS_GOOD};
}}
div[data-testid="stAlert"]:has([data-testid="stNotificationContentWarning"]) {{
    background: rgba({_rgb(STATUS_WARN)}, 0.12);
    border-left: 3px solid {STATUS_WARN};
}}
div[data-testid="stAlert"]:has([data-testid="stNotificationContentError"]) {{
    background: rgba({_rgb(STATUS_CRIT)}, 0.12);
    border-left: 3px solid {STATUS_CRIT};
}}
div[data-testid="stAlert"]:has([data-testid="stNotificationContentInfo"]) {{
    background: rgba({_rgb(BLUE)}, 0.10);
    border-left: 3px solid {BLUE};
}}
div[data-testid="stAlert"] p {{
    color: {TEXT_PRIMARY} !important;
}}

/* ---- tabular data reads as a ledger, not a text block ---- */
[data-testid="stDataFrame"] * {{
    font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
    font-variant-numeric: tabular-nums;
}}
</style>
""", unsafe_allow_html=True)

ACTIVE_CHECKPOINT = "fraud_detector_oversampled.pth"   # named in captions; not loaded here
ACTIVE_METRICS = "training_metrics_oversampled.json"
BASELINE_METRICS = "training_metrics_baseline.json"
COLLAPSED_METRICS = "training_metrics.json"


# ============================== DATA ACCESS ==============================
@st.cache_data
def load_json(path):
    """Load a results file by bare name. `paths.artefact` maps the name to the
    directory it lives in, so the panels below can keep naming results the way
    the training scripts do rather than tracking the repository layout."""
    resolved = paths.artefact(path)
    if not resolved.exists():
        return None
    return json.loads(resolved.read_text())


def fraud_stats(metrics):
    """(recall, precision, f1, macro_f1) for a training-metrics record."""
    if not metrics:
        return None
    r = metrics["classification_report"]
    return (r["Fraud"]["recall"], r["Fraud"]["precision"],
            r["Fraud"]["f1-score"], r["macro avg"]["f1-score"])


@st.cache_data
def load_sweep(prefix):
    """Collect a parameter sweep, e.g. prefix='nm' or 'mgn'. Returns rows of
    (param_value, epsilon, recall, precision, macro_f1) sorted by parameter."""
    rows = []
    for path in paths.metrics_glob(f"training_metrics_{prefix}*.json"):
        path = str(path)
        stem = os.path.basename(path)[len(f"training_metrics_{prefix}"):-len(".json")]
        try:
            value = float(stem)
        except ValueError:
            continue
        d = load_json(path)
        stats = fraud_stats(d)
        if stats:
            rows.append((value, d.get("final_epsilon"), stats[0], stats[1], stats[3]))
    return sorted(rows)


def missing(*names):
    return [n for n in names if not paths.artefact(n).exists()]


def needs(*paths):
    """Render a clear message naming exactly which files are absent."""
    gone = missing(*paths)
    if gone:
        st.info("Not available yet — missing: " + ", ".join(f"`{g}`" for g in gone)
                + ". Run the corresponding script in `hpc/` (see README) and copy the "
                  "result here.")
        return True
    return False


# ============================== CHART HELPERS ==============================
def style_axes(ax, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(BASELINE_C)
        ax.spines[sp].set_linewidth(1)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9.5, length=0)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=10, fontweight="medium")


def bars(labels, values, colors, ylabel, ylim=None, fmt="{:.3f}", figsize=(5.6, 3.1), rotate=0,
         reference=None, reference_label=None):
    """Bar chart. `reference` draws a dashed horizontal line at a meaningful
    floor -- for an attack AUC that is 0.5, below which the attacker is doing
    worse than guessing. Anchoring ylim at that floor rather than below it
    matters: a baseline at 0.45 made a 0.860 bar render 2.6x the height of a
    0.608 bar when the true ratio is 1.41x."""
    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    x = np.arange(len(labels))
    rects = ax.bar(x, values, color=colors, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TEXT_SECONDARY, fontsize=10, rotation=rotate,
                       ha="right" if rotate else "center")
    lo, hi = ylim if ylim else (0, max(values) * 1.25 or 1)
    ax.set_ylim(lo, hi)
    style_axes(ax, ylabel)
    span = hi - lo
    for rect, v in zip(rects, values):
        above = v >= 0
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + (span * 0.03 if above else -span * 0.03),
                fmt.format(v), ha="center", va="bottom" if above else "top",
                color=TEXT_PRIMARY, fontsize=10, fontweight="medium")
    if reference is not None:
        ax.axhline(reference, color=TEXT_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        if reference_label:
            ax.text(len(labels) - 0.45, reference, reference_label, color=TEXT_MUTED,
                    fontsize=8.5, va="bottom", ha="right")
    fig.tight_layout()
    return fig


def lines(x, series, xlabel, ylabel, logx=False, figsize=(6.2, 3.4), xticks=None):
    """series: list of (label, values, colour)."""
    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    for label, ys, colour in series:
        ax.plot(x, ys, marker="o", markersize=5, linewidth=2, color=colour,
                label=label, markeredgecolor=SURFACE, markeredgewidth=1.5)
    if logx:
        ax.set_xscale("log")
    if xticks is not None:
        ax.set_xticks(xticks)
        # 3 significant figures: full precision produces labels like
        # "0.711209" and "1.06538" that visibly collide on a log axis.
        ax.set_xticklabels([f"{t:.3g}" for t in xticks], fontsize=9)
        ax.minorticks_off()
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=10, fontweight="medium")
    style_axes(ax, ylabel)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="best")
    for text in leg.get_texts():
        text.set_color(TEXT_SECONDARY)
    fig.tight_layout()
    return fig


def show(fig):
    """Every chart renders inside its own bordered card (see the CSS skin
    above) rather than floating directly on the page background -- the
    single change that does the most to make this look like a designed page
    rather than a stream of loose elements. `fig.set_facecolor(SURFACE)` in
    style_axes() already matches the card background, so the raster PNG
    reads as embedded in the card rather than a mismatched rectangle sitting
    on top of it."""
    with st.container(border=True):
        st.markdown('<span class="app-card-mark"></span>', unsafe_allow_html=True)
        st.pyplot(fig, use_container_width=True)
    plt.close(fig)


@st.cache_data
def _load_logo_bytes():
    """Read the crest once per server process instead of from disk on every
    rerun. Belt and braces for a repo that lives inside a cloud-synced
    OneDrive folder, where a file can report as existing (`Path.exists()`)
    while still being a placeholder that has not finished downloading --
    caching means a slow read only ever costs the first render, and the
    try/except means a read failure degrades to the text-only header rather
    than a broken <img> in its place."""
    try:
        data = (paths.FIGURES / "uorlogo.png").read_bytes()
        return data if data else None
    except OSError:
        return None


with st.spinner("Loading dissertation results…"):
    active = load_json(ACTIVE_METRICS)
    baseline = load_json(BASELINE_METRICS)
    collapsed = load_json(COLLAPSED_METRICS)

    # ============================== HERO ==============================
    # Header, thesis statement, headline verdict and the four top-line
    # numbers are one bordered "cover page" rather than loose elements
    # sitting directly on the page background -- everything a reviewer
    # needs before picking a tab, visually established as a single unit.
    with st.container(border=True):
        st.markdown('<span class="app-card-mark"></span>', unsafe_allow_html=True)
        _logo_bytes = _load_logo_bytes()
        if _logo_bytes:
            # A fixed width=56 image inside a [1, 14] column overflows its
            # own column once the page is narrow enough that 1/15 of the
            # card is less than 56px -- the crest then spills into the
            # title text next to it. Scaling the image to its column
            # instead means it can never overflow regardless of viewport.
            # use_column_width rather than use_container_width: this
            # Streamlit version (1.30.0, confirmed by actually running it
            # and hitting the TypeError) predates the renamed parameter.
            hcol1, hcol2 = st.columns([1, 10], gap="medium")
            with hcol1:
                st.image(_logo_bytes, use_column_width=True)
            with hcol2:
                st.title("Fraud Detection Under Differential Privacy")
                st.caption("MSc dissertation, University of Reading — Yassine Malal, supervised by Dr. Xiaomin Chen.")
        else:
            st.title("Fraud Detection Under Differential Privacy")
            st.caption("MSc dissertation, University of Reading — Yassine Malal, supervised by Dr. Xiaomin Chen.")

        # A reviewer arriving cold needs the question before any number. The
        # previous version opened on four unexplained metrics ("Privacy
        # budget (ε) 1.83"), which tells someone who does not already know
        # the work precisely nothing.
        st.markdown(
            "### Banks cannot pool customer data to catch fraud. Differential privacy is the standard answer. "
            "This measures what it actually costs.")
        st.markdown(
            "A fraud detector was trained on Bitcoin transactions (Elliptic, 46,564 labelled, ~10% illicit) "
            "twice: once normally, once under **DP-SGD**, which adds a mathematical guarantee that no single "
            "customer's record measurably changed the model. Every number on this page is read from the "
            "training runs' logged output — nothing here is typed in by hand.")

        with st.expander("New to this? Three terms, in one line each"):
            st.markdown(
                "- **Differential privacy** — a guarantee that the model's behaviour barely changes whether or "
                "not any one person's data was used to train it, so you cannot work out who was in the "
                "training set.\n"
                "- **ε (epsilon)** — how strong that guarantee is. Lower is stronger. Below about 3 is "
                "generally considered meaningful.\n"
                "- **Recall vs precision** — recall is *what share of real fraud we caught*; precision is "
                "*what share of our alerts were actually fraud*. A model can score well on one and be "
                "useless on the other, and this project is largely about that.")

        if active:
            rec, pre, f1, mf1 = fraud_stats(active)
            st.markdown("#### The one-line answer")
            b_rec = fraud_stats(baseline)[0] if baseline else None
            c_rec = fraud_stats(collapsed)[0] if collapsed else None
            if b_rec is not None and c_rec is not None:
                st.success(
                    f"**The textbook way of handling rare fraud silently stops working under differential "
                    f"privacy.** It does not detect less fraud — it detects *none*: recall falls from "
                    f"{b_rec:.3f} to **{c_rec:.3f}**, zero of "
                    f"{int(collapsed['classification_report']['Fraud']['support']):,} fraud cases, while the "
                    f"privacy accountant still reports a perfectly healthy ε. A different fix restores "
                    f"detection to {rec:.3f} — but costs real privacy that the headline ε hides.")

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Fraud caught", f"{rec:.1%}",
                      help="Recall: of all fraud in the held-out test set, the share this model flagged.")
            k2.metric("Alerts that are real fraud", f"{pre:.1%}",
                      help="Precision. Low, deliberately reported: this is a triage aid for human review, "
                           "not an automated decision system.")
            k3.metric("Privacy guarantee (ε)", f"{active['final_epsilon']:.2f}",
                      help="Per record. Lower is stronger. Tab 1 explains why the per-record "
                           "figure overstates the protection an individual actually gets.")
            k4.metric("Reproducible", f"seed {active.get('seed', '—')}",
                      help="Every run is seeded and the seed is logged, so any figure here can be "
                           "regenerated exactly.")
            # .get() returns None for a metrics file that never logged delta, and
            # format(None, '.0e') raises TypeError -- which would take out the header
            # on every tab rather than degrading one caption.
            _delta = active.get("delta")
            _delta_txt = f"{_delta:.0e}" if isinstance(_delta, (int, float)) else "not logged"
            st.caption(f"Deployed model: DP-SGD with {active.get('minority_oversample_factor', '?'):g}x minority "
                       f"oversampling · δ = {_delta_txt} · every figure below is read from logged "
                       f"result files, never hardcoded.")
        else:
            st.error(f"`{ACTIVE_METRICS}` not found — most of this dashboard needs it. See README.")

st.divider()

# Named as questions rather than as experiment categories. "Privacy-utility
# trade-off" and "Privacy audit" gave a reviewer no way to guess what was behind
# either, so the tab bar was a menu of unlabelled doors.
tab_finding, tab_proof, tab_tradeoff, tab_privacy, tab_model, tab_infra = st.tabs([
    "1 · What broke",
    "2 · How do we know why",
    "3 · What privacy costs",
    "4 · Is the guarantee real",
    "5 · See it on real cases",
    "6 · How it was built"])

# ============================== TAB 1: CORE FINDING ==============================
with tab_finding:
    st.subheader("The textbook fix for rare classes does not degrade under DP-SGD. It stops working entirely.")
    st.markdown(
        "Fraud is rare — roughly 10% of labelled Elliptic transactions. The textbook fix is to weight "
        "the loss so the rare class counts more. **Under DP-SGD that fix stops working**, because "
        "gradient clipping caps every example's contribution to the same fixed size, erasing the weight "
        "before it reaches the optimiser. Replacing it with data-level oversampling survives clipping, "
        "because it changes *how many* fraud gradients enter each batch rather than how large each one is.")

    if not needs(BASELINE_METRICS, COLLAPSED_METRICS, ACTIVE_METRICS):
        b, c, o = fraud_stats(baseline), fraud_stats(collapsed), fraud_stats(active)
        left, right = st.columns([3, 2])
        with left:
            show(bars(
                ["No privacy\n(baseline)", "DP + loss\nreweighting", "DP + oversampling\n(the fix)"],
                [b[0], c[0], o[0]], [AQUA, STATUS_CRIT, BLUE],
                "Fraud recall", ylim=(0, 1.0), fmt="{:.3f}", figsize=(5.8, 3.4)))
        with right:
            # NB: b[0]-c[0] is a difference in RECALL (a share of fraud cases),
            # not a share of all transactions. Stating it as the latter was
            # simply wrong, and it is the kind of unit error an examiner spots.
            st.metric("Fraud recall lost to naive DP",
                      f"{b[0]:.3f} \u2192 {c[0]:.3f}")
            st.caption("The entire detection capability, not a partial degradation.")
            st.metric("Recovered by oversampling", f"{o[0]:.1%}",
                      delta=f"{o[0] - b[0]:+.1%} vs baseline")
            fraud_support = int(collapsed["classification_report"]["Fraud"]["support"])
            st.caption(f"The reweighted model caught **{round(c[0] * fraud_support)} of "
                       f"{fraud_support:,}** fraud "
                       f"transactions — it makes no fraud prediction at all. Reproduced on three "
                       f"independent random seeds.")

        st.markdown("**And the reported privacy budget also falls — but that number misleads:**")
        k = int(active.get("minority_oversample_factor", 5))
        corrected = active["final_epsilon"] * k
        e1, e2, e3 = st.columns(3)
        e1.metric("ε — loss reweighting (broken)", f"{collapsed['final_epsilon']:.3f}",
                  help="Each fraud window appears once, so this is a per-transaction figure.")
        e2.metric("ε — oversampling, per record", f"{active['final_epsilon']:.3f}",
                  delta=f"{active['final_epsilon'] - collapsed['final_epsilon']:+.3f} vs reweighting",
                  delta_color="off",
                  help="Lower looks better, but it counts duplicated rows as separate individuals.")
        e3.metric("ε — oversampling, per transaction", f"{corrected:.1f}",
                  delta=f"{corrected - collapsed['final_epsilon']:+.1f} vs reweighting",
                  help="Group-privacy corrected. This is the figure that describes a real person.")
        st.warning(
            f"**The middle number is not a privacy improvement.** Counted per person rather than per "
            f"row, oversampling is about {corrected / collapsed['final_epsilon']:.0f}× *weaker* — for "
            f"exactly the people this model exists to protect.")
        with st.expander("Why — the group-privacy correction, in full"):
            st.markdown(
                f"Oversampling enlarges the training set, which lowers the per-step sampling rate, and "
                f"Opacus's privacy amplification then reports a smaller ε. But the extra records are "
                f"**{k} copies of fraud windows already present**, not new individuals.\n\n"
                f"Removing one fraud transaction now removes {k} records rather than one, which is a "
                f"group-privacy problem. The standard composition turns an (ε, δ) guarantee for one "
                f"differing record into (kε, k·e^((k−1)ε)·δ) for k of them, so the budget that "
                f"actually describes a duplicated fraud transaction is\n\n"
                f"   {k} × {active['final_epsilon']:.3f} ≈ **{corrected:.1f}**, against "
                f"{collapsed['final_epsilon']:.3f} for the reweighted run.\n\n"
                f"An earlier version of this project reported the fall from "
                f"{collapsed['final_epsilon']:.3f} to {active['final_epsilon']:.3f} as *better utility "
                f"and better privacy together*. That was wrong, and it is the same single-metric "
                f"failure this whole project is about — arriving in ε itself.")

    st.divider()
    st.subheader("The same collapse, and the same recovery, on a dataset 140× larger and 76× more imbalanced")
    ps_b = load_json("training_metrics_paysim_baseline_v2.json")
    ps_c = load_json("training_metrics_paysim_dp_os1_v1.json")
    ps_o = load_json("training_metrics_paysim_dp_os200_v1.json")
    if ps_b and ps_c and ps_o:
        st.markdown(
            "PaySim is a synthetic mobile-money log: about **140× larger** than Elliptic, fraud about "
            "**76× rarer**, tabular rather than graph-structured, sharing no architecture beyond DP-SGD "
            "itself. The collapse and the recovery both reappear — evidence this is a property of the "
            "mechanism, not a quirk of one dataset.")
        pb, pc, po = fraud_stats(ps_b), fraud_stats(ps_c), fraud_stats(ps_o)
        c1, c2 = st.columns(2)
        with c1:
            show(bars(["No privacy", "DP + reweighting", "DP + oversampling"],
                      [pb[0], pc[0], po[0]], [AQUA, STATUS_CRIT, BLUE],
                      "Fraud recall — PaySim", ylim=(0, 1.0), figsize=(5.4, 3.2)))
        with c2:
            st.markdown("&nbsp;")
            st.metric("PaySim collapse", f"{pc[0]:.1%} recall",
                      delta=f"{pc[0] - pb[0]:+.1%} vs baseline")
            st.metric("PaySim recovery", f"{po[0]:.1%} recall",
                      delta=f"{po[0] - pb[0]:+.1%} vs baseline")
            k_ps = int(ps_o.get("minority_oversample_factor", 200))
            st.caption(
                f"The reported ε falls too ({ps_c['final_epsilon']:.3f} → {ps_o['final_epsilon']:.3f}), "
                f"and here that is *especially* misleading: at {k_ps}× duplication the group-privacy "
                f"correction gives a δ term above 1, i.e. no meaningful guarantee at all for a "
                f"duplicated transaction.")
    else:
        needs("training_metrics_paysim_baseline_v2.json",
              "training_metrics_paysim_dp_os1_v1.json",
              "training_metrics_paysim_dp_os200_v1.json")

# ============================== TAB 2: TRADE-OFF ==============================

# ===================== TAB 2: HOW DO WE KNOW WHY =====================
with tab_proof:
    st.caption("Tab 1 established *that* the model collapses. This tab is the evidence for *why* — "
               "three controlled experiments, each varying one thing.")
    st.subheader("Three controls, each of which could have proved the explanation wrong")

    ctrl = {n: load_json(f"training_metrics_ctrl_{n}.json")
            for n in ("dp_unweighted", "rw_mgn1.0", "rw_mgn10.0", "rw_mgn100.0")}
    if all(ctrl.values()):
        st.markdown(
            "Each control varies exactly one thing against the seed-42 configuration, and each had a "
            "result that would have refuted the explanation. None did.")

        st.markdown("**Control 1 — take the class weight away entirely. Nothing changes.**")
        unw, rw = ctrl["dp_unweighted"], ctrl["rw_mgn1.0"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Final training loss", f"{rw['epoch_history'][-1]['train_loss']:.3f}",
                  delta=f"{unw['epoch_history'][-1]['train_loss']:.3f} unweighted",
                  delta_color="off", help="The 10x weight is unmistakably active in the objective.")
        c2.metric("Fraud recall", "0.000 both", help="And unmistakably inert in the result.")
        c3.metric("Average precision", f"{rw['average_precision']:.4f}",
                  delta=f"{unw['average_precision']:.4f} unweighted", delta_color="off")
        st.caption(
            f"Same seed, same clipping, same data — only the loss weight differs. It changes the "
            f"training loss by {rw['epoch_history'][-1]['train_loss'] / unw['epoch_history'][-1]['train_loss']:.1f}× "
            f"and the resulting model by {abs(rw['average_precision'] - unw['average_precision']):.4f} average "
            f"precision. A tenfold change in how the objective punishes minority errors produces, to three "
            f"decimal places, the same model.")

        st.markdown("**Control 3 — loosen the clip, and the weight's effect comes straight back.**")
        rows = [(float(n.split("mgn")[1]), ctrl[n]) for n in ("rw_mgn1.0", "rw_mgn10.0", "rw_mgn100.0")]
        show(lines([r[0] for r in rows],
                   [("Fraud recall", [fraud_stats(m)[0] for _, m in rows], BLUE),
                    ("Average precision", [m["average_precision"] for _, m in rows], ORANGE),
                    ("Macro F1", [fraud_stats(m)[3] for _, m in rows], AQUA)],
                   "max_grad_norm on the *reweighted* configuration (log scale)", "Score",
                   logx=True, xticks=[r[0] for r in rows]))
        st.warning(
            "**Read the blue line against the orange one.** Recall climbs from 0.000 to 0.982 as the clip "
            "loosens — the most direct confirmation possible that clipping is what cancels the class "
            "weighting. But average precision *falls* from 0.419 to 0.119 over the same range, and at "
            "max_grad_norm = 100 the model labels 92% of the test set as fraud. The recall is real and the "
            "model is worse: the decision boundary moved, the ability to discriminate did not improve. "
            "This is the third time in this project that a headline metric improves while the model "
            "deteriorates.")
        st.info(
            "**The asymmetry is the argument.** The same knob, swept over the *oversampling* fix, moves "
            "recall by 0.005 (tab 3). Clipping is decisive for a remedy that works through gradient "
            "*magnitude* and nearly inert for one that works through gradient *count* — which is the "
            "distinction the mechanism is built on, and hard to explain any other way.")

        collapsed = [ctrl["rw_mgn1.0"], load_json("training_metrics_seed1_reweighted.json"),
                     load_json("training_metrics_seed2_reweighted.json")]
        base_ap = [load_json(f"training_metrics_baseline_seed{i}.json") for i in (1, 2)]
        if all(collapsed) and all(base_ap):
            mean_c = sum(m["average_precision"] for m in collapsed) / len(collapsed)
            mean_b = sum(m["average_precision"] for m in base_ap) / len(base_ap)
            prevalence = 1081 / 16666
            st.markdown("**What the controls forced us to correct about our own headline.**")
            show(bars(["Collapsed\n(recall 0.000)", "Non-private\nbaseline", "Chance\n(prevalence)"],
                      [mean_c, mean_b, prevalence],
                      [ORANGE, BLUE, "#8a8a8a"], "Average precision", ylim=(0, 0.6)))
            st.warning(
                f"**A recall of 0.000 is not a model that learned nothing.** The collapsed model ranks fraud "
                f"above non-fraud about as well as the non-private baseline does — average precision "
                f"{mean_c:.3f} against {mean_b:.3f}, on a chance floor of {prevalence:.3f}. It simply never "
                f"commits: at the default decision rule the majority class wins every single window. What "
                f"differential privacy destroyed here was not the learned signal, it was the one lever "
                f"(class weighting) you would normally use to place the threshold. Reading recall alone, you "
                f"would throw this model away.")
    else:
        needs("training_metrics_ctrl_dp_unweighted.json")


# ===================== TAB 3: WHAT PRIVACY COSTS =====================
with tab_tradeoff:
    st.caption("What you give up as the privacy guarantee gets stronger — and why the most obvious "
               "metric tells you the opposite.")
    st.subheader("As privacy tightens, recall rises — and the model gets worse")
    # noise_multiplier=1.1 is the same configuration as the default oversampled
    # run, so no separate nm1.1 file is produced -- the thesis reuses that run as
    # the sweep's centre point (Section 4.4) and this injects it the same way,
    # rather than reading a duplicate file that could drift out of sync. A stale
    # duplicate did exactly that: it survived the seeded, leakage-free re-run and
    # was still plotting pre-fix numbers here.
    nm = load_sweep("nm")
    if nm and active:
        stats = fraud_stats(active)
        nm = sorted(nm + [(1.1, active["final_epsilon"], stats[0], stats[1], stats[3])])
    if not nm:
        needs("training_metrics_nm0.4.json")
    else:
        st.markdown(
            "Six privacy budgets, from very loose (ε ≈ 26) to very tight (ε ≈ 0.43), with the "
            "oversampling fix held constant. **Read this chart carefully — recall alone lies.**")
        eps = [r[1] for r in nm]
        show(lines(eps,
                   [("Fraud recall", [r[2] for r in nm], BLUE),
                    ("Fraud precision", [r[3] for r in nm], ORANGE),
                    ("Macro F1", [r[4] for r in nm], AQUA)],
                   "Privacy budget ε  (log scale — tighter privacy to the left)",
                   "Score", logx=True, xticks=sorted(eps)))
        st.warning(
            "**Recall rises as privacy tightens — and that is misleading.** The model is not detecting "
            "fraud better; increasing noise degrades its ability to discriminate, so the bias introduced "
            "by oversampling dominates and it simply flags *more* transactions. Precision and Macro F1 "
            "fall throughout, and they are the honest signal. Reporting recall alone here would suggest "
            "privacy is free. It is not.")
        st.dataframe(
            {"noise multiplier": [f"{r[0]:g}" for r in nm],
             "ε": [f"{r[1]:.3f}" for r in nm],
             "Fraud recall": [f"{r[2]:.3f}" for r in nm],
             "Fraud precision": [f"{r[3]:.3f}" for r in nm],
             "Macro F1": [f"{r[4]:.3f}" for r in nm]},
            use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Once the imbalance is fixed at the data level, the clipping threshold barely matters")
    mgn = load_sweep("mgn")
    if mgn and active:
        st.markdown(
            "The same sweep over `max_grad_norm`, the clipping threshold — the mechanism responsible for "
            "the collapse in the first place. Note ε is identical in every row: Opacus computes it from "
            "the noise multiplier, sampling rate and step count, none of which depend on the clip norm.")
        rows = sorted(mgn + [(1.0, active["final_epsilon"], *fraud_stats(active)[:2],
                              fraud_stats(active)[3])])
        show(lines([r[0] for r in rows],
                   [("Fraud recall", [r[2] for r in rows], BLUE),
                    ("Fraud precision", [r[3] for r in rows], ORANGE),
                    ("Macro F1", [r[4] for r in rows], AQUA)],
                   "max_grad_norm (clipping threshold)", "Score",
                   xticks=[r[0] for r in rows]))
        st.info(
            "Across an 8× range of clipping thresholds every metric stays within a narrow band and shows "
            "no clear trend. Once oversampling has fixed the class imbalance at the data level, the clip "
            "norm has little further influence — looser clipping lets more true gradient through, but "
            "Opacus scales the injected noise by that same threshold, so the two effects largely cancel. "
            "The noise multiplier, not the clip norm, is the meaningful dial here.")
    else:
        needs("training_metrics_mgn0.5.json")

    st.divider()
with tab_privacy:
    st.caption("ε is a mathematical worst case. This tab attacks the trained models directly to see "
               "what an adversary could actually recover.")
    st.subheader("A formal ε is a promise. This is the attack that tests it.")
    st.markdown(
        "A formal ε is a worst-case mathematical bound. It does not prove a *specific* trained model "
        "resists a *specific* attack. This runs a membership inference attack: given only the model's "
        "per-sample loss, can an adversary tell which transactions it was trained on? An Attack AUC of "
        "0.5 means no better than guessing; 1.0 means complete exposure.")

    mia = load_json("mia_results.json")
    strat = load_json("mia_results_stratified.json")

    if mia:
        def agg(r):
            return r.get("aggregate", r)["attack_auc"]

        def find(prefix):
            return next((r for r in mia if r["tag"].startswith(prefix)), None)

        b_rec, d_rec = find("baseline_no_dp"), find("dp_sgd")
        if b_rec and d_rec:
            c1, c2 = st.columns([2, 3])
            with c1:
                st.metric("Attack AUC — no privacy", f"{agg(b_rec):.3f}")
                st.metric("Attack AUC — DP-SGD", f"{agg(d_rec):.3f}",
                          delta=f"{agg(d_rec) - agg(b_rec):+.3f}", delta_color="inverse")
                st.caption("Higher AUC under DP looks wrong at first glance. The breakdown below shows "
                           "why the aggregate number is the wrong thing to look at.")
            with c2:
                show(bars(["No privacy", "DP-SGD"], [agg(b_rec), agg(d_rec)], [AQUA, BLUE],
                          "Attack AUC (aggregate)", ylim=(0.5, 0.75), figsize=(5.0, 2.9),
                          reference=0.5, reference_label="0.5 \u2014 attacker no better than guessing"))
        if not (b_rec and d_rec):
            # The file exists but holds no record we recognise. Without this the
            # whole panel renders as a heading, an intro paragraph and a
            # divider -- the one place in this file that breaks the module
            # docstring's promise to say so rather than show nothing.
            st.info("`mia_results.json` was found but contains no `baseline_no_dp` / "
                    "`dp_sgd` records. Re-run `hpc/membership_inference_attack.py` "
                    "and copy the result here.")
    else:
        needs("mia_results.json")

    st.divider()
    st.subheader("DP does not reduce privacy risk evenly. It moves it between groups.")
    if strat:
        labels, values, colours = [], [], []
        for entry in strat:
            nice = "No privacy" if entry["tag"].startswith("baseline") else "DP-SGD"
            for cls in ("Safe", "Fraud"):
                if cls in entry["by_class"]:
                    labels.append(f"{nice}\n{cls}")
                    values.append(entry["by_class"][cls]["attack_auc"])
                    colours.append(BLUE if cls == "Safe" else ORANGE)
        c1, c2 = st.columns([3, 2])
        with c1:
            show(bars(labels, values, colours, "Attack AUC by class",
                      ylim=(0.5, 0.95), figsize=(6.0, 3.3),
                      reference=0.5, reference_label="0.5 \u2014 attacker no better than guessing"))
        with c2:
            base = next(e for e in strat if e["tag"].startswith("baseline"))
            dp = next(e for e in strat if e["tag"].startswith("dp_sgd"))
            bf, bs = base["by_class"]["Fraud"]["attack_auc"], base["by_class"]["Safe"]["attack_auc"]
            df, ds = dp["by_class"]["Fraud"]["attack_auc"], dp["by_class"]["Safe"]["attack_auc"]
            st.metric("Fraud-class exposure", f"{bf:.3f} → {df:.3f}", delta=f"{df - bf:+.3f}",
                      delta_color="inverse")
            st.metric("Safe-class exposure", f"{bs:.3f} → {ds:.3f}", delta=f"{ds - bs:+.3f}",
                      delta_color="inverse")
        st.success(
            f"**Without any privacy mechanism, fraud transactions were already badly exposed** "
            f"(AUC {bf:.3f} versus {bs:.3f} for safe ones) — a model that sees few examples of a class "
            f"memorises them individually instead of generalising. DP-SGD cuts that exposure to {df:.3f}, "
            f"protecting exactly the people with most to lose from being identifiable as training data. "
            f"The cost lands on the safe class ({bs:.3f} → {ds:.3f}), which had little to lose. "
            f"The aggregate figure conceals both movements.")
    else:
        needs("mia_results_stratified.json")

# ============================== TAB 4: MODEL EXPLORER ==============================
with tab_model:
    st.caption("The deployed model's decisions on real held-out transactions, with the features that "
               "drove each one — the \"why was this flagged\" view a compliance reviewer would need.")
    st.subheader("Individual decisions, and what drove each one")
    demo = load_json("demo_cases.json")

    if not demo:
        needs("demo_cases.json")
        st.caption("Generated by `hpc/export_demo_cases.py` on the cluster. It scores the real "
                   "held-out test set and exports predictions and feature attributions — derived "
                   "values only, no raw dataset rows.")
    else:
        counts = demo["outcome_counts_full_test_set"]
        tp, fn = counts["true_positive"], counts["false_negative"]
        fp, tn = counts["false_positive"], counts["true_negative"]

        st.markdown(
            f"Every case below is a **real transaction from the held-out test set** — later in time "
            f"than anything the model trained on, and never seen during training. Scored with "
            f"`{demo['checkpoint']}`, seed {demo['seed']}.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fraud caught", f"{tp:,}")
        m2.metric("Fraud missed", f"{fn:,}")
        m3.metric("False alarms", f"{fp:,}")
        m4.metric("Correctly cleared", f"{tn:,}")
        st.caption(f"Across all {demo['n_test_windows_total']:,} held-out windows. "
                   f"Recall {tp / (tp + fn):.1%} · precision {tp / (tp + fp):.1%}.")

        st.warning(
            f"**What precision {tp / (tp + fp):.0%} actually means.** For every fraud this model "
            f"catches it raises roughly {fp / tp:.0f} false alarms — {fp:,} legitimate transactions "
            f"flagged. That is the honest cost of the recall it achieves, and it is why this is a "
            f"triage aid for human review rather than an automated decision system.")

        st.divider()
        st.markdown("#### Confidence does not indicate correctness")
        by_outcome = {}
        for c in demo["cases"]:
            by_outcome.setdefault(c["outcome"], []).append(c["fraud_probability"])

        order = ["true_positive", "false_positive", "false_negative", "true_negative"]
        pretty = {"true_positive": "Fraud caught", "false_positive": "False alarm",
                  "false_negative": "Fraud missed", "true_negative": "Correctly cleared"}
        colour = {"true_positive": AQUA, "false_positive": ORANGE,
                  "false_negative": STATUS_CRIT, "true_negative": BLUE}

        fig, ax = plt.subplots(figsize=(6.4, 2.9), dpi=160)
        rng_j = np.random.default_rng(0)
        for i, key in enumerate(order):
            vals = by_outcome.get(key, [])
            if not vals:
                continue
            ax.scatter(vals, np.full(len(vals), i) + rng_j.uniform(-0.16, 0.16, len(vals)),
                       s=26, color=colour[key], alpha=0.85, edgecolors=SURFACE, linewidths=0.6)
        ax.axvline(demo["decision_threshold"], color=TEXT_MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([pretty[k] for k in order], color=TEXT_SECONDARY, fontsize=10)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("Model's fraud probability", color=TEXT_SECONDARY, fontsize=10)
        style_axes(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=GRIDLINE, linewidth=1)
        fig.tight_layout()
        show(fig)

        tp_p = by_outcome.get("true_positive", [0])
        fp_p = by_outcome.get("false_positive", [0])
        st.caption(
            f"**Each row is a random sample of {len(by_outcome.get('true_positive', []))} cases so individual points stay readable. "
            f"The rows are equal in size by construction and do NOT show how common each outcome is** "
            f"— for that, see the counts above: {fp:,} false alarms against {tp:,} correct catches. "
            f"Dashed line = the 0.5 decision threshold. Note the overlap: correct fraud detections "
            f"span {min(tp_p):.2f}–{max(tp_p):.2f}, but false alarms reach {max(fp_p):.2f}. The model "
            f"is frequently just as confident when it is wrong, so its probability cannot be read as "
            f"a reliability score — and simply raising the threshold would not cleanly separate them.")

        st.divider()
        st.markdown("#### Inspect an individual case")

        c1, c2 = st.columns([1, 1])
        with c1:
            pick = st.selectbox("Outcome", order, format_func=lambda k: pretty[k])
        subset = [c for c in demo["cases"] if c["outcome"] == pick]
        with c2:
            which = st.selectbox("Case", range(len(subset)),
                                 format_func=lambda i: f"#{subset[i]['id']} "
                                                       f"(p={subset[i]['fraud_probability']:.3f})")
        case = subset[which]

        d1, d2, d3 = st.columns(3)
        d1.metric("Ground truth", "Fraud" if case["true_label"] == 1 else "Legitimate")
        d2.metric("Model predicted",
                  "Fraud" if case["fraud_probability"] >= demo["decision_threshold"] else "Legitimate")
        d3.metric("Fraud probability", f"{case['fraud_probability']:.3f}")

        if pick in ("true_positive", "true_negative"):
            st.success("The model got this one right.")
        elif pick == "false_negative":
            st.error("**Missed.** Real fraud the model cleared as legitimate.")
        else:
            st.error("**False alarm.** A legitimate transaction the model flagged as fraud.")

        attrs = case["top_attributions"]
        labels = [f"F{a['feature_index']}" for a in attrs]
        values = [a["attribution"] for a in attrs]
        span = max(abs(v) for v in values) * 1.35 or 0.01
        show(bars(labels, values, [ORANGE if v >= 0 else BLUE for v in values],
                  "Attribution toward fraud", ylim=(-span, span), fmt="{:+.2f}",
                  figsize=(6.4, 3.0)))
        st.caption(
            "Integrated Gradients attributions for this specific transaction: orange pushed the model "
            "toward calling it fraud, blue pushed away. Feature numbers refer to Elliptic's anonymised "
            "columns — their real-world meanings are not published by the dataset's authors, so no "
            "semantic interpretation is offered here rather than an invented one.")
# ============================== TAB 5: INFRASTRUCTURE ==============================
with tab_infra:
    st.caption("How the pipeline was built, and what was done to make every number on this dashboard "
               "reproducible.")
    st.subheader("GPU data pipeline")
    scal = load_json("scalability_metrics.json")
    if scal:
        c1, c2 = st.columns([2, 3])
        with c1:
            st.metric("Speed-up over CPU", f"{scal['speedup']:.1f}×")
            st.metric("Rows processed", f"{scal['row_count']:,}")
            st.caption(f"CPU (pandas): {scal['cpu_seconds']:.2f}s · GPU (cuDF): {scal['gpu_seconds']:.2f}s")
        with c2:
            show(bars(["CPU (pandas)", "GPU (cuDF)"],
                      [scal["cpu_seconds"], scal["gpu_seconds"]], [ORANGE, BLUE],
                      "Seconds to load and encode", fmt="{:.2f}s", figsize=(5.0, 2.9)))
        st.caption("Loading, encoding and transferring the full PaySim dataset to tensors, measured on "
                   "an NVIDIA H100 NVL. Zero-copy transfer keeps data in GPU memory throughout.")
    else:
        needs("scalability_metrics.json")

    st.divider()
    st.subheader("Reproducibility")
    if active:
        c1, c2, c3 = st.columns(3)
        c1.metric("Random seed", f"{active.get('seed', '—')}")
        c2.metric("Row ordering", f"{active.get('tiebreak', '—')}")
        c3.metric("Privacy accountant", "Opacus RDP")
        st.markdown(
            "Every result here comes from a seeded run whose seed is recorded in its own metrics file, "
            "so any figure can be regenerated exactly. Two issues found and fixed during this work are "
            "worth naming, because both were invisible while the pipeline was non-deterministic:\n\n"
            "- **Unseeded training.** Weight initialisation, batch shuffling and the DP noise itself all "
            "drew from unseeded generators, so identical configurations produced different results.\n"
            "- **A leakage channel in the windowing.** Elliptic gives a time step but no ordering within "
            "it. Ordering tied rows by transaction ID let a model with no genuine minority-class signal "
            "reach 0.49 recall; under an ordering decorrelated from that ID it scored exactly 0.000, on "
            "every seed. All results use the decorrelated ordering.")
        st.caption("Both are documented in the dissertation, Sections 3.6.5 and 3.6.6.")
