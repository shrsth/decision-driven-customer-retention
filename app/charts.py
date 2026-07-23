"""Plotly chart builders for the dashboard.

Colors come from the validated dark-mode data-viz palette (colorblind-safe,
contrast-checked against the #1a1a19 surface). Charts follow the method:
thin marks, recessive grid/axes, direct identity, one axis per chart.
"""

import plotly.graph_objects as go

# --- Palette (validated dark-mode data-viz palette; this project's own) ---
SURFACE = "#1a1a19"
INK = "#ffffff"
INK_MUTED = "#898781"
GRID = "#2c2c2a"
BASELINE = "#383835"

SERIES_BLUE = "#3987e5"
SERIES_AQUA = "#199e70"

STATUS_GOOD = "#0ca30c"      # ACT
STATUS_WARNING = "#fab219"   # MONITOR
STATUS_MUTED = "#6b6a64"     # IGNORE
STATUS_CRITICAL = "#d03b3b"

FONT = '"Space Grotesk", "Inter", system-ui, sans-serif'


def _base_layout(height=320, **kwargs):
    return dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, family=FONT, size=13),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(gridcolor=GRID, zerolinecolor=BASELINE, linecolor=BASELINE,
                   tickfont=dict(color=INK_MUTED)),
        yaxis=dict(gridcolor=GRID, zerolinecolor=BASELINE, linecolor=BASELINE,
                   tickfont=dict(color=INK_MUTED)),
        hoverlabel=dict(bgcolor=SURFACE, font=dict(color=INK, family=FONT)),
        showlegend=False,
        **kwargs,
    )


def action_segment_chart(act, monitor, ignore):
    """Horizontal bar of ACT / MONITOR / IGNORE counts, status-colored."""
    labels = ["ACT", "MONITOR", "IGNORE"]
    values = [act, monitor, ignore]
    colors = [STATUS_GOOD, STATUS_WARNING, STATUS_MUTED]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:,}" for v in values], textposition="outside",
        cliponaxis=False,  # let outside labels render past the plot edge
        textfont=dict(color=INK), hovertemplate="%{y}: %{x:,}<extra></extra>",
    ))
    layout = _base_layout(height=240)
    layout["margin"] = dict(l=10, r=60, t=30, b=10)
    # headroom so the longest bar's label doesn't collide with the plot edge
    layout["xaxis"].update(range=[0, max(values) * 1.18])
    fig.update_layout(**layout)
    fig.update_yaxes(categoryorder="array", categoryarray=labels[::-1])
    fig.update_xaxes(showgrid=True)
    return fig


def calibration_chart(calibration_rows):
    """Predicted vs. actual churn rate by decile, against a y=x reference.

    Points on the diagonal mean the probabilities are honest — which is the
    whole justification for choosing logistic regression.
    """
    predicted = [r["avg_predicted"] for r in calibration_rows]
    actual = [r["actual_churn_rate"] for r in calibration_rows]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color=INK_MUTED, width=1, dash="dash"),
        hoverinfo="skip", name="Perfect calibration",
    ))
    fig.add_trace(go.Scatter(
        x=predicted, y=actual, mode="lines+markers",
        line=dict(color=SERIES_BLUE, width=2),
        marker=dict(color=SERIES_BLUE, size=9,
                    line=dict(color=SURFACE, width=2)),
        hovertemplate="predicted %{x:.2f}<br>actual %{y:.2f}<extra></extra>",
        name="Model",
    ))
    layout = _base_layout(height=340)
    layout["xaxis"].update(title="Predicted churn probability", range=[0, 0.85])
    layout["yaxis"].update(title="Actual churn rate", range=[0, 0.85])
    fig.update_layout(**layout)
    return fig


def segment_churn_chart(segment_rows):
    """Churn rate by contract + internet service, sorted, single-hue bars."""
    rows = sorted(segment_rows, key=lambda r: r["churn_rate"])
    labels = [f'{r["Contract"]} · {r["InternetService"]}' for r in rows]
    rates = [r["churn_rate"] * 100 for r in rows]

    fig = go.Figure(go.Bar(
        x=rates, y=labels, orientation="h",
        marker=dict(color=SERIES_BLUE, line=dict(width=0)),
        text=[f"{r:.0f}%" for r in rates], textposition="outside",
        cliponaxis=False,
        textfont=dict(color=INK),
        hovertemplate="%{y}<br>churn %{x:.1f}%<extra></extra>",
    ))
    layout = _base_layout(height=360)
    layout["margin"] = dict(l=10, r=40, t=30, b=10)
    layout["xaxis"].update(title="Churn rate (%)", range=[0, max(rates) * 1.2])
    fig.update_layout(**layout)
    return fig


def roi_sensitivity_chart(save_rates, rois, current_rate=None):
    """ROI as a function of the intervention save rate.

    Shows the honest truth: ROI collapses under pessimistic save rates, so the
    business case depends on the intervention actually working.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[r * 100 for r in save_rates], y=rois, mode="lines+markers",
        line=dict(color=SERIES_AQUA, width=2),
        marker=dict(color=SERIES_AQUA, size=7,
                    line=dict(color=SURFACE, width=2)),
        hovertemplate="save rate %{x:.0f}%<br>ROI %{y:.1f}x<extra></extra>",
        name="ROI",
    ))
    # break-even reference (ROI = 1x)
    fig.add_hline(y=1.0, line=dict(color=STATUS_CRITICAL, width=1, dash="dot"),
                  annotation_text="break-even", annotation_position="top left",
                  annotation_font_color=INK_MUTED)
    if current_rate is not None:
        fig.add_vline(x=current_rate * 100,
                      line=dict(color=INK_MUTED, width=1, dash="dash"))
    layout = _base_layout(height=340)
    layout["xaxis"].update(title="Intervention save rate (%)")
    layout["yaxis"].update(title="Retention ROI (x)")
    fig.update_layout(**layout)
    return fig


def policy_comparison_chart(rows):
    """Expected value captured by each targeting policy, same budget.

    The economic engine bar is highlighted; naive baselines are muted. Makes
    the case that the decision layer — not just the model — creates value.
    """
    order = sorted(rows, key=lambda r: r["expected_value"])
    labels = [r["policy"] for r in order]
    values = [r["expected_value"] for r in order]
    colors = [STATUS_GOOD if r["policy"] == "Decision engine" else STATUS_MUTED
              for r in order]
    text = [f'${v:,.0f}  ({r["roi"]:.1f}x)' for v, r in zip(values, order)]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=text, textposition="outside", cliponaxis=False,
        textfont=dict(color=INK),
        hovertemplate="%{y}: $%{x:,.0f}<extra></extra>",
    ))
    layout = _base_layout(height=300)
    layout["margin"] = dict(l=10, r=110, t=30, b=10)
    lo = min(0, min(values)) * 1.1
    layout["xaxis"].update(title="Expected value captured ($)",
                           range=[lo, max(values) * 1.32])
    fig.update_layout(**layout)
    return fig


def profit_threshold_chart(pt):
    """Realized holdout value as a function of the churn-probability cutoff.

    Marks the profit-maximizing threshold vs. the naive 0.5 cutoff.
    """
    xs = [t * 100 for t in pt["thresholds"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=pt["values"], mode="lines",
        line=dict(color=SERIES_BLUE, width=2),
        hovertemplate="cutoff %{x:.0f}%<br>value $%{y:,.0f}<extra></extra>",
        name="Realized value",
    ))
    fig.add_hline(y=0, line=dict(color=BASELINE, width=1))
    fig.add_vline(
        x=pt["best_threshold"] * 100,
        line=dict(color=STATUS_GOOD, width=1.5, dash="dash"),
        annotation_text=f"best {pt['best_threshold']*100:.0f}%",
        annotation_position="top right", annotation_font_color=INK_MUTED,
    )
    fig.add_vline(
        x=50, line=dict(color=STATUS_CRITICAL, width=1.5, dash="dot"),
        annotation_text="naive 50%",
        annotation_position="top left", annotation_font_color=INK_MUTED,
    )
    layout = _base_layout(height=320)
    layout["xaxis"].update(title="Act-if churn probability >= cutoff (%)")
    layout["yaxis"].update(title="Realized value on holdout ($)")
    fig.update_layout(**layout)
    return fig


def simulation_chart(sim):
    """Histogram of simulated ROI with the 90% CI band and break-even line."""
    rois = sim["rois"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=rois, nbinsx=44,
        marker=dict(color=SERIES_BLUE, line=dict(width=0)),
        opacity=0.85, hovertemplate="ROI %{x:.1f}x<br>%{y} scenarios<extra></extra>",
    ))
    # 90% CI shaded band
    fig.add_vrect(
        x0=sim["roi_p5"], x1=sim["roi_p95"],
        fillcolor=SERIES_BLUE, opacity=0.10, line_width=0,
    )
    fig.add_vline(x=sim["roi_mean"], line=dict(color=STATUS_GOOD, width=2),
                  annotation_text=f"mean {sim['roi_mean']:.1f}x",
                  annotation_position="top", annotation_font_color=INK_MUTED)
    fig.add_vline(x=0, line=dict(color=STATUS_CRITICAL, width=1.5, dash="dot"),
                  annotation_text="break-even",
                  annotation_position="top left", annotation_font_color=INK_MUTED)
    layout = _base_layout(height=320)
    layout["xaxis"].update(title="Realized retention ROI (x)")
    layout["yaxis"].update(title="Scenarios")
    fig.update_layout(**layout)
    return fig


def feature_importance_chart(rows):
    """Standardized LR coefficients — churn drivers vs. protective factors.

    Red bars raise churn odds; green bars lower them.
    """
    order = sorted(rows, key=lambda r: r["coefficient"])
    labels = [r["feature"] for r in order]
    coefs = [r["coefficient"] for r in order]
    colors = [STATUS_CRITICAL if c > 0 else STATUS_GOOD for c in coefs]

    fig = go.Figure(go.Bar(
        x=coefs, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="%{y}: %{x:.2f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=BASELINE, width=1))
    layout = _base_layout(height=380)
    layout["margin"] = dict(l=10, r=20, t=30, b=10)
    layout["xaxis"].update(title="Coefficient  (+ raises churn, - protects)")
    fig.update_layout(**layout)
    return fig


def sensitivity_chart(rates, net_values, break_even, assumed):
    """Realized net program value vs. the true intervention save rate.

    The action set is fixed (chosen at the assumed rate); this shows how much
    slack there is before the program stops paying for itself.
    """
    xs = [r * 100 for r in rates]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=net_values, mode="lines",
        line=dict(color=SERIES_BLUE, width=2),
        hovertemplate="save rate %{x:.0f}%<br>net $%{y:,.0f}<extra></extra>",
        name="Net value",
    ))
    fig.add_hline(y=0, line=dict(color=BASELINE, width=1))
    if break_even == break_even:  # not NaN
        fig.add_vline(
            x=break_even * 100,
            line=dict(color=STATUS_CRITICAL, width=1.5, dash="dash"),
            annotation_text=f"break-even {break_even*100:.0f}%",
            annotation_position="top left", annotation_font_color=INK_MUTED,
        )
    fig.add_vline(
        x=assumed * 100, line=dict(color=STATUS_GOOD, width=1.5, dash="dot"),
        annotation_text=f"assumed {assumed*100:.0f}%",
        annotation_position="top right", annotation_font_color=INK_MUTED,
    )
    layout = _base_layout(height=320)
    layout["xaxis"].update(title="True intervention save rate (%)")
    layout["yaxis"].update(title="Net program value ($)")
    fig.update_layout(**layout)
    return fig
