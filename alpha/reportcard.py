"""§10 Report Card Output Format — presentation layer over alpha.agents.alpha's data."""

from __future__ import annotations


def build_sparkline_svg(prices: list[float], width: int = 640, height: int = 140) -> str:
    """Renders the price_series as an inline SVG line chart with a glow
    filter — the "holographic chart overlay." Built from the same closes
    already fetched for the report, not a decorative placeholder.
    """
    if len(prices) < 2:
        return ""

    lo, hi = min(prices), max(prices)
    span = (hi - lo) or 1.0
    pad = 10
    plot_w, plot_h = width - 2 * pad, height - 2 * pad

    points = [
        (pad + i / (len(prices) - 1) * plot_w, pad + plot_h - (p - lo) / span * plot_h)
        for i, p in enumerate(prices)
    ]
    line_path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    area_path = line_path + f" L {points[-1][0]:.1f},{height - pad} L {points[0][0]:.1f},{height - pad} Z"

    rising = prices[-1] >= prices[0]
    stroke = "#3cf28f" if rising else "#ff8a3d"

    return f"""
<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="sparkFill" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{stroke}" stop-opacity="0.35"/>
      <stop offset="100%" stop-color="{stroke}" stop-opacity="0"/>
    </linearGradient>
    <filter id="sparkGlow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="3.2" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  <path d="{area_path}" fill="url(#sparkFill)" stroke="none"/>
  <path d="{line_path}" fill="none" stroke="{stroke}" stroke-width="2" filter="url(#sparkGlow)"/>
</svg>
""".strip()


def format_text(report: dict) -> str:
    """Renders the exact §10 template as plain text (used by the email digest
    and available for a CLI/plaintext view)."""
    targets = ", ".join(f"{t:.2f}" for t in report["targets"])
    forecast = report["model_forecast"]
    forecast_line = (
        f"{forecast['predicted_next_day_return']:+.3%} next-day (learned signal)"
        if forecast.get("available") else f"unavailable — {forecast['reason']}"
    )

    lines = [
        f"Ticker: {report['ticker']}",
        f"Signal: {report['signal']}",
        f"Entry: {report['entry']:.2f}            Stop-loss: {report['stop_loss']:.2f}",
        f"Target(s): {targets}     Risk/Reward: {report['risk_reward']}",
        f"Confidence: {report['confidence']}",
        f"Key levels: range zone={report['key_levels']['range_zone']}, "
        f"{len(report['key_levels']['liquidity_levels'])} liquidity level(s), "
        f"{len(report['key_levels']['order_blocks'])} order block(s)",
        f"Model forecast: {forecast_line}",
        "Confluence notes:",
    ]
    lines += [f"  - {note}" for note in report["confluence_notes"]]
    return "\n".join(lines)
