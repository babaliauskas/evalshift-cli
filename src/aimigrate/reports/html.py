"""Render :class:`ReportData` as a single self-contained HTML file.

Every asset (CSS, fonts, glyphs) is inlined so the report works offline
and can be emailed/zipped without breaking. We deliberately ship no
JavaScript in the MVP — static HTML is more reliable to render and
easier to skim.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from aimigrate.reports.json import ReportData

REPORT_HTML_FILENAME: str = "report.html"

_TEMPLATE_DIR = Path(__file__).parent / "templates"


_SEVERITY_GLYPHS: dict[str, str] = {
    "critical": "✗",
    "high": "✗",
    "medium": "⚠",
    "low": "⚠",
    "improved": "↑",
    "none": "✓",
    "insufficient": "?",
}


def render_html(report: ReportData) -> str:
    """Return the full HTML document as a string."""
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
        keep_trailing_newline=True,
    )
    env.globals["severity_glyph"] = lambda sev: _SEVERITY_GLYPHS.get(sev, "?")
    template = env.get_template("report.html.j2")
    css = (_TEMPLATE_DIR / "report.css").read_text(encoding="utf-8")

    return template.render(
        run_id=report.run_id,
        started_at=report.started_at,
        source_model=report.source_model,
        target_model=report.target_model,
        suite_path=report.suite_path,
        n_examples=report.n_examples,
        n_calls=report.n_calls,
        cached_calls=report.cached_calls,
        failed_calls=report.failed_calls,
        total_cost_usd=report.total_cost_usd,
        executive_summary=report.executive_summary,
        prompt_sections=report.prompt_sections,
        methodology_notes=report.methodology_notes,
        css=css,
    )


def write_html(report: ReportData, run_dir: Path) -> Path:
    """Write the HTML report into the run directory and return its path."""
    out_path = run_dir / REPORT_HTML_FILENAME
    out_path.write_text(render_html(report), encoding="utf-8")
    return out_path


__all__ = ["REPORT_HTML_FILENAME", "render_html", "write_html"]
