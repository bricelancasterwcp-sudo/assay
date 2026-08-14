"""`assay report`: one self-contained HTML page from N profiles (v1.4).

Feed it a directory of tier-marked profiles and the output is the
capability matrix: models as rows, verdicts as badges that wear their
honesty on the surface — provisional verdicts render dashed with their
interval, emulated tiers are labelled, every lens is one hover away,
and the dropped list prints in full. Stdlib only, inline CSS, no
JavaScript, no server: the file works offline and attaches to an
email. The GUI is the instrument's ethics made visible; nothing is
shown without its lens.
"""

from __future__ import annotations

import html
from typing import Iterable

VERDICT_ORDER = (
    "structured_extraction", "patch_editing", "loop_discipline",
    "long_context", "chat_speed", "agent_speed",
)

_CSS = """
:root { --bg:#fafaf7; --fg:#1c1c1a; --muted:#6b6b66; --card:#ffffff;
  --line:#e3e3dc; --ready:#1a7f37; --risky:#b58900; --unusable:#c0392b;
  --unmeasured:#8a8a84; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#191917; --fg:#e8e8e3; --muted:#9a9a92; --card:#22221f;
    --line:#33332e; --ready:#4fc26e; --risky:#d9a521; --unusable:#e06552;
    --unmeasured:#77776f; }
}
* { box-sizing: border-box; }
body { margin:0; padding:2rem clamp(1rem,4vw,3rem); background:var(--bg);
  color:var(--fg); font:15px/1.55 system-ui, sans-serif; }
h1 { font-size:1.6rem; margin:0 0 .2rem; }
.sub { color:var(--muted); margin-bottom:1.6rem; }
table.matrix { border-collapse:collapse; width:100%; margin:1rem 0 2rem; }
.matrix th, .matrix td { text-align:left; padding:.5rem .7rem;
  border-bottom:1px solid var(--line); vertical-align:top; }
.matrix th { font-size:.78rem; text-transform:uppercase;
  letter-spacing:.05em; color:var(--muted); }
.badge { display:inline-block; padding:.12rem .5rem; border-radius:.6rem;
  font-size:.82rem; font-weight:600; border:1.5px solid transparent; }
.badge.provisional { border-style:dashed; opacity:.92; }
.b-ready { color:var(--ready); border-color:var(--ready); }
.b-risky { color:var(--risky); border-color:var(--risky); }
.b-unusable { color:var(--unusable); border-color:var(--unusable); }
.b-unmeasured { color:var(--unmeasured); border-color:var(--unmeasured); }
.tier { font-size:.78rem; color:var(--muted); }
.tier .emu { border:1px dashed var(--muted); border-radius:.4rem;
  padding:0 .3rem; margin-left:.3rem; }
.interval { display:block; font-size:.72rem; color:var(--muted);
  font-weight:400; }
details { background:var(--card); border:1px solid var(--line);
  border-radius:.7rem; margin:.6rem 0; padding:.7rem 1rem; }
summary { cursor:pointer; font-weight:600; }
table.grid { border-collapse:collapse; margin:.6rem 0; }
.grid th, .grid td { border:1px solid var(--line); padding:.3rem .6rem;
  font-size:.85rem; }
.mono { font-family: ui-monospace, monospace; font-size:.85rem; }
.dropped { color:var(--muted); font-size:.85rem; }
.k { color:var(--muted); }
"""


def _esc(value: object) -> str:
    return html.escape(str(value))


def _lens_title(entry: dict) -> str:
    parts = [f"{k}={v}" for k, v in (entry.get("lens") or {}).items()]
    return "; ".join(parts)


def _badge(entry: dict | None) -> str:
    if not isinstance(entry, dict):
        return '<span class="badge b-unmeasured">unmeasured</span>'
    verdict = entry.get("verdict", "unmeasured")
    provisional = entry.get("provisional", False)
    classes = f"badge b-{verdict}" + (" provisional" if provisional else "")
    label = verdict + ("&#8224;" if provisional else "")
    interval = entry.get("interval95")
    extra = ""
    if interval:
        extra = (f'<span class="interval">[{interval[0]:.2f}, '
                 f'{interval[1]:.2f}]</span>')
    return (f'<span class="{classes}" title="{_esc(_lens_title(entry))}">'
            f'{label}</span>{extra}')


def _tier_cell(profile: dict) -> str:
    prov = profile.get("provenance") or {}
    tier = prov.get("tier")
    if not tier:
        return '<span class="tier">undeclared</span>'
    emu = prov.get("emulated")
    tag = ('<span class="emu">emulated</span>' if emu
           else '<span class="emu">real hw</span>' if emu is False else "")
    return f'<span class="tier">{_esc(tier)}{tag}</span>'


def _speed_cell(profile: dict) -> str:
    speed = profile.get("speed")
    if not speed:
        return '<span class="k">unmeasured</span>'
    d = speed.get("decode_tps")
    p = speed.get("prefill_tps")
    fmt = lambda v: "—" if v is None else f"{v:.0f}"
    return (f'<span class="mono">{fmt(d)} / {fmt(p)} tok/s</span> '
            f'<span class="k">({_esc(speed.get("evidence", "?"))})</span>')


def _codec_grid(codecs: dict | None) -> str:
    if not codecs:
        return '<p class="k">codecs unmeasured</p>'
    grades = ("tiny", "small", "medium")
    head = "".join(f"<th>{g}</th>" for g in grades)
    rows = []
    for codec, cells in codecs.items():
        tds = []
        for g in grades:
            c = cells.get(g) or {}
            lands, applies, n = c.get("lands"), c.get("lands_applies"), c.get("n", 0)
            if lands is None:
                tds.append('<td class="k">—</td>')
            else:
                tds.append(f'<td class="mono">{lands:.2f} / {applies:.2f} '
                           f'<span class="k">n={n}</span></td>')
        rows.append(f"<tr><td>{_esc(codec)}</td>{''.join(tds)}</tr>")
    return (f'<table class="grid"><tr><th>codec '
            f'<span class="k">(byte-eq / applies)</span></th>{head}</tr>'
            + "".join(rows) + "</table>")


def _shapes_grid(shapes: list | None) -> str:
    if not shapes:
        return ""
    rows = "".join(
        f"<tr><td class='mono'>{s['shape']}</td>"
        f"<td class='mono'>{s['max_verified'] if s['max_verified'] is not None else '—'}</td>"
        f"<td>{_esc(s['failure_mode'])}</td></tr>"
        for s in shapes)
    return ('<table class="grid"><tr><th>num_ctx shape</th>'
            '<th>max verified</th><th>mode</th></tr>' + rows + "</table>")


def _detail(profile: dict) -> str:
    model = (profile.get("model") or {}).get("name", "?")
    geo = profile.get("geometry")
    ceiling = profile.get("ceiling")
    envelope = profile.get("envelope")
    loop = profile.get("loop")
    prov = profile.get("provenance") or {}
    bits = [f"<details><summary>{_esc(model)}</summary>"]
    if geo:
        bits.append(
            f"<p><span class='k'>geometry</span> "
            f"{geo['kv_kib_per_token']} KiB/token · usable "
            f"{geo['usable_window']} (limited by {_esc(geo['limited_by'])})</p>")
    if ceiling:
        bits.append(
            f"<p><span class='k'>ceiling</span> max verified "
            f"{ceiling.get('max_verified')} · mode "
            f"{_esc(ceiling.get('failure_mode'))}</p>")
    bits.append(_shapes_grid(profile.get("ceiling_shapes")))
    if envelope:
        bits.append(f"<p><span class='k'>envelope</span> fidelity "
                    f"{envelope['fidelity']} (n={envelope['n']})</p>")
    bits.append(_codec_grid(profile.get("codecs")))
    if loop:
        bits.append(
            f"<p><span class='k'>loop</span> action fidelity "
            f"{loop['action_fidelity']} · patch {loop['patch_rate']} · "
            f"finish {loop['finish_rate']} · repeats {loop['repeat_rate']} · "
            f"anchor violations {loop['anchor_violations']} "
            f"(runs={loop['n_runs']})</p>")
    dropped = prov.get("dropped") or []
    if dropped:
        items = "".join(f"<li>{_esc(d)}</li>" for d in dropped)
        bits.append(f'<p class="dropped">dropped:</p><ul class="dropped">{items}</ul>')
    bits.append(
        f"<p class='k mono'>mode={_esc(prov.get('mode'))} · "
        f"presentation={_esc(prov.get('presentation'))} · "
        f"fixtures={_esc(prov.get('fixture_set'))} · "
        f"temperature={_esc(prov.get('temperature'))} · "
        f"finished={_esc(prov.get('finished'))}</p>")
    bits.append("</details>")
    return "".join(bits)


def render_report(profiles: Iterable[dict]) -> str:
    profiles = list(profiles)
    header = "".join(f"<th>{_esc(v)}</th>" for v in VERDICT_ORDER)
    rows = []
    for p in profiles:
        model = (p.get("model") or {}).get("name", "?")
        verdicts = p.get("verdicts") or {}
        cells = "".join(f"<td>{_badge(verdicts.get(v))}</td>"
                        for v in VERDICT_ORDER)
        rows.append(
            f"<tr><td><strong>{_esc(model)}</strong><br>{_tier_cell(p)}</td>"
            f"<td>{_speed_cell(p)}</td>{cells}</tr>")
    details = "".join(_detail(p) for p in profiles)
    versions = {p.get("assay_profile_version") for p in profiles}
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>assay capability report</title>
<style>{_CSS}</style></head><body>
<h1>assay capability report</h1>
<p class="sub">{len(profiles)} profile(s) · schema version(s)
{_esc(sorted(v for v in versions if v is not None))} · every verdict wears its
lens (hover a badge); &#8224; = provisional at this sample size.</p>
<table class="matrix"><tr><th>model / tier</th>
<th>decode / prefill</th>{header}</tr>{"".join(rows)}</table>
{details}
</body></html>
"""
