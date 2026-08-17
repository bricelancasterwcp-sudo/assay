"""`assay report`: one self-contained HTML page from N profiles (v1.4).

Feed it a directory of tier-marked profiles and the output is the
capability matrix: models as rows, verdicts as badges that wear their
honesty on the surface — provisional verdicts render dashed with their
interval, emulated tiers are labelled, every lens is one hover away,
and the dropped list prints in full. Stdlib only, inline CSS, no
JavaScript, no server: the file works offline and attaches to an
email. The GUI is the instrument's ethics made visible; nothing is
shown without its lens.

v1.7 adds the two knobs a PUBLISHED page needs — its own title and an
intro paragraph above the matrix — and names, per profile, the probe
version that measured it. The header has always declared schema
versions, which describe the shape of the documents; the probe version
is the instrument, it differs row by row on a campaign that spanned an
instrument change, and until now it reached a reader only through
``profile.render_table``.
"""

from __future__ import annotations

import html
from typing import Iterable

VERDICT_ORDER = (
    "structured_extraction", "patch_editing", "loop_discipline",
    # v1.6, beside loop_discipline deliberately: one says whether the
    # model can follow a scripted agent loop, the other whether the
    # endpoint takes native tools at all, and the reader wiring a model
    # into an agent needs both at once.
    "tool_calling",
    "long_context", "long_output", "chat_speed", "agent_speed",
)

#: The four colours the stylesheet actually defines. Any other verdict
#: string renders uncoloured rather than borrowing a colour it did not
#: earn — and never reaches the class attribute, so a hostile profile
#: cannot write CSS classes (or escape the attribute) through it.
_VERDICT_CLASSES = ("ready", "risky", "unusable", "unmeasured")
#: ``long_output`` names its own limit — ``degrades-at-2048`` — so its
#: verdict string is dynamic and cannot be enumerated. Every member of
#: the family means the same thing to a reader deciding whether to
#: trust the model: it works, up to a point. That is risky.
_DEGRADES_PREFIX = "degrades-at"
#: ``unsupported`` (v1.6) is a rung the model never got to attempt: the
#: endpoint refused the tools parameter. It earns no rung colour, so it
#: borrows the grey the page already uses for "nothing to show here" —
#: bucketed like ``degrades-at``, and for the same reason: the CSS is
#: not extended for a word, and the LABEL still says which of the two
#: greys this one is.
_UNRATED_VERDICTS = frozenset({"unsupported"})
#: The codec grades every schema this project has written carries, in
#: the order they are read in. Grades a profile carries beyond these
#: (v1.7's json shape grades) follow them, sorted — see
#: ``_grade_columns``.
_GRADES = ("tiny", "small", "medium")

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
.intro { max-width:70ch; margin:0 0 1.8rem; }
.intro p { margin:.55rem 0; }
"""

#: The page's name when the caller does not give it one. A constant
#: rather than a literal in the template because it appears twice — the
#: tab and the heading — and the two drifting apart is the kind of
#: nothing-bug that ships.
_DEFAULT_TITLE = "assay capability report"


def _esc(value: object) -> str:
    """Every profile-sourced value reaches the page through here.

    A profile is an UNTRUSTED document: it is written from an endpoint's
    replies, it travels by email, and this page opens in a browser. The
    rule is the whole defence and it has no exceptions — a value that
    "is obviously a number" is only a number in the profiles we wrote
    ourselves, and the one that isn't is the one that ships the markup.
    ``_num`` is the numeric wrapper and falls back here for anything it
    cannot format, so a cell is escaped whichever of the two it uses.
    """
    return html.escape(str(value))


def _num(value: object, spec: str = ".2f") -> str:
    """A measured number, or a dash. Never 0 for "not measured".

    Anything that is not a number it can format falls through to
    ``_esc``: a profile that wrote text where a rate belongs still
    reaches the page, escaped, instead of raising a ``ValueError`` out
    of the formatter and taking the whole report with it.
    """
    if value is None:
        return "—"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return format(value, spec)
    return _esc(value)


def _word(value: object) -> str:
    """A measured classification, or a dash — ``_num``'s counterpart for
    the cells that carry a word rather than a rate. It escapes rather
    than formats, because a number that arrived where a word belongs is
    document text like any other and must not reach ``format`` with a
    string spec (which raises and takes the page with it)."""
    return "—" if value is None else _esc(value)


def _show(value: object) -> str:
    """``None`` is unmeasured, in words. Same convention as
    ``profile.render_table`` — the two views of one profile must not
    disagree about what a missing measurement is called."""
    return "unmeasured" if value is None else str(value)


def _lens_title(entry: dict) -> str:
    """The hover text: every lens field, with unmeasured ones SAID.

    A capped ladder scores nothing, so its lens carries
    ``deepest_scored_tokens: None``; interpolating that straight hovered
    as ``=None`` — Python's word for unmeasured, on a page that says
    "unmeasured" everywhere else. A measured 0 still prints as 0.
    """
    parts = [f"{k}={_show(v)}" for k, v in (entry.get("lens") or {}).items()]
    return "; ".join(parts)


def _verdict_class(verdict: str) -> str:
    """The colour class for a verdict, or none at all.

    The label keeps the verdict verbatim; only the COLOUR is bucketed.
    ``degrades-at-<N>`` is the reason this exists: it is one verdict per
    measured rung, so it can only be matched by prefix, and the
    alternative — interpolating it straight into ``class=`` — produced
    ``b-degrades-at-2048``, which no rule styles and which would let an
    attacker-controlled profile write the class attribute.
    """
    if verdict.startswith(_DEGRADES_PREFIX):
        return " b-risky"
    if verdict in _UNRATED_VERDICTS:
        return " b-unmeasured"
    return f" b-{verdict}" if verdict in _VERDICT_CLASSES else ""


def _badge(entry: dict | None) -> str:
    # v1 wrote verdicts as bare strings; a string carries no lens and no
    # interval, and inventing either would be worse than saying so.
    if not isinstance(entry, dict):
        return '<span class="badge b-unmeasured">unmeasured</span>'
    verdict = str(entry.get("verdict", "unmeasured"))
    provisional = entry.get("provisional", False)
    classes = ("badge" + _verdict_class(verdict)
               + (" provisional" if provisional else ""))
    label = _esc(verdict) + ("&#8224;" if provisional else "")
    interval = entry.get("interval95")
    extra = ""
    if interval:
        extra = (f'<span class="interval">[{_num(interval[0])}, '
                 f'{_num(interval[1])}]</span>')
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
    fmt = lambda v: _num(v, ".0f")
    return (f'<span class="mono">{fmt(d)} / {fmt(p)} tok/s</span> '
            f'<span class="k">({_esc(speed.get("evidence", "?"))})</span>')


def _grade_columns(codecs: dict) -> list[str]:
    """The grade columns this profile has, in ``diff._ordered_grades``'s
    order: the long-standing grades first, then anything else sorted.

    Derived from the cells, never declared. v1.7 grades ``json_object``
    at six and the patch codecs at three, so the matrix stopped being
    rectangular; a hardcoded triple dropped the three shape grades off
    the page entirely — measured, recorded, and shown to nobody. Local
    rather than imported from ``assay.codecs`` for the reason the whole
    module reads raw dicts: this page renders documents from every era,
    including ones written before today's grade set existed.
    """
    seen = {grade for grades in codecs.values() if isinstance(grades, dict)
            for grade in grades}
    return ([grade for grade in _GRADES if grade in seen]
            + sorted(seen - set(_GRADES)))


def _codec_grid(codecs: dict | None) -> str:
    if not codecs:
        return '<p class="k">codecs unmeasured</p>'
    grades = _grade_columns(codecs)
    head = "".join(f"<th>{_esc(g)}</th>" for g in grades)
    rows = []
    for codec, cells in codecs.items():
        tds = []
        for g in grades:
            c = cells.get(g) or {}
            lands, applies, n = c.get("lands"), c.get("lands_applies"), c.get("n", 0)
            if lands is None:
                tds.append('<td class="k">—</td>')
            else:
                # v1 profiles measured `lands` and never wrote
                # `lands_applies`. Dash the half that was not measured:
                # 0.00 there would read as "nothing applied cleanly",
                # a finding v1 never made.
                tds.append(f'<td class="mono">{_num(lands)} / {_num(applies)} '
                           f'<span class="k">n={_esc(n)}</span></td>')
        rows.append(f"<tr><td>{_esc(codec)}</td>{''.join(tds)}</tr>")
    return (f'<table class="grid"><tr><th>codec '
            f'<span class="k">(byte-eq / applies)</span></th>{head}</tr>'
            + "".join(rows) + "</table>")


def _shapes_grid(shapes: list | None) -> str:
    if not shapes:
        return ""
    rows = "".join(
        f"<tr><td class='mono'>{_esc(s['shape'])}</td>"
        f"<td class='mono'>"
        f"{_esc(s['max_verified']) if s['max_verified'] is not None else '—'}"
        f"</td>"
        f"<td>{_esc(s['failure_mode'])}</td></tr>"
        for s in shapes)
    return ('<table class="grid"><tr><th>num_ctx shape</th>'
            '<th>max verified</th><th>mode</th></tr>' + rows + "</table>")


def _long_output_grid(long_output: dict | None) -> str:
    """The ladder behind a long_output verdict, rung by rung.

    The verdict is one word (or one word and a number); the grid is why.
    A rung whose reply was too short to score spent a call and measured
    nothing, and prints as dashes — never as 0.00, which would read as
    the most degenerate output ever recorded. Rungs that never ran at
    all are named separately, because a missing rung and a healthy rung
    are not the same finding.
    """
    if not long_output:
        return '<p class="k">long_output unmeasured</p>'
    rungs = long_output.get("rungs") or []
    bits = []
    if not rungs:
        bits.append('<p class="k">long_output unmeasured (no rung ran)</p>')
    else:
        rows = "".join(
            f'<tr><td class="mono">{_num(r.get("target_tokens"), "g")}</td>'
            f'<td class="mono">{_num(r.get("generated_tokens"), "g")}</td>'
            f'<td class="mono">{_num(r.get("distinct_ratio"))}</td>'
            f'<td class="mono">{_num(r.get("zlib_ratio"))}</td>'
            f'<td>{_degenerate_cell(r.get("degenerate"))}</td></tr>'
            for r in rungs if isinstance(r, dict))
        bits.append(
            '<table class="grid"><tr><th>long output target</th>'
            '<th>generated</th><th>distinct</th><th>zlib</th>'
            '<th>degenerate</th></tr>' + rows + "</table>")
    skipped = long_output.get("skipped") or []
    if skipped:
        items = "".join(f"<li>{_esc(s)}</li>" for s in skipped)
        bits.append('<p class="dropped">rungs skipped:</p>'
                    f'<ul class="dropped">{items}</ul>')
    return "".join(bits)


def _degenerate_cell(value: object) -> str:
    if value is None:
        return '<span class="k">—</span>'
    return "yes" if value else "no"


def _moe_detail(geo: dict) -> str:
    """``· MoE <used>-of-<count>``, only when BOTH counts are measured.

    ``.get`` because this reads a raw document: a profile written before
    the expert keys existed has neither, which is a schema fact, not a
    crash. A dense model has both null and gets no marker (it is not a
    0-expert MoE); one measured half gets none either — half a fact
    printed as a whole one is the overclaim the marker exists to avoid.
    """
    used, count = geo.get("expert_used_count"), geo.get("expert_count")
    if used is None or count is None:
        return ""
    return f" · MoE {_esc(used)}-of-{_esc(count)}"


#: The hover on the tools key. The probe's system line ANNOUNCES the
#: rubric it scores — call exactly one tool, use the arguments the
#: request names, quote the result token — so every rate here is a rate
#: of instructed behaviour. A reader comparing them against a harness
#: that leaves the rules unsaid should expect these high, and nothing on
#: this page may be read as a model reaching for tools on its own.
_TOOLS_TITLE = ("rates of instructed behaviour: the probe's system line "
                "announces the rubric it scores (one call, named "
                "arguments, quote the result token)")


def _tools_detail(tools: dict | None) -> str:
    """The tools line, with the three outcomes kept apart.

    ``unsupported`` prints as itself rather than as "unmeasured" — the
    endpoint was asked and said no, which is a fact about the endpoint
    and not a gap in the run — exactly as ``profile._render_tools``
    prints it, because the two views of one profile must not disagree.

    Every rate goes through ``_num``, so the halves a run could not
    measure read as dashes: ``right_tool_rate`` and ``args_valid_rate``
    are over the T1s that called at ALL and are None whenever nothing
    called, while the composite beside them is a measured 0.0.
    """
    key = f"<span class='k' title=\"{_esc(_TOOLS_TITLE)}\">tools</span>"
    if not isinstance(tools, dict):
        return f"<p>{key} unmeasured</p>"
    if tools.get("supported") is False:
        return (f"<p>{key} unsupported "
                "(the endpoint refused the tools parameter)</p>")
    if tools.get("supported") is None or tools.get("composite") is None:
        return f"<p>{key} unmeasured</p>"
    return (f"<p>{key} composite {_num(tools.get('composite'))} "
            f"(n={_esc(tools.get('n_tasks'))}) · "
            f"call {_num(tools.get('call_rate'))} · "
            f"right-tool {_num(tools.get('right_tool_rate'))} · "
            f"args {_num(tools.get('args_valid_rate'))} · "
            f"result-use {_num(tools.get('result_use_rate'))}</p>")


def _parallel_grid(parallel: dict | None) -> str:
    """What k concurrent requests did to one endpoint, row per k (v1.7).

    ``mode`` leads the row because it is the headline: an endpoint that
    BATCHES shares its throughput (each lane slower, the aggregate
    roughly held) and one that QUEUES simply serializes (each lane's
    rate divided by k), and the two look nothing alike to anyone running
    a fleet of agents. An average over both shapes would hide exactly
    the difference this family exists to show.

    Every rate goes through ``_num``, and for the family's own reason: a
    lane that errored is named in ``lane_errors`` and excluded from the
    mean, so a k where every lane failed carries None rates. 0.00 there
    would publish an unreachable endpoint as a very slow one.

    The baseline and the overlap tolerance ride above the table. A ratio
    without its denominator is not a measurement, and the tolerance is a
    CHOSEN constant that travels with the fact that it was chosen — a
    reader must not take 0.25s for a derived one.
    """
    if not parallel:
        return '<p class="k">parallel unmeasured</p>'
    rows = parallel.get("rows") or []
    head = (f"<p><span class='k'>parallel</span> single-lane baseline "
            f"{_num(parallel.get('baseline_decode_tps'))} tok/s · overlap "
            f"tolerance {_num(parallel.get('tolerance_s'))}s "
            f"({_word(parallel.get('tolerance_provenance'))})</p>")
    body = "".join(
        f'<tr><td class="mono">{_num(r.get("k"), "g")}</td>'
        f'<td>{_word(r.get("mode"))}</td>'
        f'<td class="mono">{_num(r.get("per_lane_decode_tps"))}</td>'
        f'<td class="mono">{_num(r.get("total_throughput_tps"))}</td>'
        f'<td class="mono">{_num(r.get("degradation_ratio"))}</td>'
        f'<td class="mono">{_num(r.get("n_lanes_ok"), "g")}</td>'
        f'<td class="k">{_word(r.get("evidence"))}</td></tr>'
        for r in rows if isinstance(r, dict))
    errors = [error for r in rows if isinstance(r, dict)
              for error in (r.get("lane_errors") or [])]
    tail = ""
    if errors:
        # Named, never folded into a rate: a lane that failed is
        # infrastructure evidence about the box, and the page is where a
        # reader meets it.
        items = "".join(f"<li>{_esc(e)}</li>" for e in errors)
        tail = ('<p class="dropped">lanes that errored:</p>'
                f'<ul class="dropped">{items}</ul>')
    skipped = parallel.get("skipped") or []
    if skipped:
        # The rung grid's idiom: a k the meter refused is not a k that
        # came back clean, and an absent row says neither.
        items = "".join(f"<li>{_esc(s)}</li>" for s in skipped)
        tail += ('<p class="dropped">k values skipped:</p>'
                 f'<ul class="dropped">{items}</ul>')
    if not body:
        return head + '<p class="k">no k was measured</p>' + tail
    return (head + '<table class="grid"><tr><th>concurrent lanes</th>'
            '<th>mode</th><th>per-lane tok/s</th><th>total tok/s</th>'
            '<th>vs 1 lane</th><th>lanes ok</th><th>evidence</th></tr>'
            + body + "</table>" + tail)


def _loop_detail(loop: dict) -> str:
    """The scripted loop, both scripts.

    Every rate is ``float | None`` and the error-script trio did not
    exist before v1.6, so all of them reach the page through ``_num``:
    a schema that had no error script must read as a dash, never as a
    0.00 recovery rate — that is the page claiming a model never got out
    of a failed patch when nothing ever handed it one. ``n_error_runs``
    rides beside the two rates because they do not say how much evidence
    is behind them: a budget-truncated 1/1 and a complete 5/5 both read
    1.00, and only the denominator says which was measured.
    """
    return (
        f"<p><span class='k'>loop</span> action fidelity "
        f"{_num(loop.get('action_fidelity'))} · "
        f"patch {_num(loop.get('patch_rate'))} · "
        f"finish {_num(loop.get('finish_rate'))} · "
        f"repeats {_num(loop.get('repeat_rate'))} · "
        f"anchor violations {_esc(loop.get('anchor_violations'))} "
        f"(runs={_esc(loop.get('n_runs'))}) · "
        f"recovery {_num(loop.get('recovery_rate'))} · "
        f"doom {_num(loop.get('doom_loop_rate'))} "
        f"(error runs={_num(loop.get('n_error_runs'), 'g')})</p>")


def _detail(profile: dict) -> str:
    """One profile's expanded detail. Every number on it goes through
    ``_num``, the two oldest lines included: ``ceiling.max_verified`` is
    None when the ladder verified nothing and ``envelope.fidelity`` is
    None when the budget died at n == 0, and both used to interpolate
    raw — printing Python's ``None`` on a page that dashes unmeasured
    everywhere else. ``max_verified`` keeps the ``g`` spec because it is
    a token count, not a rate.
    """
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
            f"{_esc(geo['kv_kib_per_token'])} KiB/token · usable "
            f"{_esc(geo['usable_window'])} "
            f"(limited by {_esc(geo['limited_by'])})"
            f"{_moe_detail(geo)}</p>")
    if ceiling:
        bits.append(
            f"<p><span class='k'>ceiling</span> max verified "
            f"{_num(ceiling.get('max_verified'), 'g')} · mode "
            f"{_esc(ceiling.get('failure_mode'))}</p>")
    bits.append(_shapes_grid(profile.get("ceiling_shapes")))
    if envelope:
        bits.append(f"<p><span class='k'>envelope</span> fidelity "
                    f"{_num(envelope['fidelity'])} "
                    f"(n={_esc(envelope['n'])})</p>")
    bits.append(_codec_grid(profile.get("codecs")))
    if loop:
        bits.append(_loop_detail(loop))
    bits.append(_long_output_grid(profile.get("long_output")))
    bits.append(_tools_detail(profile.get("tools")))
    # parallel last of the families: it is read AGAINST the single-lane
    # decode rate in the matrix row above, so it belongs after everything
    # that describes one request at a time.
    bits.append(_parallel_grid(profile.get("parallel")))
    dropped = prov.get("dropped") or []
    if dropped:
        items = "".join(f"<li>{_esc(d)}</li>" for d in dropped)
        bits.append(f'<p class="dropped">dropped:</p><ul class="dropped">{items}</ul>')
    bits.append(
        # The probe version leads the provenance line: it says which
        # INSTRUMENT produced everything above it, and a row measured by
        # an older probe is not comparable to one beside it just because
        # both are on the same page. ``_word`` rather than raw
        # interpolation for the module's standing reason — a document
        # that declares no probe reads as a dash, never as Python's
        # ``None`` — and it escapes, so the key is as untrusted as the
        # rest of the document.
        f"<p class='k mono'>probe={_word(profile.get('probe_version'))} · "
        f"mode={_esc(prov.get('mode'))} · "
        f"presentation={_esc(prov.get('presentation'))} · "
        f"fixtures={_esc(prov.get('fixture_set'))} · "
        f"temperature={_esc(prov.get('temperature'))} · "
        f"finished={_esc(prov.get('finished'))}</p>")
    bits.append("</details>")
    return "".join(bits)


def render_report(profiles: Iterable[dict], *, page_title: str | None = None,
                  intro_html: str | None = None) -> str:
    """The page. ``None`` for both extras is this version's standard
    report: neither parameter adds anything to it — no title change, no
    intro, not even an empty wrapper.

    It is NOT the v1.6 page to the byte, and the limit travels with the
    claim: v1.7 deliberately added the per-row ``probe=`` field and the
    intro's two CSS rules to the shared page, and nothing else about it
    moved.

    **The escape contract, because the two parameters are not alike.**
    ``page_title`` is TEXT: it goes through ``_esc`` like every other
    string on this page, because a caller that builds a title out of a
    profile field (a tier, a model name) must not become the caller that
    writes markup. ``intro_html`` is AUTHOR-supplied MARKUP and is
    inserted verbatim — it carries links, which is the whole reason it
    is HTML — and it is therefore the single trusted input to this
    module. A caller that interpolates any profile-sourced value into it
    owns escaping that value (``scripts/build_matrix.py`` does, with
    ``html.escape``). Nothing in this file ever routes document text
    into it.
    """
    profiles = list(profiles)
    title = _DEFAULT_TITLE if page_title is None else _esc(page_title)
    # Empty rather than an empty wrapper: a stray <section> in every
    # operator's report is not "the same page" just because it is
    # invisible.
    intro = ("" if intro_html is None
             else f'<section class="intro">{intro_html}</section>\n')
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
<title>{title}</title>
<style>{_CSS}</style></head><body>
<h1>{title}</h1>
<p class="sub">{len(profiles)} profile(s) · schema version(s)
{_esc(sorted(v for v in versions if v is not None))} · every verdict wears its
lens (hover a badge); &#8224; = provisional — this sample cannot separate the
verdict from its neighbours, or the ladder behind it did not finish.</p>
{intro}<table class="matrix"><tr><th>model / tier</th>
<th>decode / prefill</th>{header}</tr>{"".join(rows)}</table>
{details}
</body></html>
"""
