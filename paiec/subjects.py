"""Subject attributes, and the ability prior built from them.

Item difficulty does not transfer between benchmarks; a model's general strength
does, because a 2026 frontier model beats a 2023 7B model on almost anything.
The attributes that carry it are benchmark-independent by construction, which is
exactly what the item text was not.

Target: the logit of the pair's accuracy, centred inside its benchmark. Only the
relative standing is predicted. The level of an unseen benchmark cannot be known
in advance and has to come from labels; mixing the training level in makes ALC
worse, which was measured.

Two ways in, deliberately kept apart. `subject_frame` + `design` are what the
experiments used and are frozen so their numbers stay reproducible. `Spec` +
`design_row` are the run-time path, where there is one subject and no dataframe
to take medians from, so the medians travel inside the spec. The submission
ships this module, so pandas is imported only by the offline half.

Optional design terms (Spec's keyword arguments, all off by default, so a
default Spec, its to_dict and every design row are what they were before they
existed; tests/test_hier.py pins that bit for bit). They were measured for the
hidden test in experiments/subject_side.py (docs/findings.md, "Subject side at
budgets 0 and 1"), and none is switched on in what ships:
  harnesses     one column per canonical harness string (canon_harness) seen on
                at least `harness_min` training rows, on top of the harness
                present/absent flag, which stays and is what an unseen harness
                string gets (the back-off)
  effort_order  reasoning_effort as an ordered level (EFFORT_RANK: minimal <
                low < medium < high < xhigh, max read as xhigh): a has-effort
                flag and the centred rank, in place of the per-level dummies
  date_form     how the release date enters: 'linear' (days since 2023 over
                400, the default), 'clip' (the same, the date held to the
                training rows' range, so no extrapolation), 'hinge' (linear plus
                a second slope from the training median on), 'hinge_clip' (both)
                or 'log' (log of days since 2023, floored at 60, over 400)
"""
import re
from datetime import date

import numpy as np

SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB]\b")
SMALL = re.compile(r"(mini|flash|nano|lite|small|tiny|8b|7b|3b|1\.5b|4b)", re.I)
BIG = re.compile(r"(opus|ultra|pro\b|large|405b|70b|72b|235b|671b)", re.I)
THINK = re.compile(r"(think|reason|r1\b|o[134]\b|preview)", re.I)
ISO = re.compile(r"(\d{4})(?:-(\d{1,2})(?:-(\d{1,2})(?:[T ][\d:.]*(?:Z|[+-]\d{2}:?\d{2})?)?)?)?")
EPOCH = date(2023, 1, 1)
#: reasoning_effort in order, centred on medium (Spec.effort_order). The public
#: data hold low, medium, high, xhigh and max, all on matharena; max (one
#: subject) is read as xhigh rather than given a step of its own
EFFORT_RANK = {"minimal": -2.0, "low": -1.0, "medium": 0.0, "high": 1.0, "xhigh": 2.0,
               "max": 2.0}
_EFFORT_ALIAS = {"minimum": "minimal", "min": "minimal", "med": "medium", "mid": "medium",
                 "extrahigh": "xhigh", "veryhigh": "xhigh", "maximum": "max"}
DATE_FORMS = ("linear", "clip", "hinge", "hinge_clip", "log")
HARNESS_CHARS = 64


def days_since_2023(text) -> float:
    """Days from 2023-01-01 to a release date, or nan when there is none.

    The public data hold ISO dates only (YYYY-MM-DD, YYYY-MM, empty), parsed here
    to the same day pandas.to_datetime gives, since the run-time predictor cannot
    count on pandas. Any other form goes to pandas when it is installed. A time
    zone is dropped rather than, as before, making the subtraction raise.
    """
    text = text.strip() if isinstance(text, str) else ""
    if not text:
        return np.nan
    m = ISO.fullmatch(text)
    if m:
        try:
            return (date(int(m[1]), int(m[2] or 1), int(m[3] or 1)) - EPOCH).days
        except ValueError:
            return np.nan
    try:
        import pandas as pd
        rd = pd.to_datetime(text, errors="coerce")
        if pd.isna(rd):
            return np.nan
        return (rd.tz_localize(None) - pd.Timestamp(EPOCH)).days
    except Exception:
        return np.nan


def _field(subject: dict, key: str) -> str:
    """A field as text: the official fields are strings, and anything else is
    read as its str() rather than making the whole prior fail."""
    v = subject.get(key)
    if isinstance(v, str):
        return v
    try:
        return "" if v is None else str(v)
    except Exception:
        return ""


def canon_harness(text) -> str:
    """A harness string as one key: lower case, runs of space, '-', '_', '/'
    and '.' as one space, at most HARNESS_CHARS characters ('MSWE-Agent' and
    'mswe_agent' are one harness; the version is a field of its own)."""
    t = text if isinstance(text, str) else ""
    return re.sub(r"[\s_\-/.]+", " ", t[:4 * HARNESS_CHARS].lower()).strip()[:HARNESS_CHARS]


def effort_rank(text) -> float:
    """The EFFORT_RANK of a reasoning_effort string, nan when it names no level
    ('' and 'none' included)."""
    t = re.sub(r"[\s_\-]+", "", text.lower()) if isinstance(text, str) else ""
    return EFFORT_RANK.get(_EFFORT_ALIAS.get(t, t), np.nan)


def attrs(subject: dict) -> dict:
    """Attribute dictionary for one subject, from the fields the evaluator gives.
    harness_id and effort_rank serve only the optional Spec terms."""
    name = f"{_field(subject, 'normalized_name')} {_field(subject, 'subject_features_extra')}"
    m = SIZE.search(name)
    days = days_since_2023(subject.get("release_date"))
    return dict(
        provider=_field(subject, "provider").strip().lower() or "unknown",
        has_date=int(days == days),
        days=days,
        log_size=np.log1p(float(m.group(1))) if m else np.nan,
        has_size=int(bool(m)),
        small=int(bool(SMALL.search(name))), big=int(bool(BIG.search(name))),
        think=int(bool(THINK.search(name))),
        effort=_field(subject, "reasoning_effort").strip().lower() or "none",
        harness=int(bool(_field(subject, "harness").strip())),
        harness_id=canon_harness(_field(subject, "harness")),
        effort_rank=effort_rank(_field(subject, "reasoning_effort")),
    )


def subject_frame(pairs):
    import pandas as pd
    rows = []
    for p in pairs:
        y = np.array([r.label for r in p.responses], float)
        a = attrs(p.subject)
        a.update(subject_id=p.subject_id, benchmark_id=p.benchmark_id,
                 acc=float(y.mean()), n=len(y),
                 name_len=len(p.subject.get("normalized_name") or ""))
        rows.append(a)
    df = pd.DataFrame(rows)
    df["logit"] = np.log((df.acc * df.n + 0.5) / ((1 - df.acc) * df.n + 0.5))
    df["z"] = df.logit - df.groupby("benchmark_id").logit.transform("mean")
    return df


def design(df, providers, efforts):
    """Frozen design used by the experiments. Medians come from `df` itself."""
    X = [np.ones(len(df))]
    d = df.days.fillna(df.days.median())
    X += [((d - 400) / 400).values, df.has_date.values.astype(float)]
    X += [df.log_size.fillna(df.log_size.median()).values, df.has_size.values.astype(float)]
    X += [df.small.values.astype(float), df.big.values.astype(float),
          df.think.values.astype(float), df.harness.values.astype(float)]
    for p in providers:
        X.append((df.provider == p).values.astype(float))
    for e in efforts:
        X.append((df.effort == e).values.astype(float))
    return np.column_stack(X)


class Spec:
    """Everything the run-time design needs that the training set decided. The
    keyword arguments are the optional terms of the module docstring; at their
    defaults the design is the one every earlier prior used."""

    def __init__(self, providers, efforts, med_days, med_log_size, *, harnesses=(),
                 effort_order=False, date_form="linear", date_lo=None, date_hi=None):
        self.providers = list(providers)
        self.efforts = list(efforts)
        self.med_days = float(med_days)
        self.med_log_size = float(med_log_size)
        if date_form not in DATE_FORMS:
            raise ValueError(f"date_form must be one of {DATE_FORMS}, got {date_form!r}")
        self.harnesses = [str(h) for h in harnesses]
        self.effort_order = bool(effort_order)
        self.date_form = date_form
        self.date_lo = None if date_lo is None else float(date_lo)
        self.date_hi = None if date_hi is None else float(date_hi)

    @classmethod
    def from_frame(cls, df, min_count=8, *, harness_ids=False, harness_min=None,
                   effort_order=False, date_form="linear"):
        """The spec of a training frame (subject_frame's or attrs rows). Providers
        and effort levels with at least min_count rows get a column; with
        harness_ids so do harness strings with at least harness_min (default
        min_count); with effort_order the effort is one ordered column and no
        effort level gets a dummy; a clipped date_form keeps the dated rows'
        range."""
        prov = [p for p, c in df.provider.value_counts().items()
                if c >= min_count and p != "unknown"]
        eff = [] if effort_order else [e for e, c in df.effort.value_counts().items()
                                       if c >= min_count and e != "none"]
        harn = []
        if harness_ids and "harness_id" in df:
            k = min_count if harness_min is None else harness_min
            counts = df.harness_id[df.harness_id != ""].value_counts()
            harn = sorted((h for h, c in counts.items() if c >= k),
                          key=lambda h: (-int(counts[h]), h))
        lo = hi = None
        if date_form in ("clip", "hinge_clip"):
            d = df.days.dropna()
            if len(d):
                lo, hi = float(d.min()), float(d.max())
        return cls(prov, eff, df.days.median(), df.log_size.median(), harnesses=harn,
                   effort_order=effort_order, date_form=date_form, date_lo=lo, date_hi=hi)

    def to_dict(self):
        d = dict(providers=self.providers, efforts=self.efforts,
                 med_days=self.med_days, med_log_size=self.med_log_size)
        if self.harnesses:
            d["harnesses"] = list(self.harnesses)
        if self.effort_order:
            d["effort_order"] = True
        if self.date_form != "linear":
            d["date_form"] = self.date_form
        if self.date_lo is not None:
            d["date_lo"], d["date_hi"] = self.date_lo, self.date_hi
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(d["providers"], d["efforts"], d["med_days"], d["med_log_size"],
                   harnesses=d.get("harnesses", ()), effort_order=d.get("effort_order", False),
                   date_form=d.get("date_form", "linear"), date_lo=d.get("date_lo"),
                   date_hi=d.get("date_hi"))


def _clip_days(days, spec):
    if spec.date_lo is not None:
        days = min(max(days, spec.date_lo), spec.date_hi)
    return days


def design_row(a: dict, spec: Spec) -> np.ndarray:
    days = spec.med_days if a["days"] != a["days"] else a["days"]
    ls = spec.med_log_size if a["log_size"] != a["log_size"] else a["log_size"]
    form = spec.date_form
    if form == "linear":
        dcol = (days - 400) / 400
    elif form == "log":
        dcol = float(np.log(max(days, 60.0) / 400))
    else:
        if form in ("clip", "hinge_clip"):
            days = _clip_days(days, spec)
        dcol = (days - 400) / 400
    row = [1.0, dcol, float(a["has_date"]), ls, float(a["has_size"]),
           float(a["small"]), float(a["big"]), float(a["think"]), float(a["harness"])]
    row += [1.0 if a["provider"] == p else 0.0 for p in spec.providers]
    row += [1.0 if a["effort"] == e else 0.0 for e in spec.efforts]
    if form in ("hinge", "hinge_clip"):
        row.append(max(days - spec.med_days, 0.0) / 400)
    if spec.effort_order:
        r = a.get("effort_rank", np.nan)
        row += [0.0, 0.0] if r != r else [1.0, float(r)]
    if spec.harnesses:
        h = a.get("harness_id", "")
        row += [1.0 if h == k else 0.0 for k in spec.harnesses]
    return np.array(row)


def design_matrix(df, spec: Spec) -> np.ndarray:
    return np.vstack([design_row(r, spec) for r in df.to_dict("records")])
