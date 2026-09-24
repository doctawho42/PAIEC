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


def attrs(subject: dict) -> dict:
    """Attribute dictionary for one subject, from the fields the evaluator gives."""
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
    """Everything the run-time design needs that the training set decided."""

    def __init__(self, providers, efforts, med_days, med_log_size):
        self.providers = list(providers)
        self.efforts = list(efforts)
        self.med_days = float(med_days)
        self.med_log_size = float(med_log_size)

    @classmethod
    def from_frame(cls, df, min_count=8):
        prov = [p for p, c in df.provider.value_counts().items()
                if c >= min_count and p != "unknown"]
        eff = [e for e, c in df.effort.value_counts().items()
               if c >= min_count and e != "none"]
        return cls(prov, eff, df.days.median(), df.log_size.median())

    def to_dict(self):
        return dict(providers=self.providers, efforts=self.efforts,
                    med_days=self.med_days, med_log_size=self.med_log_size)

    @classmethod
    def from_dict(cls, d):
        return cls(d["providers"], d["efforts"], d["med_days"], d["med_log_size"])


def design_row(a: dict, spec: Spec) -> np.ndarray:
    days = spec.med_days if a["days"] != a["days"] else a["days"]
    ls = spec.med_log_size if a["log_size"] != a["log_size"] else a["log_size"]
    row = [1.0, (days - 400) / 400, float(a["has_date"]), ls, float(a["has_size"]),
           float(a["small"]), float(a["big"]), float(a["think"]), float(a["harness"])]
    row += [1.0 if a["provider"] == p else 0.0 for p in spec.providers]
    row += [1.0 if a["effort"] == e else 0.0 for e in spec.efforts]
    return np.array(row)


def design_matrix(df, spec: Spec) -> np.ndarray:
    return np.vstack([design_row(r, spec) for r in df.to_dict("records")])
