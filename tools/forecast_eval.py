"""Offline: how well does the online calibration forecast the REMAINING first-pass cost?

Usage (from experiments/runs): python ../../tools/forecast_eval.py RUN_NAME...  (12.11: 19 runs)
"""

import collections
import json
import math
import re
import sys

LINE = re.compile(r"^\[agent\s+[\d.]+s\]\s+(\S+)\s+(answered|infeasible|failed)\s+(\S+)\s+cost=\s*([\d.]+)")
QUANTILES = (0.1, 0.25, 0.5, 0.75)


def seq(run):
    summ = {t["id"]: t for t in json.load(open(f"{run}/logs/summary.json"))["tasks"]}
    out, seen = [], set()
    for line in open(f"{run}/logs/stderr.log", errors="replace"):
        m = LINE.match(line)
        if m and m.group(1) not in seen and m.group(3).startswith("pass1"):
            seen.add(m.group(1))
            t = summ[m.group(1)]
            out.append((t["type"], t["estimate"], float(m.group(4))))
    llm_tasks = [
        t for t in summ.values()
        if t["estimate"] and t["status"] != "skipped" and t.get("source", "") not in ("fastpath", "precheck")
    ]
    return out, len(seen) >= 0.97 * len(llm_tasks)


def geo(done, w=3.0):
    s = collections.defaultdict(float)
    n = collections.Counter()
    for ty, e, a in done:
        if e > 0 and a > 0:
            s[ty] += math.log(a / e)
            n[ty] += 1
    return lambda ty: math.exp(s[ty] / (n[ty] + w))


def arith(done, w=3.0, pooled=False):
    se = collections.defaultdict(float)
    sa = collections.defaultdict(float)
    n = collections.Counter()
    for ty, e, a in done:
        se[ty] += e
        sa[ty] += a
        n[ty] += 1
    total_e, total_a = sum(se.values()), sum(sa.values())
    g = total_a / total_e if total_e else 1.0

    def f(ty):
        prior = g if pooled else 1.0
        m = se[ty] / n[ty] if n[ty] else 0
        return (sa[ty] + w * m * prior) / (se[ty] + w * m) if n[ty] else prior

    return f


def recent(done, k=30, w=3.0):
    """Recency window: the last k tasks of each type."""
    by = collections.defaultdict(list)
    for ty, e, a in done:
        by[ty].append((e, a))

    def f(ty):
        xs = by[ty][-k:]
        if not xs:
            return 1.0
        se = sum(e for e, _ in xs)
        sa = sum(a for _, a in xs)
        m = se / len(xs)
        return (sa + w * m) / (se + w * m)

    return f


def loglog(done):
    """Ratio depending on estimate size: log-log regression per type."""
    by = collections.defaultdict(list)
    for ty, e, a in done:
        if e > 0 and a > 0:
            by[ty].append((math.log(e), math.log(a)))
    ar = arith(done)

    def per(ty, e):
        xs = by[ty]
        if len(xs) < 15:
            return e * ar(ty)
        mx = sum(x for x, _ in xs) / len(xs)
        my = sum(y for _, y in xs) / len(xs)
        vx = sum((x - mx) ** 2 for x, _ in xs)
        b = sum((x - mx) * (y - my) for x, y in xs) / vx if vx > 1e-9 else 1.0
        b = max(0.5, min(1.5, b))
        resid = [y - (my + b * (x - mx)) for x, y in xs]
        s2 = sum(r * r for r in resid) / max(1, len(resid) - 2)
        return math.exp(my + b * (math.log(e) - mx) + s2 / 2)  # mean of lognormal

    return per


def print_stats(res, names):
    for name in names:
        for q in QUANTILES:
            r = res[(name, q)]
            print(f"{name:14} {int(q * 100):2}%: медиана {sorted(r)[len(r) // 2]:.2f}  мин {min(r):.2f}  макс {max(r):.2f}")


def main():
    variants = {"сейчас: геом.": geo, "ариф.": arith, "ариф.+общий": lambda d: arith(d, pooled=True)}
    res = collections.defaultdict(list)
    for run in sys.argv[1:]:
        s, complete = seq(run)
        if not complete or len(s) < 40:
            print(f"{run}: skip ({len(s)} tasks, complete={complete})")
            continue
        row = []
        for q in QUANTILES:
            k = int(q * len(s))
            done, rest = s[:k], s[k:]
            actual = sum(a for _, _, a in rest)
            cells = []
            for name, mk in variants.items():
                f = mk(done)
                fc = sum(e * f(ty) for ty, e, _ in rest)
                res[(name, q)].append(fc / actual)
                cells.append(f"{fc / actual:4.2f}")
            row.append("/".join(cells))
            f = recent(done)
            res[("окно 30", q)].append(sum(e * f(ty) for ty, e, _ in rest) / actual)
            g = loglog(done)
            res[("лог-лог", q)].append(sum(g(ty, e) for ty, e, _ in rest) / actual)
        print(f"{run:18} {len(s):3} задач  прогноз/факт (геом/ариф/ариф+общий) при 10%,25%,50%,75%: " + "  ".join(row))
    print()
    print_stats(res, variants)
    print()
    print_stats(res, ("окно 30", "лог-лог"))


if __name__ == "__main__":
    main()
