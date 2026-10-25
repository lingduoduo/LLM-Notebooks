"""Semicircular hierarchical sunburst of the room taxonomy's keyword coverage.

Rings are L1 category, L2 group and L3 concept; each wedge's angle is the number
of keyword rows in keywords.csv mapped to it. Keywords mapped directly to an
L1 category or L2 group get a "Category-level" / "Direct to ..." wedge so every
row is counted once. Angles stay proportional: a category too thin to label gets
an outside callout, and a category with no keywords (IDENTITY, which is
self-declared only) keeps its place as a dashed marker with a callout.

Install: pip install matplotlib numpy
Run: python semicircle_taxonomy.py [--output taxonomy_semicircle]
"""

import argparse
import math
from collections import Counter

from build import load, load_taxonomy

# Axis colors match the taxonomy map page (validated categorical slots 1-6).
AXIS_COLORS = {
    "content": "#2a78d6",
    "format": "#eb6834",
    "interaction": "#1baf7a",
    "identity": "#eda100",
    "commerce": "#e87ba4",
    "generic": "#008300",
}
INK = "#14171c"
EMPTY_NOTES = {"identity": "self-declared only"}


def taxonomy_rows(taxonomy, vocabulary):
    """Return (category, group, concept, count, axis) rows in taxonomy order."""
    counts = Counter(row["concept_id"] for row in vocabulary)
    nodes = taxonomy["nodes"]
    children = {}
    for node in nodes:
        children.setdefault(node["parent_id"], []).append(node)
    rows = []
    for category in (n for n in nodes if n["level"] == 1):
        name, axis = category["category_id"], category["axis"]
        if counts[category["id"]]:
            rows.append(
                (
                    name,
                    "Category-level",
                    f"Direct to {name}",
                    counts[category["id"]],
                    axis,
                )
            )
        for group in children.get(category["id"], []):
            if counts[group["id"]]:
                rows.append(
                    (
                        name,
                        group["label"],
                        f"Direct to {group['label']}",
                        counts[group["id"]],
                        axis,
                    )
                )
            for concept in children.get(group["id"], []):
                if counts[concept["id"]]:
                    rows.append(
                        (
                            name,
                            group["label"],
                            concept["label"],
                            counts[concept["id"]],
                            axis,
                        )
                    )
    return rows


def category_order(taxonomy):
    """Return [(category code, axis)] for every L1 category in taxonomy order."""
    return [(n["category_id"], n["axis"]) for n in taxonomy["nodes"] if n["level"] == 1]


def draw_taxonomy(rows, title, output_prefix="taxonomy_semicircle", categories=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch, Wedge

    tree = {}
    axes = {}
    for category, group, concept, count, axis in rows:
        if count <= 0:
            continue
        words = tree.setdefault(category, {}).setdefault(group, {})
        words[concept] = words.get(concept, 0) + count
        axes[category] = axis
    totals = {
        cat: sum(sum(words.values()) for words in groups.values())
        for cat, groups in tree.items()
    }
    grand_total = sum(totals.values())
    if grand_total == 0:
        raise ValueError("No positive counts to plot")
    if categories is None:
        categories = list(axes.items())

    fig, ax = plt.subplots(figsize=(18, 10), subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor("white")
    ring_width = 0.95
    inner_radius = 1.55
    max_radius = inner_radius + 3 * ring_width
    ax.set_xlim(-max_radius - 0.25, max_radius + 0.25)
    # Headroom above the arc for callouts.
    ax.set_ylim(-0.2, max_radius + 1.4)
    # Data units per point, to decide whether a label fits its wedge.
    origin, unit = ax.transData.transform([(0, 0), (1, 0)])
    units_per_pt = fig.dpi / 72 / (unit[0] - origin[0])
    # Start at 180° (left) and traverse clockwise to 0° (right).
    current = 180.0

    def segment(theta_start, theta_end, ring, color, label, fontsize=9, bold=False):
        radius = inner_radius + (ring + 1) * ring_width
        ax.add_patch(
            Wedge(
                (0, 0),
                radius,
                theta_end,
                theta_start,
                width=ring_width,
                facecolor=color,
                edgecolor="white",
                linewidth=1.5,
            )
        )
        sweep = theta_start - theta_end
        mid = (theta_start + theta_end) / 2
        theta = math.radians(mid)
        text_radius = inner_radius + (ring + 0.5) * ring_width
        arc = text_radius * math.radians(sweep)
        # Try the full label, then just its first line (e.g. the category code).
        for label in dict.fromkeys((label, label.split("\n")[0])):
            lines = label.split("\n")
            width = max(map(len, lines)) * fontsize * 0.6 * units_per_pt
            height = len(lines) * fontsize * 1.25 * units_per_pt
            if width <= arc * 0.92:
                # Tangential text, kept upright.
                rotation = mid - 90
                break
            if height <= arc and width <= ring_width * 0.95:
                # Narrow wedge: radial text along the wedge.
                rotation = mid
                break
        else:
            return False
        if rotation < -90:
            rotation += 180
        if rotation > 90:
            rotation -= 180
        ax.text(
            text_radius * math.cos(theta),
            text_radius * math.sin(theta),
            label,
            ha="center",
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            fontsize=fontsize,
            weight="bold" if bold else "normal",
            color=INK,
            clip_on=True,
        )
        return True

    def tint(base, white):
        return tuple(base * (1 - white) + np.ones(3) * white)

    callouts = []  # (angle, text)
    for cat, axis in categories:
        if cat not in tree:
            # Zero keywords: keep the category's place with a dashed marker.
            rad = math.radians(current)
            ax.plot(
                [inner_radius * math.cos(rad), max_radius * math.cos(rad)],
                [inner_radius * math.sin(rad), max_radius * math.sin(rad)],
                color=AXIS_COLORS[axis],
                linewidth=1.6,
                linestyle=(0, (3, 2)),
            )
            note = f" ({EMPTY_NOTES[axis]})" if axis in EMPTY_NOTES else ""
            callouts.append((current, f"{cat} · 0 keywords{note}"))
            continue
        groups = tree[cat]
        base = np.array(plt.matplotlib.colors.to_rgb(AXIS_COLORS[axis]))
        cat_end = current - 180 * totals[cat] / grand_total
        placed = segment(
            current,
            cat_end,
            0,
            tint(base, 0.15),
            f"{cat}\n{totals[cat] / grand_total:.0%}",
            10,
            True,
        )
        if not placed:
            n = totals[cat]
            callouts.append(
                ((current + cat_end) / 2, f"{cat} · {n} keyword{'' if n == 1 else 's'}")
            )
        group_start = current
        for group, words in groups.items():
            group_end = group_start - 180 * sum(words.values()) / grand_total
            segment(group_start, group_end, 1, tint(base, 0.45), group, 9, True)
            word_start = group_start
            for concept, count in words.items():
                word_end = word_start - 180 * count / grand_total
                segment(word_start, word_end, 2, tint(base, 0.72), concept, 8)
                word_start = word_end
            group_start = group_end
        current = cat_end

    # Callouts: one label column per side, leftmost anchor gets the top label so
    # leader lines fan out without crossing.
    for side in (1, -1):
        items = sorted(
            (c for c in callouts if (c[0] < 90) == (side == 1)),
            key=lambda c: -c[0] * side,
        )
        if not items:
            continue
        anchors = [
            (
                max_radius * math.cos(math.radians(a)),
                max_radius * math.sin(math.radians(a)),
            )
            for a, _ in items
        ]
        column = (
            max(x for x, _ in anchors) + 0.6
            if side == 1
            else min(x for x, _ in anchors) - 0.6
        )
        for i, ((x, y), (_, text)) in enumerate(zip(anchors, items)):
            label_y = max_radius + 0.3 + (len(items) - 1 - i) * 0.32
            ax.plot([x, column], [y, label_y], color=INK, linewidth=0.8)
            ax.plot([x], [y], marker="o", markersize=3, color=INK)
            ax.text(
                column + 0.05 * side,
                label_y,
                text,
                ha="left" if side == 1 else "right",
                va="center",
                fontsize=9,
                color=INK,
            )

    ax.axis("off")
    ax.set_title(title, fontsize=20, pad=22, color=INK)
    ax.text(
        0,
        0.35,
        f"{grand_total:,}\nkeyword rows",
        ha="center",
        va="center",
        fontsize=13,
        weight="bold",
        color=INK,
    )
    shown = [a for a in AXIS_COLORS if a in {axis for _, axis in categories}]
    ax.legend(
        handles=[Patch(color=AXIS_COLORS[a], label=f"{a} axis") for a in shown],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=len(shown),
        frameon=False,
    )
    plt.tight_layout()
    outputs = []
    for ext in ("png", "svg"):
        path = f"{output_prefix}.{ext}"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    plt.close(fig)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="taxonomy_semicircle")
    args = parser.parse_args()
    taxonomy = load_taxonomy()
    _, vocabulary = load()
    paths = draw_taxonomy(
        taxonomy_rows(taxonomy, vocabulary),
        f"Room Taxonomy Keyword Coverage (v{taxonomy['version']})",
        args.output,
        category_order(taxonomy),
    )
    print("Created", " and ".join(paths))
