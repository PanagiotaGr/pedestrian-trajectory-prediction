import os
import csv
import argparse
import matplotlib.pyplot as plt


LABEL_COLORS = {
    "road": "tab:gray",
    "sidewalk": "tab:pink",
    "ground": "tab:purple",
    "curb": "tab:brown",
    "road_line": "tab:olive",
    "crosswalk": "tab:orange",
    "bikelane": "tab:blue",
    "unknown": "tab:cyan",
}

CLASS_MARKERS = {
    "person": "o",
    "scooter": "^",
    "bicycle": "P",
    "motorcycle": "X",
    "stroller": "D",
    "car": "s",
    "truck": "s",
    "bus": "s",
    "unknown": "x",
}


def load_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def choose_rows(rows, sample_id=None):
    if sample_id is None:
        sample_id = rows[0]["sample_id"]
    selected = [r for r in rows if r["sample_id"] == sample_id]
    if not selected:
        raise ValueError(f"Δεν βρέθηκαν γραμμές για sample_id={sample_id}")
    return selected


def main():
    parser = argparse.ArgumentParser(description="Plot all agents relative to one pedestrian")
    parser.add_argument(
        "--csv",
        default=os.path.expanduser("~/imptc_project/results/pedestrian_math_with_map.csv"),
    )
    parser.add_argument("--sample-id", default="0004")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--show-velocity", action="store_true")
    parser.add_argument(
        "--out",
        default=os.path.expanduser("~/imptc_project/results/pedestrian_relative_view.png"),
    )
    args = parser.parse_args()

    rows = load_rows(args.csv)
    selected = choose_rows(rows, sample_id=args.sample_id)

    selected.sort(key=lambda r: safe_float(r.get("dist_xy", 1e9)))
    selected = selected[:args.top_k]

    fig, ax = plt.subplots(figsize=(9, 9))

    # target pedestrian στο κέντρο
    label_a = selected[0].get("label_a", "unknown")
    ax.scatter(
        [0], [0],
        s=260,
        marker="*",
        c=["gold"],
        edgecolors="black",
        linewidths=1.0,
        zorder=10,
        label=f"target pedestrian ({label_a})",
    )
    ax.text(0.15, 0.15, "A (target)", fontsize=12, weight="bold")

    used_legend = set()

    for r in selected:
        x = safe_float(r["rel_x_local"])
        y = safe_float(r["rel_y_local"])
        vx = safe_float(r.get("rel_vx_local", 0.0))
        vy = safe_float(r.get("rel_vy_local", 0.0))

        other_id = r.get("other_id", "?")
        other_class = r.get("other_class_name", "unknown") or "unknown"
        label_b = r.get("label_b", "unknown") or "unknown"

        color = LABEL_COLORS.get(label_b, "tab:cyan")
        marker = CLASS_MARKERS.get(other_class, "x")

        legend_key = (other_class, label_b)
        legend_label = None
        if legend_key not in used_legend:
            legend_label = f"{other_class} on {label_b}"
            used_legend.add(legend_key)

        ax.scatter(
            [x], [y],
            s=110,
            marker=marker,
            c=[color],
            edgecolors="black",
            linewidths=0.6,
            zorder=6,
            label=legend_label,
        )

        # πιο καθαρό label δίπλα
        txt = f'{other_id}\n{other_class}'
        ax.text(x + 0.08, y + 0.08, txt, fontsize=8)

        if args.show_velocity:
            ax.arrow(
                x, y, vx, vy,
                length_includes_head=True,
                head_width=0.15,
                head_length=0.22,
                alpha=0.75,
                zorder=5,
            )

    # local axes
    ax.axhline(0, linewidth=1.2, color="steelblue")
    ax.axvline(0, linewidth=1.2, color="steelblue")

    # κατευθύνσεις
    ax.annotate("front", xy=(2.0, 0.0), xytext=(3.0, 0.0),
                arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=11)
    ax.annotate("rear", xy=(-2.0, 0.0), xytext=(-3.3, 0.0),
                arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=11)
    ax.annotate("left", xy=(0.0, 2.0), xytext=(0.0, 3.2),
                arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=11)
    ax.annotate("right", xy=(0.0, -2.0), xytext=(0.0, -3.2),
                arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=11)

    ax.set_xlabel("local x relative to pedestrian A")
    ax.set_ylabel("local y relative to pedestrian A")
    ax.set_title(
        f'Where the other agents are relative to pedestrian A\n'
        f'sample {selected[0]["sample_id"]}, scene {selected[0]["scene_path"]}, target {selected[0]["target_id"]}'
    )
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out, dpi=220)
    print("Saved plot:", args.out)


if __name__ == "__main__":
    main()
