import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

FAKE_COLOR = "#E76F51"
REAL_COLOR = "#2A9D8F"
DPI = 150
STYLE = "whitegrid"

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "it", "its", "this", "that", "these", "those", "i", "we",
    "you", "he", "she", "they", "his", "her", "their", "our", "my", "as",
    "by", "from", "not", "no", "so", "if", "than", "then", "up", "out",
    "about", "into", "through", "during", "before", "after", "over", "under",
    "again", "further", "once", "said", "also", "new", "one", "two", "more",
    "can", "just", "all", "who", "what", "when", "which", "there",
}


def _save(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")


def plot_class_distribution(df, save_path=None):
    sns.set_style(STYLE)
    counts = df["label"].value_counts().sort_index()
    labels = ["Fake (0)", "Real (1)"]
    colors = [FAKE_COLOR, REAL_COLOR]
    pcts = counts / counts.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.2)
    for i, (cnt, pct) in enumerate(zip(counts.values, pcts.values)):
        axes[0].text(i, cnt + 200, f"{cnt:,}\n({pct:.1f}%)", ha="center", fontsize=11)
    axes[0].set_title("Class Distribution — Count", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Number of Articles")
    axes[0].set_ylim(0, counts.max() * 1.15)

    axes[1].pie(
        counts.values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 11},
    )
    axes[1].set_title("Class Distribution — Proportion", fontsize=13, fontweight="bold")

    fig.suptitle("WELFake — Label Distribution", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_wordcount_histogram(df, save_path=None):
    sns.set_style(STYLE)
    if "word_count" not in df.columns:
        df = df.copy()
        df["word_count"] = df["text"].fillna("").str.split().str.len()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    for ax, (lbl, name, color) in zip(
        axes, [(0, "Fake", FAKE_COLOR), (1, "Real", REAL_COLOR)]
    ):
        data = df[df["label"] == lbl]["word_count"]
        ax.hist(data, bins=80, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
        ax.axvline(data.mean(), color="black", linestyle="--", linewidth=1.2,
                   label=f"Mean: {data.mean():.0f}")
        ax.axvline(data.median(), color="grey", linestyle=":", linewidth=1.2,
                   label=f"Median: {data.median():.0f}")
        ax.set_title(f"{name} Articles — Word Count Distribution", fontsize=12, fontweight="bold")
        ax.set_xlabel("Word Count")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=10)
        ax.set_xlim(0, min(data.quantile(0.99), 3000))

    fig.suptitle("Word Count Distribution by Class", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_length_boxplot(df, save_path=None):
    sns.set_style(STYLE)
    df2 = df.copy()
    df2["word_count"] = df2["text"].fillna("").str.split().str.len()
    df2["title_word_count"] = df2["title"].fillna("").str.split().str.len()
    df2["Class"] = df2["label"].map({0: "Fake", 1: "Real"})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    palette = {"Fake": FAKE_COLOR, "Real": REAL_COLOR}

    sns.boxplot(
        data=df2[df2["word_count"] < df2["word_count"].quantile(0.99)],
        x="Class", y="word_count", palette=palette, ax=axes[0],
        linewidth=1.2, flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
    )
    axes[0].set_title("Text Word Count by Class", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Word Count")

    sns.boxplot(
        data=df2,
        x="Class", y="title_word_count", palette=palette, ax=axes[1],
        linewidth=1.2, flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
    )
    axes[1].set_title("Title Word Count by Class", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Word Count")

    fig.suptitle("Length Distribution — Fake vs Real", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_missing_values(df, save_path=None):
    sns.set_style(STYLE)
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    cols = null_counts.index.tolist()
    colors = ["#BDBDBD" if v == 0 else FAKE_COLOR for v in null_counts.values]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(cols, null_counts.values, color=colors, edgecolor="white", linewidth=1.0)
    for i, (cnt, pct) in enumerate(zip(null_counts.values, null_pct.values)):
        axes[0].text(i, cnt + 5, f"{cnt:,}", ha="center", fontsize=10)
    axes[0].set_title("Missing Value Counts per Column", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Null Count")
    axes[0].set_xlabel("Column")

    axes[1].bar(cols, null_pct.values, color=colors, edgecolor="white", linewidth=1.0)
    for i, pct in enumerate(null_pct.values):
        axes[1].text(i, pct + 0.02, f"{pct:.2f}%", ha="center", fontsize=10)
    axes[1].set_title("Missing Value Percentage per Column", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Null %")
    axes[1].set_xlabel("Column")

    fig.suptitle("Missing Values Overview", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_top_words(df, save_path=None, top_n=20):
    sns.set_style(STYLE)

    def get_top_words(subset):
        text = " ".join(subset.fillna("").astype(str).str.lower())
        words = re.findall(r"\b[a-z]{3,}\b", text)
        filtered = [w for w in words if w not in STOPWORDS]
        return Counter(filtered).most_common(top_n)

    fake_words = get_top_words(
        df[df["label"] == 0]["title"].fillna("") + " " + df[df["label"] == 0]["text"].fillna("")
    )
    real_words = get_top_words(
        df[df["label"] == 1]["title"].fillna("") + " " + df[df["label"] == 1]["text"].fillna("")
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, words, name, color in [
        (axes[0], fake_words, "Fake", FAKE_COLOR),
        (axes[1], real_words, "Real", REAL_COLOR),
    ]:
        terms, counts = zip(*words)
        y_pos = np.arange(len(terms))
        ax.barh(y_pos, counts, color=color, edgecolor="white", linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(terms, fontsize=10)
        ax.invert_yaxis()
        ax.set_title(f"Top {top_n} Words — {name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Frequency")

    fig.suptitle("Most Frequent Words by Class", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_missingness_heatmap(df, save_path=None):
    sns.set_style(STYLE)
    df2 = df.copy()
    df2["Class"] = df2["label"].map({0: "Fake", 1: "Real"})

    # --- Panel 1: missingno-style heatmap (sample 500 rows) ---
    sample = df2.drop(columns=["Class"]).sample(n=min(500, len(df2)), random_state=42)
    miss_matrix = sample.isnull().astype(int)

    # --- Panel 2: missingness by class ---
    miss_by_class = (
        df2.groupby("Class")[["title", "text", "label"]]
        .apply(lambda g: g.isnull().sum())
        .T
    )

    # --- Panel 3: co-occurrence heatmap (which columns are missing together) ---
    miss_cols = df2[["title", "text"]].isnull()
    cooccurrence = miss_cols.T.dot(miss_cols).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1 — row-level missingness pattern
    sns.heatmap(
        miss_matrix,
        ax=axes[0],
        cbar=True,
        cmap=sns.color_palette(["#F0F0F0", FAKE_COLOR], as_cmap=True),
        yticklabels=False,
        linewidths=0,
        cbar_kws={"label": "Missing (1) / Present (0)"},
    )
    axes[0].set_title("Missingness Pattern\n(500-row sample)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row index (sampled)")

    # Panel 2 — missing count per class
    miss_by_class.plot(
        kind="bar",
        ax=axes[1],
        color=[FAKE_COLOR, REAL_COLOR],
        edgecolor="white",
        linewidth=0.8,
    )
    axes[1].set_title("Missing Values per Column\nby Class", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Null Count")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    axes[1].legend(title="Class")
    for container in axes[1].containers:
        axes[1].bar_label(container, fontsize=9, padding=2)

    # Panel 3 — co-occurrence
    sns.heatmap(
        cooccurrence,
        ax=axes[2],
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Co-missing count"},
    )
    axes[2].set_title("Missingness Co-occurrence\n(columns missing together)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Column")
    axes[2].set_ylabel("Column")

    fig.suptitle("Missingness Analysis", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_before_after_dedup(rows_before, rows_after, save_path=None):
    """
    Horizontal bar chart showing dataset size before and after deduplication.

    Parameters
    ----------
    rows_before : int
        Row count before deduplication.
    rows_after : int
        Row count after deduplication.
    save_path : str, optional
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    sns.set_style(STYLE)
    removed = rows_before - rows_after
    pct_removed = removed / rows_before * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(
        ["Before Deduplication", "After Deduplication"],
        [rows_before, rows_after],
        color=[FAKE_COLOR, REAL_COLOR],
        edgecolor="white",
        height=0.45,
    )
    for bar, val in zip(bars, [rows_before, rows_after]):
        ax.text(
            bar.get_width() + 300, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", ha="left", fontsize=11, fontweight="bold",
        )
    ax.annotate(
        f"Removed {removed:,} duplicate rows ({pct_removed:.1f}%)",
        xy=(rows_after, 0), xytext=(rows_after * 0.5, 0.6),
        fontsize=11, color=FAKE_COLOR, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=FAKE_COLOR, lw=1.4),
    )
    ax.set_xlabel("Number of Articles", fontsize=12)
    ax.set_title("Dataset Size Before and After Deduplication", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlim(0, rows_before * 1.12)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_length_after_cleaning(df, save_path=None):
    """
    Two-panel figure showing word count distribution after cleaning.
    Left: overlapping histogram by class. Right: boxplot by class.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with word_count and label columns.
    save_path : str, optional
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    sns.set_style(STYLE)
    if "word_count" not in df.columns:
        df = df.copy()
        df["word_count"] = df["content"].fillna("").str.split().str.len()

    clip_max = int(df["word_count"].quantile(0.99))
    df2 = df.copy()
    df2["Class"] = df2["label"].map({0: "Fake", 1: "Real"})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for lbl, name, color in [(0, "Fake", FAKE_COLOR), (1, "Real", REAL_COLOR)]:
        data = df2[df2["label"] == lbl]["word_count"].clip(upper=clip_max)
        axes[0].hist(data, bins=60, color=color, alpha=0.6, label=name, edgecolor="none")
        axes[0].axvline(data.mean(), color=color, linestyle="--", linewidth=1.5,
                        label=f"{name} mean: {data.mean():.0f}")

    axes[0].set_xlabel("Word Count", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("Word Count Distribution After Cleaning", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)

    palette = {"Fake": FAKE_COLOR, "Real": REAL_COLOR}
    plot_data = df2[df2["word_count"] <= clip_max]
    sns.boxplot(
        data=plot_data, x="Class", y="word_count",
        palette=palette, ax=axes[1], linewidth=1.2,
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
    )
    axes[1].set_title("Word Count Spread by Class", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Class", fontsize=12)
    axes[1].set_ylabel("Word Count", fontsize=12)

    fig.suptitle("Text Length Analysis After Cleaning", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_split_distribution(train_df, val_df, test_df, save_path=None):
    """
    Stacked horizontal bar chart showing class distribution across splits.

    Parameters
    ----------
    train_df : pd.DataFrame
    val_df : pd.DataFrame
    test_df : pd.DataFrame
    save_path : str, optional
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    sns.set_style(STYLE)
    splits = {"Train": train_df, "Validation": val_df, "Test": test_df}

    fake_counts = [df[df["label"] == 0].shape[0] for df in splits.values()]
    real_counts = [df[df["label"] == 1].shape[0] for df in splits.values()]
    totals      = [f + r for f, r in zip(fake_counts, real_counts)]
    labels      = list(splits.keys())

    fig, ax = plt.subplots(figsize=(11, 4))
    y = np.arange(len(labels))
    bar_h = 0.45

    bars_f = ax.barh(y, fake_counts, height=bar_h, color=FAKE_COLOR,
                     label="Fake (0)", edgecolor="white")
    bars_r = ax.barh(y, real_counts, height=bar_h, left=fake_counts,
                     color=REAL_COLOR, label="Real (1)", edgecolor="white")

    # Labels inside segments
    for bar, val in zip(bars_f, fake_counts):
        if val > 500:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                    f"{val:,}", ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")

    for bar, val, fc in zip(bars_r, real_counts, fake_counts):
        if val > 500:
            ax.text(fc + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                    f"{val:,}", ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")

    # Total and percentage outside bar
    for i, total in enumerate(totals):
        pct_fake = fake_counts[i] / total * 100
        pct_real = real_counts[i] / total * 100
        ax.text(total + 200, i,
                f"Total: {total:,}  ({pct_fake:.1f}% F / {pct_real:.1f}% R)",
                va="center", ha="left", fontsize=9, color="#444")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Number of Articles", fontsize=12)
    ax.set_title("Train / Validation / Test Split Distribution", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlim(0, max(totals) * 1.28)
    ax.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_wordclouds(df, save_path=None):
    from wordcloud import WordCloud  # type: ignore

    def get_text(subset_df):
        return " ".join(
            (subset_df["title"].fillna("") + " " + subset_df["text"].fillna(""))
            .astype(str)
            .str.lower()
        )

    fake_text = get_text(df[df["label"] == 0])
    real_text = get_text(df[df["label"] == 1])

    wc_kwargs = dict(
        width=800, height=400, background_color="white",
        stopwords=STOPWORDS, max_words=150, collocations=False,
    )

    wc_fake = WordCloud(colormap="Reds", **wc_kwargs).generate(fake_text)
    wc_real = WordCloud(colormap="Greens", **wc_kwargs).generate(real_text)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].imshow(wc_fake, interpolation="bilinear")
    axes[0].axis("off")
    axes[0].set_title("Fake Articles — Word Cloud", fontsize=13, fontweight="bold", pad=12)

    axes[1].imshow(wc_real, interpolation="bilinear")
    axes[1].axis("off")
    axes[1].set_title("Real Articles — Word Cloud", fontsize=13, fontweight="bold", pad=12)

    fig.suptitle("Word Clouds by Class", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, save_path)
    return fig
