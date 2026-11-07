import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


with_answer = pd.read_csv("results_with_answer.csv", index_col=0)
without_answer = pd.read_csv("results_without_answer.csv", index_col=0)
precheck = pd.read_csv("scheme3_sufficient_rate.csv", index_col=0)

schemes = ["Scheme 1", "Scheme 2", "Scheme 3"]


def plot_outcomes(table, columns, labels, title, filename):
    table = table.loc[schemes]
    fig, ax = plt.subplots(figsize=(9, 5))
    bottoms = [0.0] * len(schemes)

    for column, label in zip(columns, labels):
        values = (table[column] * 100).tolist()
        bars = ax.bar(schemes, values, bottom=bottoms, label=label)

        for bar, value, lower in zip(bars, values, bottoms):
            if value >= 8:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    lower + value / 2,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=10
                )

        bottoms = [
            bottom + value
            for bottom, value in zip(bottoms, values)
        ]

    ax.set_ylim(0, 105)
    ax.set_ylabel("Доля вопросов, %")
    ax.set_title(title)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2
    )
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


plot_outcomes(
    with_answer,
    [
        "correct_answer_rate",
        "wrong_answer_rate",
        "false_abstain",
        "invalid_format_rate"
    ],
    [
        "Правильный ответ",
        "Неправильный ответ",
        "Ошибочный отказ",
        "Неверный формат"
    ],
    "Вопросы с ответом: результаты трёх схем",
    "01_with_answer.png"
)

plot_outcomes(
    without_answer,
    [
        "correct_abstention_rate",
        "unsupported_answer_rate",
        "invalid_format_rate"
    ],
    [
        "Правильный отказ",
        "Необоснованный ответ",
        "Неверный формат"
    ],
    "Вопросы без ответа: результаты трёх схем",
    "02_without_answer.png"
)


def percent_table(table):
    result = table.loc[schemes].copy()

    for col in result.columns:
        if col != "invalid_format_count":
            result[col] = result[col] * 100

    return result.round(2).to_string()


def difference(table, metric, base, other="Scheme 3"):
    return 100 * (
        table.loc[other, metric] - table.loc[base, metric]
    )


lines = [
    "РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА",
    "",
    "1. Вопросы с ответом (доли в %; invalid_format_count — количество):",
    percent_table(with_answer),
    "",
    "2. Вопросы без ответа (доли в %):",
    percent_table(without_answer),
    "",
    "3. Предварительная проверка третьей схемы:",
    precheck.to_string(),
    "",
    "4. Разница двухэтапной схемы с другими (процентные пункты):"
]

for base in schemes[:2]:
    lines.extend([
        f"Scheme 3 минус {base}:",
        f"  Необоснованные ответы: {difference(without_answer, 'unsupported_answer_rate', base):+.2f} п.п.",
        f"  Ошибочные отказы: {difference(with_answer, 'false_abstain', base):+.2f} п.п.",
        f"  Правильные ответы: {difference(with_answer, 'correct_answer_rate', base):+.2f} п.п."
    ])


report = "\n".join(lines)

print(report)
