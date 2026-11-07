
import pandas as pd
import evaluate
from datasets import load_dataset

metric = evaluate.load("squad_v2")
data = load_dataset("rajpurkar/squad_v2", split="train")
data = {example["id"]: example for example in data}

def calculate_metrics(row):
    if row["invalid_format"] == 1:
        return pd.Series({"em": float("nan"), "f1": float("nan")})

    id = row["id"]
    prediction = row["processed_response"]

    if row["abstain"] == 1:
        prediction = ""

    result = metric.compute(
        predictions=[{
            "id": id,
            "prediction_text": prediction,
            "no_answer_probability": 0.0
        }],
        references=[{
            "id": id,
            "answers": data[id]["answers"]
        }]
    )

    return pd.Series({
        "em": result["exact"] / 100,
        "f1": result["f1"] / 100
    })


answers = pd.read_csv('answers.csv', index_col="Unnamed: 0")


answers["invalid_format"] = (
    (answers["abstain"] != 1) &
    (
        answers["processed_response"].isna() |
        answers["processed_response"].astype(str).str.strip().isin(
            ["INVALID_FORMAT", "None", ""]
        )
    )
).astype(int)

answers[["em", "f1"]] = answers.apply(calculate_metrics, axis=1)

n_1 = len(answers[
    (answers.scheme == 1) & (answers.isAnswerable == 1)
])
n_2 = len(answers[
    (answers.scheme == 2) & (answers.isAnswerable == 1)
])
n_3 = len(answers[
    (answers.scheme == 3) & (answers.isAnswerable == 1)
])

correct_1 = 0
uncorrect_1 = 0
uncorrect_abstain_1 = 0
invalid_1 = 0
em_1 = 0
f1_1 = 0

correct_2 = 0
uncorrect_2 = 0
uncorrect_abstain_2 = 0
invalid_2 = 0
em_2 = 0
f1_2 = 0

correct_3 = 0
uncorrect_3 = 0
uncorrect_abstain_3 = 0
invalid_3 = 0
em_3 = 0
f1_3 = 0

for i in range(len(answers)):
    answer = answers.iloc[i]

    if answer.isAnswerable == 1:
        if answer.scheme == 1:
            if answer.invalid_format == 1:
                invalid_1 += 1
            elif answer.em == 1:
                correct_1 += 1
            elif answer.abstain == 1:
                uncorrect_abstain_1 += 1
            else:
                uncorrect_1 += 1

            if answer.invalid_format == 0:
                em_1 += answer.em
                f1_1 += answer.f1

        if answer.scheme == 2:
            if answer.invalid_format == 1:
                invalid_2 += 1
            elif answer.em == 1:
                correct_2 += 1
            elif answer.abstain == 1:
                uncorrect_abstain_2 += 1
            else:
                uncorrect_2 += 1

            if answer.invalid_format == 0:
                em_2 += answer.em
                f1_2 += answer.f1

        if answer.scheme == 3:
            if answer.invalid_format == 1:
                invalid_3 += 1
            elif answer.em == 1:
                correct_3 += 1
            elif answer.abstain == 1:
                uncorrect_abstain_3 += 1
            else:
                uncorrect_3 += 1

            if answer.invalid_format == 0:
                em_3 += answer.em
                f1_3 += answer.f1


scheme_1_with_answer = {
    "correct_answer_rate": correct_1 / n_1,
    "wrong_answer_rate": uncorrect_1 / n_1,
    "false_abstain": uncorrect_abstain_1 / n_1,
    "invalid_format_rate": invalid_1 / n_1,
    "invalid_format_count": invalid_1,
    "EM": em_1 / (n_1 - invalid_1) if n_1 > invalid_1 else float("nan"),
    "F1": f1_1 / (n_1 - invalid_1) if n_1 > invalid_1 else float("nan")
}

scheme_2_with_answer = {
    "correct_answer_rate": correct_2 / n_2,
    "wrong_answer_rate": uncorrect_2 / n_2,
    "false_abstain": uncorrect_abstain_2 / n_2,
    "invalid_format_rate": invalid_2 / n_2,
    "invalid_format_count": invalid_2,
    "EM": em_2 / (n_2 - invalid_2) if n_2 > invalid_2 else float("nan"),
    "F1": f1_2 / (n_2 - invalid_2) if n_2 > invalid_2 else float("nan")
}

scheme_3_with_answer = {
    "correct_answer_rate": correct_3 / n_3,
    "wrong_answer_rate": uncorrect_3 / n_3,
    "false_abstain": uncorrect_abstain_3 / n_3,
    "invalid_format_rate": invalid_3 / n_3,
    "invalid_format_count": invalid_3,
    "EM": em_3 / (n_3 - invalid_3) if n_3 > invalid_3 else float("nan"),
    "F1": f1_3 / (n_3 - invalid_3) if n_3 > invalid_3 else float("nan")
}

table_1 = pd.DataFrame(
    [scheme_1_with_answer, scheme_2_with_answer, scheme_3_with_answer],
    index=["Scheme 1", "Scheme 2", "Scheme 3"]
)

correct_abstent_1 = 0
wrong_abstent_1 = 0
invalid_without_1 = 0

correct_abstent_2 = 0
wrong_abstent_2 = 0
invalid_without_2 = 0

correct_abstent_3 = 0
wrong_abstent_3 = 0
invalid_without_3 = 0

for i in range(len(answers)):
    answer = answers.iloc[i]

    if answer.isAnswerable == 0:
        if answer.scheme == 1:
            if answer.invalid_format == 1:
                invalid_without_1 += 1
            elif answer.abstain == 1:
                correct_abstent_1 += 1
            else:
                wrong_abstent_1 += 1

        if answer.scheme == 2:
            if answer.invalid_format == 1:
                invalid_without_2 += 1
            elif answer.abstain == 1:
                correct_abstent_2 += 1
            else:
                wrong_abstent_2 += 1

        if answer.scheme == 3:
            if answer.invalid_format == 1:
                invalid_without_3 += 1
            elif answer.abstain == 1:
                correct_abstent_3 += 1
            else:
                wrong_abstent_3 += 1



n_without_1 = len(answers[
    (answers.scheme == 1) & (answers.isAnswerable == 0)
])

n_without_2 = len(answers[
    (answers.scheme == 2) & (answers.isAnswerable == 0)
])

n_without_3 = len(answers[
    (answers.scheme == 3) & (answers.isAnswerable == 0)
])


scheme_1_without_answer = {
    "correct_abstention_rate": correct_abstent_1 / n_without_1,
    "unsupported_answer_rate": wrong_abstent_1 / n_without_1,
    "invalid_format_rate": invalid_without_1 / n_without_1
}

scheme_2_without_answer = {
    "correct_abstention_rate": correct_abstent_2 / n_without_2,
    "unsupported_answer_rate": wrong_abstent_2 / n_without_2,
    "invalid_format_rate": invalid_without_2 / n_without_2
}

scheme_3_without_answer = {
    "correct_abstention_rate": correct_abstent_3 / n_without_3,
    "unsupported_answer_rate": wrong_abstent_3 / n_without_3,
    "invalid_format_rate": invalid_without_3 / n_without_3
}

table_2 = pd.DataFrame(
    [
        scheme_1_without_answer,
        scheme_2_without_answer,
        scheme_3_without_answer
    ],
    index=["Scheme 1", "Scheme 2", "Scheme 3"]
)
            
scheme_3 = answers[answers.scheme == 3].copy()
scheme_3["sufficiency_raw"] = (
    scheme_3["sufficiency_raw"]
    .fillna("")
    .astype(str)
    .str.strip()
)

with_answer = scheme_3[scheme_3.isAnswerable == 1]
without_answer = scheme_3[scheme_3.isAnswerable == 0]

n_with = len(with_answer)
n_without = len(without_answer)

correct_sufficient = (
    with_answer.sufficiency_raw == "SUFFICIENT"
).sum()

false_insufficient = (
    with_answer.sufficiency_raw == "INSUFFICIENT"
).sum()

invalid_with = (
    ~with_answer.sufficiency_raw.isin(
        ["SUFFICIENT", "INSUFFICIENT"]
    )
).sum()

correct_insufficient = (
    without_answer.sufficiency_raw == "INSUFFICIENT"
).sum()

false_sufficient = (
    without_answer.sufficiency_raw == "SUFFICIENT"
).sum()

invalid_without = (
    ~without_answer.sufficiency_raw.isin(
        ["SUFFICIENT", "INSUFFICIENT"]
    )
).sum()


table_3 = pd.DataFrame(
    {
        "SUFFICIENT": [
            correct_sufficient,
            false_sufficient
        ],
        "INSUFFICIENT": [
            false_insufficient,
            correct_insufficient
        ],
        "INVALID": [
            invalid_with,
            invalid_without
        ],
        "total": [
            n_with,
            n_without
        ],
        "correct_rate": [
            correct_sufficient / n_with,
            correct_insufficient / n_without
        ],
        "error_rate": [
            false_insufficient / n_with,
            false_sufficient / n_without
        ]
    },
    index=["With answer", "Without answer"]
)

table_1.to_csv("results_with_answer.csv")
table_2.to_csv("results_without_answer.csv")
table_3.to_csv("scheme3_sufficient_rate.csv")

answers.to_csv("evaluated_answers.csv", index=False)