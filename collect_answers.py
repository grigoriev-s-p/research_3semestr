from datasets import load_dataset
from openai import OpenAI
import pandas as pd
import csv
SIZE_OF_DATA = 15

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="11211"
)

def process_response(raw_response):
    response = raw_response.strip()

    if response == "ABSTAIN":
        return "ABSTAIN"

    if response.startswith("ANSWER:"):
        answer = response[len("ANSWER:"):].strip()

        if answer:
            return answer

    return "INVALID_FORMAT"
def is_abstain(response):
    return 1 if response == "ABSTAIN" else 0
def prompt_1(context, question):
    prompt = f"""Answer the question using only the provided context.

Do not use any outside knowledge.
If the context contains enough information to answer
the question, return a short answer extracted from
the context.

If the context does not contain enough information,
return ABSTAIN.

Output exactly one of the following formats:
ANSWER: <short answer>
ABSTAIN

Do not provide explanations or additional text.

Context:
{context}

Question:
{question}
"""
    return prompt

def prompt_2(context, question):
    prompt = f"""Answer the question using only the provided context.

Before answering, determine whether the context
contains enough information to answer the question.

Do not use any outside knowledge.
If the context is sufficient, return a short answer
extracted from the context.

If the context is insufficient, return ABSTAIN.

Output exactly one of the following formats:
ANSWER: <short answer>
ABSTAIN

Do not output the sufficiency assessment,
explanations, or any additional text.

Context:
{context}

Question:
{question}
"""
    return prompt

def prompt_3(context, question):
    prompt = f"""Determine whether the provided context contains
enough information to answer the question.

Use only the provided context.
Do not rely on outside knowledge.

Return exactly one label:

SUFFICIENT - if the context contains enough
information to answer the question.

INSUFFICIENT - if the context does not contain
enough information to answer the question.

Do not answer the question.
Do not provide explanations or additional text.

Context:
{context}

Question:
{question}
"""
    return prompt

def request_to_LLM(prompt):
    request = client.chat.completions.create(
        model="qwen3.5-9b", 
        messages=[  
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
    )
    response = request.choices[0].message.content
    return response


data = load_dataset("rajpurkar/squad_v2", split = "train").shuffle(seed = 42)
data_with_answer = data.filter(lambda example: len(example['answers']['text']) != 0)
data_without_answer =  data.filter(lambda example: len(example['answers']['text']) == 0)




answers = []

for index in range(SIZE_OF_DATA):
    example_with = data_with_answer[index]

    id = example_with['id']
    context = example_with['context']
    question = example_with['question']

    request_1 = request_to_LLM(prompt_1(context, question))
    processed_response = process_response(request_1)
    answers.append({"id": id, "scheme":1, "response": request_1, "processed_response": processed_response,
                "abstain":is_abstain(processed_response), "isAnswerable": 1})

    request_2 = request_to_LLM(prompt_2(context, question))
    processed_response = process_response(request_2)
    answers.append({"id": id, "scheme":2, "response": request_2, "processed_response": processed_response,
                    "abstain":is_abstain(processed_response), "isAnswerable": 1})

    request_3 = request_to_LLM(prompt_3(context, question))
    if request_3 == "SUFFICIENT":
        request = request_to_LLM(prompt_1(context, question))
        processed_response = process_response(request)
        answers.append({"id": id, "scheme":3, "response": request, "processed_response": processed_response,
                        "abstain": is_abstain(processed_response), "isAnswerable": 1,"sufficiency_raw": request_3})
    elif request_3 == "INSUFFICIENT":
        answers.append({"id": id, "scheme":3, "response": request_3, "processed_response": "None",
                        "abstain": 1, "isAnswerable": 1,"sufficiency_raw": request_3})
    else:
        answers.append({"id": id, "scheme":3, "response": request_3, "processed_response": process_response(request_3),
                        "abstain": 0, "isAnswerable": 1,"sufficiency_raw": request_3})

    # ==================================================================================================================

    id = data_without_answer[index]['id']
    context = data_without_answer[index]['context']
    question = data_without_answer[index]['question']

    request_1 = request_to_LLM(prompt_1(context, question))
    processed_response = process_response(request_1)
    answers.append({"id": id, "scheme":1, "response": request_1, "processed_response": processed_response,
                    "abstain":is_abstain(processed_response), "isAnswerable": 0})

    request_2 = request_to_LLM(prompt_2(context, question))
    processed_response = process_response(request_2)
    answers.append({"id": id, "scheme":2, "response": request_2, "processed_response": processed_response,
                    "abstain":is_abstain(processed_response), "isAnswerable": 0})

    request_3 = request_to_LLM(prompt_3(context, question))
    if request_3 == "SUFFICIENT":
        request = request_to_LLM(prompt_1(context, question))
        processed_response = process_response(request)
        answers.append({"id": id, "scheme":3, "response": request, "processed_response": processed_response,
                        "abstain": is_abstain(processed_response), "isAnswerable": 0, "sufficiency_raw": request_3})
    elif request_3 == "INSUFFICIENT":
        answers.append({"id": id, "scheme":3, "response": request_3, "processed_response": "None",
                        "abstain": 1, "isAnswerable": 0,"sufficiency_raw": request_3})
    else:
        answers.append({"id": id, "scheme":3, "response": request_3, "processed_response": process_response(request_3),
                        "abstain": 0, "isAnswerable": 0,"sufficiency_raw": request_3})

df = pd.DataFrame(answers)
df.to_csv("answers.csv")