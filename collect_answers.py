from datasets import load_dataset
from openai import OpenAI
import pandas as pd
import time
SIZE_OF_DATA = 15

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="11211"
)

def save_to_checkpoint(records):
    df = pd.DataFrame(records)
    df.to_csv("answers_checkpoint.csv", index = False)

def process_response(raw_response):
    response = raw_response.strip()

    if response == "ABSTAIN":
        return "ABSTAIN"

    if response.startswith("ANSWER:"):
        answer = response[len("ANSWER:"):].strip()

        if answer:
            return answer

    return "INVALID_FORMAT"

def process_response_2(raw_response):
    lines = raw_response.strip().splitlines()
    lines = [line.strip() for line in lines if line.strip()]

    if len(lines) != 2:
        return "INVALID_FORMAT", "INVALID_FORMAT"

    if lines[0] == "SUFFICIENCY: SUFFICIENT":
        sufficiency_raw = "SUFFICIENT"
    elif lines[0] == "SUFFICIENCY: INSUFFICIENT":
        sufficiency_raw = "INSUFFICIENT"
    else:
        sufficiency_raw = "INVALID_FORMAT"

    processed_response = process_response(lines[1])

    return sufficiency_raw, processed_response



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

First, determine whether the context contains enough
information to answer the question.

Do not use any outside knowledge.
If the context contains sufficient information,
provide a short answer extracted from the context.

If the context does not contain sufficient information,
abstain from answering.

Output exactly two lines in one of the following formats:

SUFFICIENCY: SUFFICIENT
ANSWER: <short answer>

OR

SUFFICIENCY: INSUFFICIENT
ABSTAIN

The first line must indicate your assessment of
context sufficiency.

The second line must contain either a short answer
or ABSTAIN, depending on your assessment.

Do not provide explanations or any additional text.

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
    for attempt in range(3):
        try:
            request = client.chat.completions.create(
                model="qwen3.5-9b", 
                messages=[  
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_retries = 0
            )
            response = request.choices[0].message.content
            return response
        except Exception as error:
            print(error)
            if attempt < 2:
                time.sleep(2)
            else:
                raise


data = load_dataset("rajpurkar/squad_v2", split = "valid").shuffle(seed = 42)
data_with_answer = data.filter(lambda example: len(example['answers']['text']) != 0)
data_without_answer =  data.filter(lambda example: len(example['answers']['text']) == 0)




answers = []

for index in range(SIZE_OF_DATA):
    example_with = data_with_answer[index]

    id = example_with['id']
    context = example_with['context']
    question = example_with['question']

    # ==================== ВОПРОСЫ С ОТВЕТОМ ====================

    # Схема 1
    try:
        request_1 = request_to_LLM(prompt_1(context, question))
    except Exception as error:
        print(error)
        answers.append({
            "id": id, "scheme": 1,
            "response": None, "processed_response": None,
            "abstain": None, "isAnswerable": 1,
            "request_status": "error",
            "error_stage": "answer_generation",
            "error": str(error)
        })
    else:
        processed_response = process_response(request_1 or "")
        answers.append({
            "id": id, "scheme": 1,
            "response": request_1, "processed_response": processed_response,
            "abstain": is_abstain(processed_response),
            "isAnswerable": 1,
            "request_status": "completed",
            "error_stage": None, "error": None
        })
    save_to_checkpoint(answers)

    # Схема 2
    try:
        request_2 = request_to_LLM(prompt_2(context, question))
    except Exception as error:
        print(error)
        answers.append({
            "id": id, "scheme": 2,
            "response": None, "processed_response": None,
            "abstain": None, "isAnswerable": 1,
            "request_status": "error",
            "error_stage": "answer_generation",
            "error": str(error)
        })
    else:
        sufficiency_raw, processed_response = process_response_2(request_2 or "")
        answers.append({
            "id": id, "scheme": 2,
            "response": request_2, "processed_response": processed_response,
            "abstain": is_abstain(processed_response),
            "sufficiency_raw":sufficiency_raw,
            "isAnswerable": 1,
            "request_status": "completed",
            "error_stage": None, "error": None
        })
    save_to_checkpoint(answers)

    # Схема 3: предварительная проверка
    try:
        request_3 = request_to_LLM(prompt_3(context, question))
    except Exception as error:
        print(error)
        answers.append({
            "id": id, "scheme": 3,
            "response": None, "processed_response": None,
            "abstain": None, "isAnswerable": 1,
            "sufficiency_raw": None,
            "request_status": "error",
            "error_stage": "sufficiency_check",
            "error": str(error)
        })
        save_to_checkpoint(answers)

    else:
        sufficiency_result = (request_3 or "").strip()

        answers.append({
            "id": id, "scheme": 3,
            "response": None, "processed_response": None,
            "abstain": None, "isAnswerable": 1,
            "sufficiency_raw": request_3,
            "request_status": "pending",
            "error_stage": None, "error": None
        })
        save_to_checkpoint(answers)

        if sufficiency_result == "SUFFICIENT":
            try:
                request = request_to_LLM(prompt_1(context, question))
            except Exception as error:
                print(error)
                answers[-1].update({
                    "request_status": "error",
                    "error_stage": "answer_generation",
                    "error": str(error)
                })
            else:
                processed_response = process_response(request or "")
                answers[-1].update({
                    "response": request,
                    "processed_response": processed_response,
                    "abstain": is_abstain(processed_response),
                    "request_status": "completed",
                    "error_stage": None,
                    "error": None
                })
            save_to_checkpoint(answers)

        elif sufficiency_result == "INSUFFICIENT":
            answers[-1].update({
                "response": "ABSTAIN",
                "processed_response": "ABSTAIN",
                "abstain": 1,
                "request_status": "completed",
                "error_stage": None,
                "error": None
            })
            save_to_checkpoint(answers)

        else:
            answers[-1].update({
                "response": None,
                "processed_response": "INVALID_FORMAT",
                "abstain": 0,
                "request_status": "completed",
                "error_stage": None,
                "error": None
            })
            save_to_checkpoint(answers)

    # ==================== ВОПРОСЫ БЕЗ ОТВЕТА ====================

    id = data_without_answer[index]['id']
    context = data_without_answer[index]['context']
    question = data_without_answer[index]['question']

    # Схема 1
    try:
        request_1 = request_to_LLM(prompt_1(context, question))
    except Exception as error:
        print(error)
        answers.append({
            "id": id, "scheme": 1,
            "response": None, "processed_response": None,
            "abstain": None, "isAnswerable": 0,
            "request_status": "error",
            "error_stage": "answer_generation",
            "error": str(error)
        })
    else:
        processed_response = process_response(request_1 or "")
        answers.append({
            "id": id, "scheme": 1,
            "response": request_1, "processed_response": processed_response,
            "abstain": is_abstain(processed_response),
            "isAnswerable": 0,
            "request_status": "completed",
            "error_stage": None, "error": None
        })
    save_to_checkpoint(answers)

    # Схема 2
    try:
        request_2 = request_to_LLM(prompt_2(context, question))
    except Exception as error:
        print(error)
        answers.append({
            "id": id, "scheme": 2,
            "response": None, "processed_response": None,
            "abstain": None, "isAnswerable": 0,
            "request_status": "error",
            "error_stage": "answer_generation",
            "error": str(error)
        })
    else:
        sufficiency_raw, processed_response = process_response_2(request_2 or "")
        answers.append({
            "id": id, "scheme": 2,
            "response": request_2, "processed_response": processed_response,
            "abstain": is_abstain(processed_response),
            "sufficiency_raw":sufficiency_raw,
            "isAnswerable": 0,
            "request_status": "completed",
            "error_stage": None, "error": None
        })
    save_to_checkpoint(answers)

    # Схема 3: предварительная проверка
    try:
        request_3 = request_to_LLM(prompt_3(context, question))
    except Exception as error:
        print(error)
        answers.append({
            "id": id, "scheme": 3,
            "response": None, "processed_response": None,
            "abstain": None, "isAnswerable": 0,
            "sufficiency_raw": None,
            "request_status": "error",
            "error_stage": "sufficiency_check",
            "error": str(error)
        })
        save_to_checkpoint(answers)

    else:
        sufficiency_result = (request_3 or "").strip()

        answers.append({
            "id": id, "scheme": 3,
            "response": None, "processed_response": None,
            "abstain": None, "isAnswerable": 0,
            "sufficiency_raw": request_3,
            "request_status": "pending",
            "error_stage": None, "error": None
        })
        save_to_checkpoint(answers)

        if sufficiency_result == "SUFFICIENT":
            try:
                request = request_to_LLM(prompt_1(context, question))
            except Exception as error:
                print(error)
                answers[-1].update({
                    "request_status": "error",
                    "error_stage": "answer_generation",
                    "error": str(error)
                })
            else:
                processed_response = process_response(request or "")
                answers[-1].update({
                    "response": request,
                    "processed_response": processed_response,
                    "abstain": is_abstain(processed_response),
                    "request_status": "completed",
                    "error_stage": None,
                    "error": None
                })
            save_to_checkpoint(answers)

        elif sufficiency_result == "INSUFFICIENT":
            answers[-1].update({
                "response": "ABSTAIN",
                "processed_response": "ABSTAIN",
                "abstain": 1,
                "request_status": "completed",
                "error_stage": None,
                "error": None
            })
            save_to_checkpoint(answers)

        else:
            answers[-1].update({
                "response": None,
                "processed_response": "INVALID_FORMAT",
                "abstain": 0,
                "request_status": "completed",
                "error_stage": None,
                "error": None
            })
            save_to_checkpoint(answers)

df = pd.DataFrame(answers)
df.to_csv("answers.csv", index = False)
