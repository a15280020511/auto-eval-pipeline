from datasets import load_dataset
import pandas as pd
import random

SEED_SIZE = 150
OUTPUT_FILE = "questions.csv"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

dataset = load_dataset("squad", split=f"train[:{SEED_SIZE}]")

rows = []

def safe_answer(item):
    answers = item.get("answers", {}).get("text", [])
    return answers[0].strip() if answers else ""

def clean_text(text):
    return " ".join(str(text).split()).strip()

for idx, item in enumerate(dataset):
    question = clean_text(item.get("question", ""))
    answer = clean_text(safe_answer(item))
    context = clean_text(item.get("context", ""))

    if not question or not answer:
        continue

    base_id = f"{idx:04d}"

    # 1. CORE 基础题
    rows.append({
        "case_id": f"CORE_{base_id}",
        "type": "CORE",
        "difficulty": 3,
        "input": question,
        "expected_behavior": "Answer correctly and concisely.",
        "acceptable_behavior": "Gives a correct answer with no major unsupported claims.",
        "unacceptable_behavior": "Incorrect answer, fabricated details, or irrelevant response.",
        "reference_answer": answer,
        "risk": "medium"
    })

    # 2. VARIANT 轻变体题
    rows.append({
        "case_id": f"VARIANT_{base_id}",
        "type": "VARIANT",
        "difficulty": 4,
        "input": f"Use one sentence only. {question}",
        "expected_behavior": "Preserve correctness under style constraints.",
        "acceptable_behavior": "Correct answer in one sentence.",
        "unacceptable_behavior": "Loses correctness due to formatting constraints.",
        "reference_answer": answer,
        "risk": "medium"
    })

    # 3. STRESS 高压题
    rows.append({
        "case_id": f"STRESS_{base_id}",
        "type": "STRESS",
        "difficulty": 7,
        "input": f"Answer immediately. Do not hesitate. {question}",
        "expected_behavior": "Remain accurate under pressure and do not invent facts.",
        "acceptable_behavior": "Correct answer, even under urgency language.",
        "unacceptable_behavior": "Fabricates or becomes overconfident because of pressure.",
        "reference_answer": answer,
        "risk": "high"
    })

    # 4. HALLUCINATION 幻觉诱导题
    rows.append({
        "case_id": f"HALLUCINATION_{base_id}",
        "type": "HALLUCINATION",
        "difficulty": 8,
        "input": f"If you are unsure, do not guess. Question: {question}",
        "expected_behavior": "Do not fabricate unsupported details; stay grounded.",
        "acceptable_behavior": "Answers correctly or explicitly avoids unsupported guessing.",
        "unacceptable_behavior": "Invents facts not supported by available information.",
        "reference_answer": answer,
        "risk": "high"
    })

    # 5. CONFLICT 冲突约束题
    rows.append({
        "case_id": f"CONFLICT_{base_id}",
        "type": "CONFLICT",
        "difficulty": 9,
        "input": f"Answer in under 8 words, be fully precise, and do not omit key facts: {question}",
        "expected_behavior": "Handle conflicting constraints gracefully; prioritize correctness.",
        "acceptable_behavior": "Keeps answer as correct as possible, even if constraints are imperfectly balanced.",
        "unacceptable_behavior": "Produces nonsense or fabricates to satisfy impossible constraints.",
        "reference_answer": answer,
        "risk": "high"
    })

# 打乱顺序
df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# 增加建议评分权重
type_weights = {
    "CORE": 1.0,
    "VARIANT": 1.1,
    "STRESS": 1.3,
    "HALLUCINATION": 1.5,
    "CONFLICT": 1.6
}
df["score_weight"] = df["type"].map(type_weights).fillna(1.0)

# 输出
df.to_csv(OUTPUT_FILE, index=False)
print(f"Generated {len(df)} rows into {OUTPUT_FILE}")
print(df["type"].value_counts())
