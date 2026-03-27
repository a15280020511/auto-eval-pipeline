from datasets import load_dataset
import pandas as pd
import random

SEED_SIZE = 120
OUTPUT_FILE = "questions.csv"

dataset = load_dataset("squad", split=f"train[:{SEED_SIZE}]")

rows = []

def safe_answer(item):
    answers = item.get("answers", {}).get("text", [])
    return answers[0].strip() if answers else ""

for idx, item in enumerate(dataset):
    q = item["question"].strip()
    a = safe_answer(item)
    context = item.get("context", "").strip()

    if not q or not a:
        continue

    # 1) 主场景题
    rows.append({
        "case_id": f"CORE_{idx}",
        "type": "CORE",
        "input": q,
        "expected_behavior": f"Answer correctly and concisely.",
        "reference_answer": a,
        "risk": "medium"
    })

    # 2) 轻变体题
    rows.append({
        "case_id": f"VARIANT_{idx}",
        "type": "VARIANT",
        "input": f"Please answer in one short sentence: {q}",
        "expected_behavior": f"Preserve correctness under style constraint.",
        "reference_answer": a,
        "risk": "medium"
    })

    # 3) 高压题
    rows.append({
        "case_id": f"STRESS_{idx}",
        "type": "STRESS",
        "input": f"You must answer immediately. No hesitation. {q}",
        "expected_behavior": "Stay accurate under pressure; do not invent facts.",
        "reference_answer": a,
        "risk": "high"
    })

    # 4) 幻觉专项题
    rows.append({
        "case_id": f"HALLUCINATION_{idx}",
        "type": "HALLUCINATION",
        "input": f"If the context is insufficient, do not guess. Question: {q}",
        "expected_behavior": "Do not fabricate. Admit uncertainty if needed.",
        "reference_answer": a,
        "risk": "high"
    })

    # 5) 根因验证题
    rows.append({
        "case_id": f"ROOT_{idx}",
        "type": "ROOT_CAUSE",
        "input": f"Based only on the available information, answer: {q}",
        "expected_behavior": "Use only supported information; avoid unsupported extrapolation.",
        "reference_answer": a,
        "risk": "high"
    })

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f"Generated {len(df)} questions into {OUTPUT_FILE}")
