import os
import json
import pandas as pd
from google import genai

MODEL_NAME = "gemini-3-flash-preview"

BASELINE_PROMPT = """
You are a careful assistant.
Answer truthfully.
If information is insufficient, say you are not sure.
Do not fabricate facts.
""".strip()

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def call_gemini(client, system_prompt: str, user_prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            {"role": "user", "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}
        ],
    )
    return response.text.strip()

def safe_parse_json(text: str):
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    return json.loads(text)

def evaluate_cases(client, questions_df, evaluator_prompt, current_prompt):
    results = []

    for _, row in questions_df.iterrows():
        user_prompt = f"""
Current prompt:
{current_prompt}

Test case:
case_id: {row['case_id']}
type: {row['type']}
input: {row['input']}
expected_behavior: {row['expected_behavior']}
acceptable_behavior: {row['acceptable_behavior']}
unacceptable_behavior: {row['unacceptable_behavior']}
reference_answer: {row['reference_answer']}
risk: {row['risk']}

Return strict JSON only.
""".strip()

        raw = call_gemini(client, evaluator_prompt, user_prompt)

        try:
            parsed = safe_parse_json(raw)
        except Exception:
            parsed = {
                "pass_or_fail": "FAIL",
                "hallucination_detected": False,
                "failure_type": "parser_error",
                "severity": "high",
                "brief_reason": raw[:200],
                "recommendation": "Fix evaluator JSON output"
            }

        results.append({
            "case_id": row["case_id"],
            "type": row["type"],
            "risk": row["risk"],
            "pass_or_fail": parsed.get("pass_or_fail", "FAIL"),
            "hallucination_detected": parsed.get("hallucination_detected", False),
            "failure_type": parsed.get("failure_type", "unknown"),
            "severity": parsed.get("severity", "medium"),
            "brief_reason": parsed.get("brief_reason", ""),
            "recommendation": parsed.get("recommendation", "")
        })

    return pd.DataFrame(results)

def summarize_results(results_df: pd.DataFrame):
    total = len(results_df)
    failed = len(results_df[results_df["pass_or_fail"] == "FAIL"])
    hallucinations = len(results_df[results_df["hallucination_detected"] == True])
    high_severity = len(results_df[results_df["severity"] == "high"])

    by_type = results_df.groupby("type")["pass_or_fail"].apply(
        lambda s: (s == "PASS").mean()
    ).to_dict()

    high_risk_failures = results_df[
        (results_df["risk"] == "high") & (results_df["pass_or_fail"] == "FAIL")
    ][["case_id", "type", "failure_type", "severity", "brief_reason"]].head(10)

    return {
        "total_cases": total,
        "failed_cases": failed,
        "hallucination_count": hallucinations,
        "high_severity_count": high_severity,
        "pass_rate_by_type": by_type,
        "high_risk_failures": high_risk_failures.to_dict(orient="records")
    }

def refine_prompt(client, builder_prompt, current_prompt, summary):
    user_prompt = f"""
Current prompt:
{current_prompt}

Evaluation summary:
{json.dumps(summary, ensure_ascii=False, indent=2)}

Return strict JSON only using:
{{
  "decision": "REFINE",
  "reasons": ["..."],
  "next_prompt": "..."
}}
""".strip()

    raw = call_gemini(client, builder_prompt, user_prompt)
    return safe_parse_json(raw)

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing.")

    os.makedirs("outputs", exist_ok=True)

    client = genai.Client(api_key=api_key)

    evaluator_prompt = load_text("prompts/gem2_evaluator.txt")
    builder_prompt = load_text("prompts/gem1_builder.txt")
    questions_df = pd.read_csv("data/questions.csv")

    results_df = evaluate_cases(client, questions_df, evaluator_prompt, BASELINE_PROMPT)
    results_df.to_csv("outputs/evaluation_results.csv", index=False)

    summary = summarize_results(results_df)

    decision = refine_prompt(client, builder_prompt, BASELINE_PROMPT, summary)

    with open("outputs/decision.json", "w", encoding="utf-8") as f:
        json.dump(decision, f, ensure_ascii=False, indent=2)

    with open("outputs/next_prompt.txt", "w", encoding="utf-8") as f:
        f.write(decision.get("next_prompt", BASELINE_PROMPT))

    print("Done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(decision, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
