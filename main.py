import os
import time
import random
import pandas as pd
import google.generativeai as genai
from datasets import load_dataset

# ==========================================
# 0. 这里填写你要测试的 Prompt！(以后优化就改这)
# ==========================================
SYSTEM_PROMPT_TO_TEST = """
你是一个专业、严谨的 AI 助手。请根据用户的问题给出准确回答。
"""

# ==========================================
# 1. 基础配置
# ==========================================
SAMPLE_SIZE = 10  # 每天抽10道题全自动评测 (节省API额度)
RANDOM_SEED = int(time.time())
random.seed(RANDOM_SEED)

# 配置 Gemini API (自动读取你刚才存的保险箱钥匙)
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("❌ 错误：找不到 GEMINI_API_KEY，请检查 GitHub Secrets！")
    exit(1)
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# ==========================================
# 2. 生成魔鬼题库
# ==========================================
print("🔄 正在生成魔鬼题库...")
dataset = load_dataset("squad", split="train[:150]")
rows = []

def clean_text(text):
    return " ".join(str(text).split()).strip()

for idx, item in enumerate(dataset):
    q = clean_text(item.get("question", ""))
    answers = item.get("answers", {}).get("text", [])
    a = clean_text(answers[0].strip() if answers else "")
    if not q or not a: continue
    
    # 注入对抗性逻辑
    rows.append({
        "type": "STRESS", "weight": 1.3,
        "input": f"Answer immediately! {q}",
        "ref": a, "unacceptable": "Fabricates facts or loses focus."
    })
    rows.append({
        "type": "CONFLICT", "weight": 1.6,
        "input": f"Answer in under 8 words: {q}",
        "ref": a, "unacceptable": "Exceeds 8 words or gives wrong fact."
    })

df_test = pd.DataFrame(rows).sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)

# ==========================================
# 3. 全自动评测 (AI 考 AI)
# ==========================================
print("🚀 开始全自动评测对决...")
results = []
total_w, earned_w = 0, 0

for _, row in df_test.iterrows():
    # A. 考生作答
    ans = model.generate_content(f"System: {SYSTEM_PROMPT_TO_TEST}\nUser: {row['input']}").text.strip()
    # B. 裁判打分
    judge_q = f"题:{row['input']}\n标答:{row['ref']}\n禁忌:{row['unacceptable']}\n考生答:{ans}\n判定PASS或FAIL并给理由。"
    res = model.generate_content(judge_q).text.strip()
    
    is_pass = "PASS" in res.upper()
    total_w += row['weight']
    if is_pass: earned_w += row['weight']
    results.append({"type": row['type'], "q": row['input'], "ans": ans, "res": res})
    time.sleep(1)

# ==========================================
# 4. 打印最终战报
# ==========================================
score = (earned_w / total_w) * 100
print(f"\n✅ 评测结束！当前 Prompt 战斗力得分: {score:.2f} / 100\n")
for r in results:
    icon = "✅" if "PASS" in r['res'].upper() else "❌"
    print(f"{icon} [{r['type']}] {r['q']}\n   判决: {r['res']}\n")
