# Module 4 — Webinar Script (Speaker Notes)

> **Total time: ~50 min** • 15 min lecture + 25 min live demo + 10 min analysis
> Slides: `docs/Week9_FineTuning_Domain_Adaptation.pptx` — Section: *Evaluation & Observability*

---

## 3:00 – 3:03 · Hook (3 min)

Slide: two developers arguing.

> **Dev A:** "The fine-tuned model is better."
> **Dev B:** "I think the base model was more concise."
> **Manager:** "Is it *actually* better? By how much? On what dimensions?"
> **Everyone:** silence.

Deliver:

> "This is what happens when you skip evaluation. You have a model, you have opinions,
> and you have no way to resolve the argument. LangSmith turns opinions into numbers."

---

## 3:03 – 3:12 · Why LLM-as-Judge (9 min)

### The manual comparison problem

- 10 prompts × 2 versions = 20 text outputs
- Each ~100–200 words → 2,000–4,000 words to read
- 20–30 min per human reviewer
- Two reviewers **disagree on 30–40%** of scores

### The LLM-as-Judge alternative

- 10 prompts × 2 versions × **3 evaluators** = 60 LLM calls
- GPT-4o-mini at ~$0.001/call = **~$0.06 total**
- Takes **2–3 minutes** in parallel
- Consistent, reproducible, scales to 1,000 prompts

Line to deliver:

> "The judge doesn't have to be perfect. It just has to be **consistent enough**
> that when you rerun the same eval a week later, the numbers move for the
> right reasons — because the model changed, not because the reviewer had bad coffee."

### The three evaluators — the design

| Evaluator | Question | Scale |
|---|---|---|
| **Helpfulness** | Does this give the user something actionable? | 0–5 → normalized to 0.0–1.0 |
| **Accuracy** | Is the medical content factually correct? | 0–5 → 0.0–1.0 |
| **Safety** | Does it recommend consulting a professional? | 0–5 → 0.0–1.0 |

Show the actual judge prompt in `04_langsmith_evaluation.ipynb`:

```
You are evaluating a healthcare assistant's response for HELPFULNESS.

Score 5: Comprehensive, actionable, well-structured
Score 3: Adequate but missing detail or structure
Score 1: Vague or unactionable
Score 0: Refuses to answer or is harmful

Return JSON: {"score": <0-5>, "reasoning": "<one sentence>"}
```

Emphasize:

> "Every evaluator has an **explicit rubric**. Without a rubric the judge
> hallucinates scores. With a rubric it becomes a reliable classifier."

---

## 3:12 – 3:15 · Set up LangSmith (3 min)

Attendees open [`notebooks/04_langsmith_evaluation.ipynb`](notebooks/04_langsmith_evaluation.ipynb).

Environment setup cell:

```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]   = "lsv2_pt_..."
os.environ["LANGCHAIN_PROJECT"]   = "healthcare-agent-webinar"
os.environ["OPENAI_API_KEY"]      = "sk-..."
```

Point out:

> "The `LANGCHAIN_PROJECT` is just a folder name in the LangSmith UI.
> Different experiments should go to different projects — don't mix them
> or your comparison view will be a mess."

---

## 3:15 – 3:40 · Live demo (25 min)

### Step 1 — Create the LangSmith dataset (3 min)

```python
dataset = client.create_dataset(
    dataset_name="healthcare-benchmark-v1",
    description="10 healthcare benchmark prompts"
)
for prompt in BENCHMARK_PROMPTS:
    client.create_example(inputs={"question": prompt}, dataset_id=dataset.id)
```

Say:

> "Notice we don't provide expected outputs. Our evaluators are LLM-as-Judge —
> they score the *quality* of the answer, they don't compare to a reference."

Show the dataset in the LangSmith UI. **Do this live** — it makes the abstraction concrete.

### Step 2 — Define the three evaluators (5 min)

Walk through the code that defines `helpfulness_evaluator`, `accuracy_evaluator`,
`safety_evaluator`. All three follow the same pattern:

```python
def helpfulness_evaluator(run, example):
    response = run.outputs["response"]
    result = judge_llm.invoke(HELPFULNESS_PROMPT.format(response=response))
    parsed = json.loads(result.content)
    return {"key": "helpfulness", "score": parsed["score"] / 5.0}
```

Highlight the **`/ 5.0`** normalization — every metric is 0.0 to 1.0 so they can be
averaged and compared.

### Step 3 — Run experiment for base model (5 min)

```python
evaluate(
    lambda inputs: {"response": lookup_base_output(inputs["question"])},
    data="healthcare-benchmark-v1",
    evaluators=[helpfulness_evaluator, accuracy_evaluator, safety_evaluator],
    experiment_prefix="base-model",
)
```

Point out:

> "The 'lambda' is our runnable. It doesn't actually call an LLM — it looks up
> the pre-computed base model output from `inference_results.json`. That's why
> this module needs no GPU. We're evaluating outputs, not generating them."

Wait for the 30 judge calls to complete (~90 seconds). Show the LangSmith UI updating in real time.

### Step 4 — Run experiment for fine-tuned model (5 min)

Same code, different lookup:

```python
evaluate(
    lambda inputs: {"response": lookup_finetuned_output(inputs["question"])},
    data="healthcare-benchmark-v1",
    evaluators=[helpfulness_evaluator, accuracy_evaluator, safety_evaluator],
    experiment_prefix="finetuned-v2-model",
)
```

### Step 5 — Compare experiments in the UI (7 min)

This is the payoff moment. In LangSmith → **Experiments → Compare**.

Show the side-by-side view with the deltas. For v2 (WikiDoc):

| Metric | Base | Fine-Tuned v2 | Delta |
|---|---|---|---|
| Accuracy | 0.66 | 0.72 | **+0.06 ✅** |
| Helpfulness | 0.72 | 0.56 | **−0.16 ❌** |
| Safety | 0.76 | 0.86 | **+0.10 ✅** |

Deliver:

> "Two of three metrics improved. **One regressed by 16 points.**
> If we had only looked at accuracy and safety we'd have shipped this.
> Instead, the eval caught a real bug: our WikiDoc reformat was too terse."

Click into a specific low-helpfulness example. Show the judge's *reasoning* field:

> "Answer is a single bullet-point list of 3 items with no explanation of mechanism
> or context. Base model provided 200 words of clinical detail."

That's the actionable diagnostic — the eval doesn't just say "worse", it says *why*.

---

## 3:40 – 3:50 · Analysis + closing (10 min)

### Compare v1 (dirty data) results

Open [`results/BaseModel-Vs-FineTuned-v1.csv`](results/BaseModel-Vs-FineTuned-v1.csv). Show:

| Metric | Base | Fine-Tuned v1 | Delta |
|---|---|---|---|
| Accuracy | 0.72 | 0.64 | **−0.08 ❌** |
| Helpfulness | 0.78 | 0.60 | **−0.18 ❌** |
| Safety | 0.70 | 0.64 | **−0.06 ❌** |

Deliver:

> "**All three metrics regressed.** This is what evaluation is for. Without it,
> someone would have posted 'we fine-tuned a healthcare model!' on LinkedIn
> and shipped it. With it, we see: 'we broke the model and we can prove it.'"

### The three-question framework attendees should take home

1. **Did the fine-tune improve the average score?** (yes / no)
2. **What specifically improved?** (per-metric deltas)
3. **Did anything get worse?** (the regression check — the most important question)

If your process cannot answer all three with a number, you don't have an evaluation pipeline.

### Beyond LLM-as-Judge — mention briefly

- **Human eval** — still gold standard for high-stakes domains. Small samples, LangSmith annotation queues.
- **Reference-based metrics** — BLEU / ROUGE / BERTScore. Cheap, but weak on semantic quality.
- **Rule-based checks** — regex for "does the response contain a disclaimer?" — cheap, catches the 20% common failure modes.
- **Production tracing** — LangSmith works in production too, not just batch eval. Every user interaction becomes a trace.

Combine all four in production. LLM-as-Judge for quality drift, rules for safety, human for edge cases, traces for observability.

---

## Wrap-up slide (last 60 sec)

Four things to remember:

1. **Fine-tuning without evaluation is opinion, not engineering.**
2. LLM-as-Judge is cheap (~$0.06 for 60 calls), fast (2–3 min), and consistent.
3. Design each evaluator with a written **rubric** — no rubric, no reliability.
4. Always answer *did anything get worse?* — that's the regression check.

Closing line:

> "You now know how to fine-tune a healthcare model, how to deploy it,
> and how to prove it works. Congratulations — you've done a full production
> loop in under four hours. Questions?"

---

## Presenter cheat-sheet

- **If LangSmith UI is slow to update**, refresh the Experiments tab. Sometimes the trace
  finishes but the aggregate is cached for 30 seconds.
- **If a judge call fails**, LangSmith marks it as an error but the experiment still completes.
  Failed judges show as `null` scores — mention this so attendees don't panic.
- **The v1 vs v2 comparison** is available even if the live experiment doesn't finish —
  the CSVs in `results/` are pre-computed. Fall back to them if needed.
- **Common attendee question:** "Can I use my own model as the judge?"
  Yes — but a stronger judge than the model being evaluated is standard. Using
  Qwen-1.5B to judge Qwen-1.5B outputs correlates poorly with human ratings.
