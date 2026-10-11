import ast
import re

from datasets import load_dataset, concatenate_datasets

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*|-?\.\d+")
GSM_SUFFIX = "\nSolve the problem step by step. At the end, write the final answer as 'The answer is <number>.'"
MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus"]


# ---------------- answer parsing ----------------
def norm_num(s):
    if s is None:
        return None
    s = str(s).replace(",", "").replace("$", "").replace("%", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def extract_pred_number(text):
    m = re.findall(r"answer is[:\s]*\$?\s*(-?[\d,]*\.?\d+)", text, flags=re.I)
    if m:
        return norm_num(m[-1])
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m and norm_num(m[-1]) is not None:
        return norm_num(m[-1])
    nums = NUM_RE.findall(text)
    return norm_num(nums[-1]) if nums else None


def num_equal(a, b, tol=1e-4):
    return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(b))


def last_boxed(s):
    idx = s.rfind("\\boxed")
    if idx < 0:
        idx = s.rfind("\\fbox")
        if idx < 0:
            return None
    depth, start = 0, None
    for i in range(idx, len(s)):
        if s[i] == "{":
            depth += 1
            if start is None:
                start = i + 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return s[start:i]
    return None


def norm_math(s):
    if s is None:
        return None
    s = s.strip()
    for a, b in [("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""), ("dfrac", "frac"),
                 ("tfrac", "frac"), ("^\\circ", ""), ("^{\\circ}", ""), ("\\$", ""), ("$", ""), ("\\%", ""),
                 ("%", ""), (" ", ""), ("\\text{", "{"), ("\\mbox{", "{")]:
        s = s.replace(a, b)
    s = s.rstrip(".")
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    v = norm_num(s)
    return str(v) if v is not None else s


def parse_mathqa_options(opt):
    if isinstance(opt, str) and opt.strip().startswith("["):
        try:
            opt = ast.literal_eval(opt)
        except Exception:
            pass
    if isinstance(opt, list):
        out = {}
        for o in opt:
            m = re.match(r"\s*([a-e])\s*\)\s*(.*)", str(o))
            if m:
                out[m.group(1)] = m.group(2).strip(" '\"")
        return out
    pairs = re.findall(r"([a-e])\s*\)\s*(.*?)(?=\s*,\s*[a-e]\s*\)|$)", opt)
    return {k: v.strip(" '\"") for k, v in pairs}


# ---------------- datasets ----------------
def load_gsm8k():
    ds = load_dataset("openai/gsm8k", "main")
    conv = lambda ex: {"question": ex["question"],  # noqa: E731
                       "answer": ex["answer"].split("####")[-1].strip().replace(",", "")}
    return ds["train"].map(conv), ds["test"].map(conv)


def load_mathqa():
    for kw in [dict(path="allenai/math_qa", revision="refs/convert/parquet"),
               dict(path="allenai/math_qa", trust_remote_code=True)]:
        try:
            return load_dataset(split="test", **kw)
        except Exception as e:
            err = e
    print("MathQA not loaded:", repr(err)[:200])
    return None


def load_math():
    try:
        return concatenate_datasets([load_dataset("EleutherAI/hendrycks_math", s, split="test")
                                     for s in MATH_SUBJECTS])
    except Exception as e:
        print("MATH not loaded:", repr(e)[:200])
        return None


def make_eval_items(name, ds):
    items = []
    if name == "gsm8k":
        for ex in ds:
            items.append({"prompt": ex["question"] + GSM_SUFFIX, "gold": ex["answer"]})
    elif name == "mathqa":
        for ex in ds:
            opts = parse_mathqa_options(ex["options"])
            if not opts:
                continue
            opt_str = "\n".join(f"({k}) {v}" for k, v in sorted(opts.items()))
            items.append({"prompt": f"{ex['Problem']}\nOptions:\n{opt_str}\nSolve step by step, then finish with "
                                    f"'The answer is (x).' where x is the option letter.",
                          "gold": ex["correct"].strip().lower(), "opts": opts})
    elif name == "math":
        for ex in ds:
            items.append({"prompt": ex["problem"] + "\nSolve step by step and put the final answer in \\boxed{}.",
                          "gold": norm_math(last_boxed(ex["solution"]))})
    return items


def load_eval_sets(names):
    sets = {}
    if "gsm8k" in names:
        sets["gsm8k"] = make_eval_items("gsm8k", load_gsm8k()[1])
    if "mathqa" in names and (ds := load_mathqa()) is not None:
        sets["mathqa"] = make_eval_items("mathqa", ds)
    if "math" in names and (ds := load_math()) is not None:
        sets["math"] = make_eval_items("math", ds)
    return sets


def score_item(name, text, item):
    if name == "gsm8k":
        return num_equal(extract_pred_number(text), norm_num(item["gold"]))
    if name == "mathqa":
        m = re.findall(r"answer is\s*:?\s*\(?([a-e])\)?", text, flags=re.I)
        if m:
            return m[-1].lower() == item["gold"]
        pv = extract_pred_number(text)
        for k, v in item["opts"].items():
            nv = NUM_RE.findall(v)
            if nv and num_equal(pv, norm_num(nv[0])):
                return k == item["gold"]
        return False
    if name == "math":
        return item["gold"] is not None and norm_math(last_boxed(text)) == item["gold"]
    return False
