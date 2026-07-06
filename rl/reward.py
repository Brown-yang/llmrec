"""Reward functions for recommendation-oriented RL (GRPO / RFT / DPO).

Implements the report's reward design (OneReason §6, eq 7-9):
    R_rule(c)   = 1 if the generated itemic token(s) hit the ground-truth set, else 0   (eq 8)
    R_div(CoT)  = max(0, m1 - 1) / (K - 1)   where m1 = #distinct first sub-tokens (s_a)
                  among the K itemic sequences from the same prompt/CoT               (eq 9)
    R           = R_rule * R_div                                                        (eq 7)

⚠️ REWARD IS A PROXY (see RL_DESIGN.md §0.2): the real competition decodes itemic
tokens to item IDs and scores at item granularity. We have no itemic tokenizer /
catalog, so we match at the itemic-token-STRING level (exact 4-token pattern). Highly
correlated with the real metric but not identical -> always confirm with official eval.

Usable two ways:
  1. Standalone: import itemic_hit / diversity_factor / group_reward for RFT/DPO filtering.
  2. As a TRL GRPOTrainer reward_func: `make_grpo_reward_func()` returns a callable with
     signature reward_func(prompts, completions, gold, **kwargs) -> list[float].
"""

import re
from collections import defaultdict

# full itemic pattern: <|domain_begin|><s_a_N><s_b_N><s_c_N>
ITEM_FULL_RE = re.compile(r"<\|\w+?_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")
# capture the first sub-token (s_a) value -- the report's diversity is over s_a (coarse category)
ITEM_SA_RE = re.compile(r"<\|\w+?_begin\|><s_a_(\d+)><s_b_\d+><s_c_\d+>")
# parse a full itemic pattern into (domain, s_a, s_b, s_c)
ITEM_PARSE_RE = re.compile(r"<\|(\w+?)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")

# graded partial-credit weights (RL_DESIGN.md: dense reward to fight the ~0.5% exact-hit
# sparsity). A generated item scores against the BEST-matching gold item, requiring same
# domain: full 3-token match=1.0, first-two (s_a+s_b)=0.6, first (s_a, coarse category)=0.3.
GRADE_FULL, GRADE_AB, GRADE_A = 1.0, 0.6, 0.3


def extract_items(text: str) -> list[str]:
    """All full itemic patterns in order of appearance (may repeat)."""
    return ITEM_FULL_RE.findall(text)


def parse_item(s: str):
    """'<|video_begin|><s_a_1><s_b_2><s_c_3>' -> ('video', 1, 2, 3), else None."""
    m = ITEM_PARSE_RE.match(s.strip())
    return (m[1], int(m[2]), int(m[3]), int(m[4])) if m else None


def _graded_pair(gen, gold) -> float:
    """Partial-credit score between one generated item and one gold item (same-domain required)."""
    if gen is None or gold is None or gen[0] != gold[0]:
        return 0.0
    (_, ga, gb, gc), (_, da, db, dc) = gen, gold
    if (ga, gb, gc) == (da, db, dc):
        return GRADE_FULL
    if (ga, gb) == (da, db):
        return GRADE_AB
    if ga == da:
        return GRADE_A
    return 0.0


def graded_reward(completion: str, gold) -> float:
    """Dense accuracy signal: best partial match of any generated item vs any gold item.
    Returns in [0,1]: 1.0 exact, 0.6 first-two sub-tokens, 0.3 first sub-token (category), 0 else."""
    gens = [parse_item(x) for x in extract_items(completion)]
    golds = [parse_item(x) for x in normalize_gold(gold)]
    best = 0.0
    for gi in gens:
        for go in golds:
            s = _graded_pair(gi, go)
            if s > best:
                best = s
                if best == GRADE_FULL:
                    return best
    return best


def extract_sa(text: str) -> list[int]:
    """First sub-token (s_a) values of all itemic patterns in `text`."""
    return [int(x) for x in ITEM_SA_RE.findall(text)]


def normalize_gold(gold) -> set[str]:
    """gold may be a list[str] of itemic patterns, or a raw response string to parse."""
    if isinstance(gold, str):
        return set(extract_items(gold))
    return set(gold)


def itemic_hit(completion: str, gold) -> float:
    """R_rule (eq 8): 1.0 if ANY itemic token generated in `completion` is in the
    ground-truth set, else 0.0. (For single-target 懂推荐, this == Pass. For multi-target,
    use recall_hit below.)"""
    gold_set = normalize_gold(gold)
    if not gold_set:
        return 0.0
    gen = set(extract_items(completion))
    return 1.0 if (gen & gold_set) else 0.0


def recall_hit(completion: str, gold) -> float:
    """Fraction of ground-truth items covered by this completion (for multi-target gold)."""
    gold_set = normalize_gold(gold)
    if not gold_set:
        return 0.0
    gen = set(extract_items(completion))
    return len(gen & gold_set) / len(gold_set)


def diversity_factor(completions: list[str], k: int | None = None) -> float:
    """R_div (eq 9): max(0, m1 - 1) / (K - 1), m1 = #distinct s_a across a GROUP of
    completions (the K samples for the same prompt). Rewards covering diverse categories."""
    if k is None:
        k = len(completions)
    if k <= 1:
        return 0.0
    sa_values = set()
    for c in completions:
        sa_values.update(extract_sa(c))
    m1 = len(sa_values)
    return max(0, m1 - 1) / (k - 1)


def group_reward(completions: list[str], gold, use_recall: bool = False) -> list[float]:
    """Report reward for a GROUP of completions sharing one prompt/gold (eq 7):
    per-completion R_rule (or recall) * shared R_div. Returns one reward per completion."""
    div = diversity_factor(completions)
    acc_fn = recall_hit if use_recall else itemic_hit
    return [acc_fn(c, gold) * div for c in completions]


def make_grpo_reward_func(accuracy: str = "graded", diversity: bool = True,
                          div_weight: float = 1.0):
    """Return a TRL-GRPOTrainer-compatible reward function.

    accuracy: "exact"  -> R_rule 0/1 (report eq8, sparse ~0.5% hit)
              "recall" -> fraction of gold covered
              "graded" -> partial credit 0.3/0.6/1.0 (DENSE, recommended vs sparse hits)
    diversity: multiply by R_div (report eq7). div_weight lets you soften it:
               reward = acc * (1 - div_weight + div_weight * R_div).

    TRL calls reward_func(prompts, completions, **cols); `completions` is the whole batch
    (several prompts x num_generations) and dataset cols (e.g. `gold`) arrive as kwargs.
    We group by prompt to compute the shared R_div.
    """
    acc_fn = {"exact": itemic_hit, "recall": recall_hit, "graded": graded_reward}[accuracy]

    def reward_func(prompts, completions, gold=None, **kwargs):
        n = len(completions)
        golds = gold if gold is not None else [None] * n
        groups = defaultdict(list)
        for i, p in enumerate(prompts):
            groups[p if isinstance(p, str) else str(p)].append(i)

        rewards = [0.0] * n
        for _, idxs in groups.items():
            comps = [completions[i] for i in idxs]
            g = golds[idxs[0]]
            div = diversity_factor(comps) if diversity else 1.0
            div_mult = (1 - div_weight) + div_weight * div
            for i in idxs:
                rewards[i] = acc_fn(completions[i], g) * div_mult
        return rewards

    return reward_func


if __name__ == "__main__":
    # quick self-test (no GPU)
    gold = ["<|video_begin|><s_a_100><s_b_2><s_c_3>"]
    hit = "该用户最近喜欢的视频有: <|video_begin|><s_a_100><s_b_2><s_c_3>"
    miss = "该用户最近喜欢的视频有: <|video_begin|><s_a_999><s_b_9><s_c_9>"
    assert itemic_hit(hit, gold) == 1.0
    assert itemic_hit(miss, gold) == 0.0
    assert recall_hit(hit, gold) == 1.0
    # diversity: 4 samples, s_a in {100, 999, 5, 100} -> distinct {100,999,5}=3 -> (3-1)/(4-1)
    comps = [hit, miss,
             "<|video_begin|><s_a_5><s_b_1><s_c_1>",
             "<|video_begin|><s_a_100><s_b_7><s_c_7>"]
    d = diversity_factor(comps)
    assert abs(d - (3 - 1) / (4 - 1)) < 1e-9, d
    gr = group_reward(comps, gold)
    # only the exact-hit completion gets R_rule=1, times shared div
    assert gr[0] == d and gr[1] == 0.0, gr
    # graded partial credit: gold s_a=100,s_b=2,s_c=3 (video)
    assert graded_reward(hit, gold) == 1.0                                   # full
    assert graded_reward("<|video_begin|><s_a_100><s_b_2><s_c_99>", gold) == 0.6   # s_a+s_b
    assert graded_reward("<|video_begin|><s_a_100><s_b_9><s_c_9>", gold) == 0.3    # s_a only
    assert graded_reward("<|prod_begin|><s_a_100><s_b_2><s_c_3>", gold) == 0.0     # wrong domain
    assert graded_reward(miss, gold) == 0.0
    print("graded partial-credit OK")
    # TRL-style: 2 prompts x 2 gens
    rf = make_grpo_reward_func()
    prompts = ["pA", "pA", "pB", "pB"]
    comps2 = [hit, miss,
              "<|prod_begin|><s_a_1><s_b_1><s_c_1>", "<|prod_begin|><s_a_2><s_b_2><s_c_2>"]
    golds2 = [gold, gold, ["<|prod_begin|><s_a_1><s_b_1><s_c_1>"], ["<|prod_begin|><s_a_1><s_b_1><s_c_1>"]]
    out = rf(prompts, comps2, gold=golds2)
    print("self-test OK; sample rewards:", [round(x, 3) for x in out])
