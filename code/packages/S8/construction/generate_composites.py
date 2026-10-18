"""Generate 150 composites with controlled subtask count distribution + full coverage"""
import json, random
random.seed(42)

atomics = json.load(open("g:/fmlab/linkmore/experiments/atomic-subtasks.json", encoding="utf-8"))
sub_map = {a["name"]: a for a in atomics}
names = [a["name"] for a in atomics]
by_scenario = {}
for a in atomics:
    by_scenario.setdefault(a["scenario"], []).append(a["name"])
scenarios = list(by_scenario.keys())

TARGET = {2: 75, 3: 53, 4: 22}  # total 150
used_sets = set()
all_comps = []
next_id = 1

def make_composite(subtask_names):
    global next_id
    per_sub = {n: sub_map[n]["required"] for n in subtask_names}
    scs = sorted(set(sub_map[n]["scenario"] for n in subtask_names))
    c = {
        "id": next_id, "scenarios": scs, "subtask_names": subtask_names,
        "per_subtask_required": per_sub, "conflicts": [],
        "reason": f"{'跨' if len(scs)>1 else '同'}场景组合：{' + '.join(subtask_names)}"
    }
    next_id += 1
    return c

def pick_cross_scenario(n_subs):
    """Pick n_subs subtasks from different scenarios (as many as possible)"""
    n_sc = min(n_subs, len(scenarios))
    selected_sc = random.sample(scenarios, n_sc)
    picked = []
    for sc in selected_sc:
        picked.append(random.choice(by_scenario[sc]))
    # If n_subs > n_scenarios, fill remaining from random scenarios
    while len(picked) < n_subs:
        sc = random.choice(scenarios)
        candidates = [n for n in by_scenario[sc] if n not in picked]
        if candidates:
            picked.append(random.choice(candidates))
    return picked

# Phase 1: Ensure all 63 subtasks covered (use 2-subtask pairs)
uncovered = set(names)
coverage_comps = []
uncovered_list = sorted(uncovered)
random.shuffle(uncovered_list)
i = 0
while i < len(uncovered_list):
    if i + 1 < len(uncovered_list):
        n1, n2 = uncovered_list[i], uncovered_list[i+1]
        # Prefer cross-scenario
        if sub_map[n1]["scenario"] == sub_map[n2]["scenario"] and i + 2 < len(uncovered_list):
            n2 = uncovered_list[i+2]
            uncovered_list[i+1], uncovered_list[i+2] = uncovered_list[i+2], uncovered_list[i+1]
        pair = frozenset([n1, n2])
        if pair not in used_sets:
            used_sets.add(pair)
            coverage_comps.append(make_composite([n1, n2]))
            uncovered -= {n1, n2}
        i += 2
    else:
        # Odd one out - pair with random
        n1 = uncovered_list[i]
        partners = [n for n in names if sub_map[n]["scenario"] != sub_map[n1]["scenario"] and n != n1]
        n2 = random.choice(partners)
        pair = frozenset([n1, n2])
        if pair not in used_sets:
            used_sets.add(pair)
            coverage_comps.append(make_composite([n1, n2]))
            uncovered -= {n1, n2}
        i += 1

print(f"Coverage phase: {len(coverage_comps)} composites, {len(uncovered)} uncovered")

# Count what we have so far by size
counts = {2: 0, 3: 0, 4: 0}
for c in coverage_comps:
    n = len(c["subtask_names"])
    counts[n] = counts.get(n, 0) + 1

all_comps = list(coverage_comps)

# Phase 2: Fill each size bucket to target
for target_size in [2, 3, 4]:
    needed = TARGET[target_size] - counts.get(target_size, 0)
    attempts = 0
    while needed > 0 and attempts < 500:
        attempts += 1
        picked = pick_cross_scenario(target_size)
        key = frozenset(picked)
        if key not in used_sets and len(set(picked)) == target_size:
            used_sets.add(key)
            all_comps.append(make_composite(picked))
            counts[target_size] = counts.get(target_size, 0) + 1
            needed -= 1

# Re-ID
for i, c in enumerate(all_comps):
    c["id"] = i + 1

# Stats
from collections import Counter
size_dist = Counter(len(c["subtask_names"]) for c in all_comps)
st_count = Counter()
for c in all_comps:
    for n in c["subtask_names"]:
        st_count[n] += 1
covered = set(st_count.keys())

print(f"Total: {len(all_comps)}")
print(f"Size distribution: {dict(size_dist)}")
print(f"Coverage: {len(covered)}/63")
if len(covered) < 63:
    print(f"Missing: {set(names) - covered}")

with open("g:/fmlab/linkmore/experiments/composites.json", "w", encoding="utf-8") as f:
    json.dump(all_comps, f, ensure_ascii=False, indent=2)
print("Saved composites.json")
