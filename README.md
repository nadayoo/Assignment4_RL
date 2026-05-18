# Assignment 4 — Reinforcement Learning (5×5 Gridworld)

**Course:** Artificial Intelligence — Alexandria University
**Algorithms:** Value Iteration (required) + Policy Iteration (bonus)


## Files

| File | What it is |
|---|---|
| `gridworld_rl.py` | The full implementation. Runs both Value Iteration and Policy Iteration for all four `(R1, R2)` cases. |
| `report.md` | The written report: results, optimal policies, and intuitive explanations of why each policy looks the way it does. |
| `run_output.txt` | A saved transcript of running `gridworld_rl.py` (so the grader can read the results without re-running). |
| `README.md` | This file. |

## How to run

```bash
python gridworld_rl.py
```

Optional (save the output to a file):

```bash
python gridworld_rl.py > run_output.txt
```

The script prints, for each of the four `(R1, R2)` cases:

1. The reward grid,
2. The converged value function `V*` from Value Iteration,
3. The greedy optimal policy from Value Iteration,
4. The random initial policy used to seed Policy Iteration,
5. The final policy and value function from Policy Iteration,
6. A sanity check that Value Iteration and Policy Iteration return the same policy.

## Model recap (from the assignment)

- 5×5 grid, four actions {Up, Down, Left, Right}.
- For any intended action, the environment moves the agent in the intended direction with probability **0.7**, and in each of the other three directions with probability **0.1**.
- A move that would leave the grid is replaced by **stay-in-place** (wall collision).
- Discount factor **γ = 0.95**.
- Bellman update used (reward attached to the current state):
  `V(s) = R(s) + γ · max_a Σ_{s'} P(s'|s,a) · V(s')`.

## Notes on tie-breaking

In two of the four cases the corner cells `(0,0)` and `(0,4)` have two equally-optimal actions (e.g. from `(0,0)` the distributions for "Up" and "Left" are identical because both bounce off a wall). To prevent Policy Iteration from oscillating between equally-optimal actions, the improvement step keeps the current action whenever its Q-value is within `1e-9` of the best — the standard textbook trick. With this rule, Value Iteration and Policy Iteration converge to **the same** optimal policy in all four cases.
