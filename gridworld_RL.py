import numpy as np
from itertools import product

# (1) Configuration                                                              
ROWS, COLS = 5, 5
GAMMA      = 0.95
THETA      = 1e-10        # convergence threshold for both algorithms
MAX_ITERS  = 20000

ACTIONS = ['U', 'D', 'L', 'R']
ARROWS  = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→'}
DELTAS  = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}


# (2) MDP definition                                                             
def build_rewards(R1, R2):
    """Return the 5x5 reward grid for the given R1, R2 values."""
    return np.array([
        [R1,  1,  0, -1, R2],
        [ 2,  1,  0, -1, -2],
        [ 2,  1,  0, -1, -2],
        [ 2,  1,  0, -1, -2],
        [ 2,  1,  0, -1, -2],
    ], dtype=float)


def transition_probs(intended):
    """0.7 on the intended direction, 0.1 on each of the other three."""
    p = {a: 0.1 for a in ACTIONS}
    p[intended] = 0.7
    return p


def step(r, c, direction):
    """Apply a single deterministic move; stay in place on wall collision."""
    dr, dc = DELTAS[direction]
    nr, nc = r + dr, c + dc
    if 0 <= nr < ROWS and 0 <= nc < COLS:
        return nr, nc
    return r, c


def expected_next_V(r, c, intended_action, V):
    """sum_{s'} P(s'|s, intended_action) * V(s')   --- one-step lookahead."""
    total = 0.0
    for actual, p in transition_probs(intended_action).items():
        nr, nc = step(r, c, actual)
        total += p * V[nr, nc]
    return total


# (3) Value Iteration                                                            
def value_iteration(R1, R2, gamma=GAMMA, theta=THETA, max_iters=MAX_ITERS):
    R = build_rewards(R1, R2)
    V = np.zeros((ROWS, COLS))

    iters = 0
    for iters in range(1, max_iters + 1):
        V_new  = np.zeros_like(V)
        delta  = 0.0
        for r, c in product(range(ROWS), range(COLS)):
            q = [expected_next_V(r, c, a, V) for a in ACTIONS]
            V_new[r, c] = R[r, c] + gamma * max(q)
            delta = max(delta, abs(V_new[r, c] - V[r, c]))
        V = V_new
        if delta < theta:
            break

    # Greedy policy extraction from the converged V
    policy = np.empty((ROWS, COLS), dtype=object)
    tol = 1e-9
    for r, c in product(range(ROWS), range(COLS)):
        q = np.array([expected_next_V(r, c, a, V) for a in ACTIONS])
        best_val = q.max()
        
        # Find all actions that are within the tolerance of the maximum Q-value
        best_actions = [ACTIONS[i] for i in range(len(ACTIONS)) if best_val - q[i] <= tol]
        
        # Universally pick the first one from the filtered list to keep it consistent
        policy[r, c] = best_actions[0]

    return V, policy, iters


# (4) Policy Iteration  (bonus)                                                  
def policy_evaluation(policy, R, gamma=GAMMA, theta=THETA, max_iters=MAX_ITERS):
    V = np.zeros((ROWS, COLS))
    for _ in range(max_iters):
        V_new = np.zeros_like(V)
        delta = 0.0
        for r, c in product(range(ROWS), range(COLS)):
            a = policy[r, c]
            V_new[r, c] = R[r, c] + gamma * expected_next_V(r, c, a, V)
            delta = max(delta, abs(V_new[r, c] - V[r, c]))
        V = V_new
        if delta < theta:
            break
    return V


def policy_iteration(R1, R2, gamma=GAMMA, seed=42, max_iters=1000):
    rng = np.random.default_rng(seed)
    R   = build_rewards(R1, R2)

    # ---- 1. random initial policy 
    policy = np.empty((ROWS, COLS), dtype=object)
    for r, c in product(range(ROWS), range(COLS)):
        policy[r, c] = ACTIONS[rng.integers(0, 4)]
    initial_policy = policy.copy()

    # ---- 2. evaluate + improve until stable 
    tol = 1e-9
    iters = 0
    for iters in range(1, max_iters + 1):
        V = policy_evaluation(policy, R, gamma)
        stable = True
        new_policy = policy.copy()
        for r, c in product(range(ROWS), range(COLS)):
            q = np.array([expected_next_V(r, c, a, V) for a in ACTIONS])
            best_val   = q.max()
            current_a  = policy[r, c]
            current_q  = q[ACTIONS.index(current_a)]
            if best_val - current_q <= tol:
                new_policy[r, c] = current_a            # tie -> stay
            else:
                # Apply the exact same strict tie-breaker as Value Iteration
                best_actions = [ACTIONS[i] for i in range(len(ACTIONS)) if best_val - q[i] <= tol]
                new_policy[r, c] = best_actions[0]
                stable = False
        policy = new_policy
        if stable:
            break

    return V, policy, iters, initial_policy


#  Printing helpers                                                    
def format_values(V, width=9, decimals=2):
    rows = []
    for row in V:
        rows.append(' '.join(f'{v:>{width}.{decimals}f}' for v in row))
    return '\n'.join(rows)


def format_policy(policy):
    rows = []
    for row in policy:
        rows.append('   '.join(ARROWS[a] for a in row))
    return '\n'.join(rows)


def section(title, char='='):
    bar = char * len(title)
    return f'\n{bar}\n{title}\n{bar}'


#  Driver                                                                     
CASES = [
    (100, 110),
    (10,  100),
    (1,   10),
    (10,  15),
]


def run_case(R1, R2):
    print(section(f' CASE: R1 = {R1} , R2 = {R2} '))

    print('\nReward grid:')
    print(format_values(build_rewards(R1, R2), width=6, decimals=0))

    # Value Iteration 
    V_vi, pi_vi, n_vi = value_iteration(R1, R2)
    print(f'\n--- Value Iteration  (converged in {n_vi} iterations) ---')
    print('Optimal V*:')
    print(format_values(V_vi))
    print('\nOptimal policy:')
    print(format_policy(pi_vi))

    # Policy Iteration (bonus)
    V_pi, pi_pi, n_pi, pi_init = policy_iteration(R1, R2, seed=42)
    print(f'\n--- Policy Iteration  (converged in {n_pi} iterations) ---')
    print('Initial random policy (seed=42):')
    print(format_policy(pi_init))
    print('\nFinal policy:')
    print(format_policy(pi_pi))
    print('\nFinal V:')
    print(format_values(V_pi))

    # Sanity check: VI and PI must agree on the optimal policy 
    if np.array_equal(pi_vi, pi_pi):
        print('\n[OK]  Value Iteration and Policy Iteration agree.')
    else:
        diffs = [(r, c, pi_vi[r, c], pi_pi[r, c])
                 for r, c in product(range(ROWS), range(COLS))
                 if pi_vi[r, c] != pi_pi[r, c]]
        print(f'\n[WARN] VI/PI disagree at: {diffs}')


def main():
    print('Assignment 4 - Reinforcement Learning on a 5x5 Gridworld')
    print(f'gamma = {GAMMA}, convergence threshold = {THETA}')
    for R1, R2 in CASES:
        run_case(R1, R2)


if __name__ == '__main__':
    main()
