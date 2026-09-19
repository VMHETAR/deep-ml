def differential_sarsa(transitions: dict, initial_state: str, alpha: float, beta: float, num_steps: int) -> tuple:
    """
    Differential Sarsa for the average-reward continuing setting.
    
    Args:
        transitions: dict mapping (state, action) -> (reward, next_state)
        initial_state: starting state
        alpha: step size for Q-value updates
        beta: step size for average reward estimate
        num_steps: number of steps to simulate
    
    Returns:
        Tuple of (Q, R_bar) where Q is a dict {(state, action): float}
        and R_bar is a float.
    """
    Q = {}

    for state_action in transitions:
        Q[state_action] = 0.0

    R_bar = 0.0
    current_state = initial_state

    for _ in range(num_steps):

        actions = [
            action
            for (state, action) in transitions
            if state == current_state
        ]
        max_Q = max(
                Q[(current_state, action)]
                for action in actions
            )
        best_actions = [
                action
                for action in actions
                if Q[(current_state, action)] == max_Q
            ]
        greedy_action = min(best_actions)
        reward, next_state = transitions[(current_state, greedy_action)]
        next_actions = [
            action
            for (state, action) in transitions
            if state == next_state
        ]
        max_next_Q = max(
            Q[(next_state, action)]
            for action in next_actions
        )
        best_next_actions = [
                action
                for action in next_actions
                if Q[(next_state, action)] == max_next_Q
            ]
        next_action = min(best_next_actions)
        delta = (
            reward -R_bar + Q[(next_state, next_action)] - Q[(current_state, greedy_action)]
        )
        R_bar += beta * delta

        Q[(current_state, greedy_action)] += alpha * delta

        current_state = next_state

    return Q, R_bar