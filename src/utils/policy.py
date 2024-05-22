def follow_policy(policy, height, width, initial_state, goal_states):
    action_dict = {0: "L", 1: "R", 2: "U", 3: "D"}
    state = initial_state
    actions_taken = []
    seen_states = []

    while len(seen_states) == 0 or (
        state not in goal_states and state not in seen_states
    ):
        seen_states.append(state)
        row, col = state // width, state % width
        action = policy[row, col]
        actions_taken.append(action_dict[action])

        if action == 0:  # left
            col = max(0, col - 1)
        elif action == 1:  # right
            col = min(width - 1, col + 1)
        elif action == 2:  # up
            row = max(0, row - 1)
        elif action == 3:  # down
            row = min(height - 1, row + 1)

        state = row * width + col

    #Add goal state to seen states.
    seen_states.append(state)
    return "".join(actions_taken), seen_states