def termination_reason(success, done, truncated, steps, horizon):
    if success:
        return 'success'
    if done:
        return 'environment_done'
    if truncated:
        return 'environment_truncated'
    if steps >= horizon:
        return 'horizon'
    raise ValueError('Episode stopped before a valid terminal condition')
