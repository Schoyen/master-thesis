@numba.njit(cache=True, nogil=True)
def get_index_p(n, m):
    # Compute shell number
    num_shells = 2 * n + abs(m) + 1

    # Count the number of states up to num_shells
    previous_shell = 0
    for i in range(1, num_shells):
        previous_shell += i

    current_shell = previous_shell + num_shells

    if m == 0:
        if n == 0:
            # Lowest state
            return 0

        # We are in the middle of a shell
        p = (
            previous_shell
            + (current_shell - previous_shell) // 2
        )

        return p

    elif m < 0:
        # Count the number states from previous shell
        return previous_shell + n

    else:
        # Count the number states from next shell
        return current_shell - (n + 1)
