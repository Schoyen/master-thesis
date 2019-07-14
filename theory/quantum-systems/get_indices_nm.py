import numba


@numba.njit(cache=True, nogil=True)
def get_indices_nm(p):
    n, m = 0, 0
    previous_shell = 0
    current_shell = 1
    shell_counter = 1

    # Check if p is in the current shell
    while current_shell <= p:
        # Increment shell counter
        shell_counter += 1
        # Store number of states in previous and current
        # shell
        previous_shell = current_shell
        current_shell = previous_shell + shell_counter

    # Compute the index of the middle of previous and
    # current shell
    middle = (
        current_shell - previous_shell
    ) / 2 + previous_shell

    # Check if current shell has an odd number of states
    # and if it is in the middle of the shell
    if (current_shell - previous_shell) & 0x1 == 1 and abs(
        p - math.floor(middle)
    ) < 1e-8:
        n = shell_counter // 2
        m = 0

        return n, m

    # Check if m is negative
    if p < middle:
        # Count up from previous shell
        n = p - previous_shell
        m = -((shell_counter - 1) - 2 * n)

    # Here m is positive
    else:
        # Count down from next shell
        n = (current_shell - 1) - p
        m = (shell_counter - 1) - 2 * n

    return n, m
