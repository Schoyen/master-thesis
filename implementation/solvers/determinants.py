# Value of state
# state == [0b0, 0b0]

# The number of bits in an integer
BITSTRING_SIZE = 64
# Compute the integer index in state
elem_i = 68 // BITSTRING_SIZE

# Set the fourth bit in the second integer in state
state[elem_i] |= 1 << (4 - elem_i * BITSTRING_SIZE)

# New value of state
# state == [0b0, 0b10000] == [0, 16]


@numba.njit(cache=True, nogil=True, fastmath=True)
def _excite_state(state, o_remove, v_insert):
    for i, a in zip(o_remove, v_insert):
        elem_i = i // BITSTRING_SIZE
        elem_a = a // BITSTRING_SIZE

        state[elem_i] ^= 1 << (i - elem_i * BITSTRING_SIZE)
        state[elem_a] |= 1 << (a - elem_a * BITSTRING_SIZE)


@numba.njit(cache=True, nogil=True, fastmath=True)
def _create_excited_states(
    n, l, states, index, order, o_remove, v_insert
):
    if order == 0:
        _excite_state(states[index], o_remove, v_insert)
        return index + 1

    i_start = (
        0 if len(o_remove) == order else o_remove[order] + 1
    )
    a_start = (
        n if len(v_insert) == order else v_insert[order] + 1
    )

    for i in range(i_start, n):
        o_remove[order - 1] = i
        for a in range(a_start, l):
            v_insert[order - 1] = a

            index = _create_excited_states(
                n,
                l,
                states,
                index,
                order - 1,
                o_remove,
                v_insert,
            )

    return index


@numba.njit(cache=True, nogil=True, fastmath=True)
def compute_sign(state, p):
    elem_i = 0
    k = 0

    for i in range(p):
        if (i - elem_i * BITSTRING_SIZE) >= BITSTRING_SIZE:
            elem_i += 1

        k += (
            state[elem_i] >> (i - elem_i * BITSTRING_SIZE)
        ) & 1

    return (-1) ** k


@numba.njit(cache=True, nogil=True, fastmath=True)
def occupied_index(state, p):
    elem_p = p // BITSTRING_SIZE

    return (
        state[elem_p] & (1 << (p - elem_p * BITSTRING_SIZE))
    ) != 0
