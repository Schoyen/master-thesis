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


# Constants used by the popcount_64 algorithm
m_1 = 0x5555_5555_5555_5555
m_2 = 0x3333_3333_3333_3333
m_4 = 0x0F0F_0F0F_0F0F_0F0F
h_01 = 0x0101_0101_0101_0101


@numba.njit(cache=True, nogil=True, fastmath=True)
def popcount_64(num):
    num -= (num >> 1) & m_1
    num = (num & m_2) + ((num >> 2) & m_2)
    num = (num + (num >> 4)) & m_4

    return (num * h_01) >> 56


@numba.njit(cache=True, nogil=True, fastmath=True)
def state_diff(state_i, state_j):
    diff = state_i ^ state_j

    num_bits = 0
    for elem in diff:
        num_bits += popcount_64(elem)

    return num_bits


@numba.njit(cache=True, nogil=True, fastmath=True)
def get_index(state, index_num=0):
    index = 0

    for elem_p in range(len(state)):
        for p in range(BITSTRING_SIZE):
            if (state[elem_p] >> p) & 0b1 != 0:
                if index_num == 0:
                    return index

                index_num -= 1

            index += 1

    return -1
