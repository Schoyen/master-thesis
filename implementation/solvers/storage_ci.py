import numpy as np
from configuration_interaction import num_states, BITTYPE

n = 4
l = 80


def compute_info(n, l, order):
    N_i = np.dtype(BITTYPE).itemsize
    num_dets = 0
    m = l - n

    for excitation in range(order + 1):
        num_dets += num_states(n, m, excitation)

    det_storage = num_dets * N_i
    H_storage = (
        num_dets ** 2 * np.dtype(np.complex128).itemsize
    )
    H_storage /= 2 ** 30  # Gigabytes

    return num_dets, det_storage, H_storage


trunc = ["CIS", "CISD", "CISDT", "CISDTQ"]

for i, order in enumerate(range(1, 5)):
    N_s, det_s, H_s = compute_info(n, l, order)
    print(
        f"""
{trunc[i]}:
    N_s = {N_s},
    Det storage = {det_s} B,
    H storage = {H_s} GB
"""
    )
