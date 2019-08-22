from run_simulation import run_tdccsd_simulation


run_tdccsd_simulation(
    "li 0.0 0.0 0.0; h 0.0 0.0 3.08",
    "lih",
    t_end=20,
    basis="ccpvdz",
    cache_freq=10,
)
