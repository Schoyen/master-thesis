import numpy as np

from quantum_systems import ODQD

from configuration_interaction import CISD
from coupled_cluster.ccsd import CoupledClusterSinglesDoubles as CCSD
from hartree_fock import HartreeFock

n = 2
l = 20

odqd_ci = ODQD(n, l, 40, 201, alpha=0.1)
odqd_ci.setup_system()
odqd_ci._h = np.kron(odqd_ci._h, np.eye(l)) + np.kron(np.eye(l), odqd_ci._h)
odqd_ci._u.fill(0)

odqd_cc = ODQD(n, l, 40, 201, alpha=0.1)
odqd_cc.setup_system()
# odqd_cc._u.fill(0)

odqd_hf = ODQD(n, l, 40, 201, alpha=0.1)
odqd_hf.setup_system()
# odqd_hf._u.fill(0)

cisd = CISD(odqd_ci, verbose=True)
cisd.compute_ground_state()

ccsd = CCSD(odqd_cc, verbose=True)
ccsd.compute_ground_state(
    t_kwargs=dict(num_vecs=20), l_kwargs=dict(num_vecs=20)
)

hf = HartreeFock(odqd_hf, verbose=True)
hf.compute_ground_state()
