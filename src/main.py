from Piecewise import generate_piecewise
import Hydrogen_production as hp
import numpy as np

A_cell = 0.0625 # m²
j_mA_cm2 = np.linspace(1, 400, 50)
j = j_mA_cm2 * 10          # A/m²
I_range = j * A_cell

A_s, B_s, P_seg_min, P_seg_max, P_total, m_dot = generate_piecewise(
    hp,
    T=80,
    I_range=I_range,
    A_cell=0.0625,
    p=40,
    p_out=45,
    N_stacks=40,
    N_cell=250,
    n_segments=6,
    plot=True   # 👈 super useful for debugging
)

