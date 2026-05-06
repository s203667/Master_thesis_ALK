#generate piecewise linear approximation of hydrogen production curve

import numpy as np


def generate_piecewise(
    hp,
    T,
    I_range,
    A_cell,
    p,
    p_out,
    N_stacks,
    N_cell,
    n_segments=5,
    plot=False
):
    """
    Generate piecewise linear approximation of hydrogen production curve.

    Returns:
        A_s         : slopes [kg/kWh]
        B_s         : intercepts [kg/h]
        P_seg_min   : lower bounds per segment [kW]
        P_seg_max   : upper bounds per segment [kW]
        P_total     : full nonlinear power curve
        m_dot       : full nonlinear hydrogen production
    """

    # -------------------------
    # 1. Generate nonlinear curve
    # -------------------------
    P_total, m_dot, P_stack = hp.electrolyzer_h2_prod_curve(
        T, I_range, A_cell, p, p_out, N_stacks, N_cell
    )

    # -------------------------
    # 2. Sort by power
    # -------------------------
    idx = np.argsort(P_total)
    P_total = P_total[idx]
    m_dot = m_dot[idx]

    # Safety check
    if len(P_total) < n_segments:
        raise ValueError("Number of segments exceeds data points")

    # -------------------------
    # 3. Split into segments
    # -------------------------
    indices = np.arange(len(P_total))
    segments = np.array_split(indices, n_segments)

    A_s = []
    B_s = []
    P_seg_min = []
    P_seg_max = []

    # -------------------------
    # 4. Fit linear model per segment
    # -------------------------
    for seg in segments:
        P_seg = P_total[seg]
        m_seg = m_dot[seg]

        # Linear fit: m = A * P + B
        A, B = np.polyfit(P_seg, m_seg, 1)

        A_s.append(A)
        B_s.append(B)

        P_seg_min.append(float(P_seg.min()))
        P_seg_max.append(float(P_seg.max()))

    # -------------------------
    # 5. Optional plotting
    # -------------------------
    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(P_total, m_dot, 'o', label='Hydrogen model')

        for s in range(n_segments):
            P_line = np.linspace(P_seg_min[s], P_seg_max[s], 50)
            m_line = A_s[s] * P_line + B_s[s]
            plt.plot(P_line, m_line, '-', label=f'Segment {s}')

        plt.xlabel("Power [kW]")
        plt.ylabel("Hydrogen production [kg/h]")
        plt.title("Piecewise linear approximation")
        plt.grid(True)
        plt.legend()
        plt.show()

    return A_s, B_s, P_seg_min, P_seg_max, P_total, m_dot, P_stack
