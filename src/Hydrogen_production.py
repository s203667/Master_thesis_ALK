import numpy as np
import matplotlib.pyplot as plt

def simple_h2_prod(eff_kWh_per_Nm3=3.86):

    """
    Convert power input to hydrogen production.

    Parameters
    ----------
    P_kW : array-like
        Power input [kW]
    eff_kWh_per_Nm3 : float
        Energy consumption [kWh/Nm3]

    Returns
    -------
    h_kg_per_h : array
        Hydrogen production [kg/h]
    """

    load = np.array([0.25,0.5,0.75,1.0])
    P_max = 4000
    Nm3_to_kg = 0.0899
    P_kW = load * P_max

    h = (P_kW / eff_kWh_per_Nm3) * Nm3_to_kg
    return P_kW, h

def electrolyzer_UI_curve(
        I_range,
        T=80.0,          # °C
        p=1.0,           # bar (or same units as in fitted d2)
        A=0.0625,          # m^2
        r1 = 0.7*4.45153E-05,#Changed fron andrea Gloppen. Based on Hysata 290 mOhm/cm2 at 25 °C, but no reason for the scale yet. 
        r2 = 0.7*6.88874E-09,
        d1 = 0.7*-3.12996E-06,
        d2 = 0.7*4.47137E-07,
        s = 0.7*0.33824,
        t1 = -0.01539,
        t2 = 2.00181,
        t3 = 15.24178,
        plot=False):

    """
    Simulates an electrolyzer polarization curve (U-I).

    Parameters
    ----------
    I_range : array
        Current range [A]
    T : float
        Temperature [°C]
    A : float
        Electrode area [m²]
    r1, r2 : float
        Ohmic fitted parameters
    s, t1, t2, t3 : float
        Activation fitted parameters
    plot : bool
        Whether to plot the U-I curve

    Returns
    -------
    j : current density [A/m²]
    U_cell : cell voltage [V]
    """

    # constants
    F = 96485
    R = 8.314

    # temperature
    T_K = T + 273.15

    # current density
    j = (I_range / A)
    j_plot = j*0.1 # Convert to mA/cm² for plotting

    # reversible voltage
    U_rev0 = (
        1.5184
        - 1.5421e-3 * T_K
        + 9.523e-5 * T_K * np.log(T_K)
        + 9.84e-8 * T_K**2
    )

    U_rev = U_rev0 + (R * T_K / (2 * F)) * np.log(1) #with log(1) = 0, this term is zero, but can be modified for different pressures


    # activation losses
    #U_act = s * np.log((t1 + (t2 / T_K) + (t3 / T_K**2)) * j + 1)

    # ohmic losses
    #U_ohm = (r1 + r2 * T_K) * j
    U_ohm = ((r1 + d1) + r2 * T + d2 * p) * j #From Andrea Gloppen code
    
    # total voltage

    activation_argument = (t1 + t2 / T + t3 / T**2) * j + 1.0
    # Avoid log of non-positive values
    activation_argument = np.maximum(activation_argument, 1e-12)
    U_act = s * np.log10(activation_argument)

    # Total cell voltage
    U_cell = U_rev + U_ohm + U_act

    if plot:
        plt.figure()
        plt.plot(j_plot, U_cell)
        plt.xlabel("Current density [mA/cm²]")
        plt.ylabel("Cell voltage [V]")
        plt.title("Electrolyzer polarization curve")
        plt.grid(True)
        plt.show()





    # print("U_rev (V):", U_rev)
    # print("U_act (V):", U_act)
    # print("U_ohm (V):", U_ohm)
    # print("U_cell (V):", U_cell)



    return j, j_plot, U_cell


def U_tn(T, p, W_t=0.30, w_z=0.0):
    T_K = T + 273.15
    b_tnv = 1.4756 #Baseline thermoneutral voltage at 0 °C
    z = 2
    F = 96485  # C/mol

    U_HHV = b_tnv + 2.252e-4 * T + 1.52e-8 * T**2 #baseline voltage + linear temp scale + non-linear tempscale

    #Equations 2.10-2.16 fron Chunjun phd below:

    #ln of Saturation vapor pressure of water at T
    ln_pw = 37.04 - 6276.0 / T_K - 3.416 * np.log(T_K) #2.14
    
    #Molality of electrolyte
    m = (W_t*(183.1221 - 0.56845*(T_K) +  984.5679*np.exp(W_t/115.96277))) / (100*56.105) #2.15

    #Vapor pressure of electrolyte
    p_w = np.exp(0.01621 - 0.138*m+0.1933*np.sqrt(m)+1.024*ln_pw) #2.13
    
    #Correction factor for vapor pressure
    phi = 1.5 * p_w / (p-p_w) #2.12

    #Temperature correction
    Y =  42.96 + 40.762*T - 0.06682*T**2 #2.16

    #Thermoneutral voltage:
    U_tn_eq = U_HHV + (phi / (z*F))*Y #2.10

    return U_tn_eq, U_HHV

def Eff_models(T, j, p, U_cell):
    f1 = 250 #mA^2 /cm^4
    f2 = 1
    f11 = 478645.74 # A2 m-4
    f12 = -2953.15 # A2 m-4 C-1
    f21 = 1.03960
    f22 = -0.00104 #C-1 
    j_mA_cm2 = j * 0.1 # Convert to mA/cm²
    eta_f = (j_mA_cm2**2)/(f1+j_mA_cm2**2)*f2 #(j**2 / (f11 + f12*T + j**2)) * (f21 + f22*T)  

    # Voltage efficiency: thermoneutral / actual cell voltage
    U_tn_eq, U_HHV = U_tn(T, p, W_t=0.30, w_z=0.0)  # scalar
    eta_v = U_tn_eq / U_cell                 # array over j

    eta_tot = eta_f * eta_v
    return eta_f, eta_v, eta_tot


def H2_rate(T, p, j, A_cell, N_cell, eta_f):
    z = 2
    F = 96485  # C/mol
    m_H2 = 2.0158e-3 # kg/mol

    I_cell = j*A_cell # Total current [A]
    # Calculate the hydrogen production rate in moles per second

    # Convert to kg/s (molar mass of H2 is approximately 2 g/mol)
    m_dot = eta_f * N_cell * I_cell/(z * F) * m_H2 * 3600.0 # Convert from kg/s to kg/h

    return m_dot

def electrolyzer_h2_prod_curve(T, I_range, A_cell, p, p_out, N_stacks, N_cell):
    j = I_range / A_cell
    _, _, U_cell = electrolyzer_UI_curve(I_range, T=T, p=p, A=A_cell, plot=False)
    eta_f, _, _ = Eff_models(T, j, p, U_cell)
    m_dot = N_stacks * H2_rate(T, p, j, A_cell, N_cell, eta_f)

    P_stack = N_stacks * N_cell * U_cell * I_range / 1000  # kW
    P_comp = compressor_power(T, p, p_out) * m_dot
    P_total = P_stack + P_comp


    return P_total, m_dot, P_stack

def compressor_power(T, p, p_out, eta_c=0.85, gamma=1.4):
    R = 8.314 # J/(mol*K)
    M_H2 = 2.0158e-3 # kg/mol
    T_K = T + 273.15

    # Calculate the specific work for isentropic compression
    P_C = R * T_K / M_H2 * gamma / (gamma-1) * 1 / eta_c * ((p_out/p)**((gamma-1)/gamma)-1) * 1E-03 / 3600 # compressor consumption in kWh/kg H2, changes in main


    return P_C



if __name__ == "__main__":
    # Example usage
    # system parameters
    A_cell = 0.0625
    T_static = 80
    p = 40
    p_out = 45
    N_cell = 250
    N_stacks = 40
 
    #I_range = np.linspace(5, 5500, 50)  # Current range from 5 to 300 A
    j_mA_cm2 = np.linspace(1, 400, 50)
    j = j_mA_cm2 * 10          # A/m²
    I_range = j * A_cell       # A
    j, j_plot, U_cell = electrolyzer_UI_curve(I_range)

    Nm3_to_kg = 0.0899                  # conversion factor
    M_H2 = 2.0158e-3 # kg/mol
    U_tn_eq, U_HHV = U_tn(T_static, p, W_t=0.30, w_z=0.0)

    # operating range
    #I_range = np.linspace(5, 5500, 50)  # A


    # get voltage curve
    temperatures = [55, 80, 110]   # three temperatures you want to compare
    colors = ["blue", "green", "red"]

    plt.figure(figsize=(10,6))

    for T_Plot, c in zip(temperatures, colors):
        j_T, j_plot_T, U_cell_T = electrolyzer_UI_curve(I_range, T=T_Plot, p=p, A=A_cell, plot=False)
        plt.plot(j_plot_T, U_cell_T, label=f"T = {T_Plot} °C", color=c)

    plt.xlabel("Current density [mA/cm²]")
    plt.ylabel("Cell voltage [V]")
    plt.title("Electrolyzer polarization curve at different temperatures")
    plt.grid(True)
    plt.legend()
    plt.show()
  


    P_total, m_dot = electrolyzer_h2_prod_curve(
    T_static, I_range, A_cell, p, p_out, N_stacks, N_cell
    )   

    # P_target = 4000 # Target power for scaling
    # P_max_current = np.max(P_total)
    # scale_factor = P_target / P_max_current
    # N_stacks_scaled = N_stacks * scale_factor
    # P_total = P_total * scale_factor
    # m_dot = m_dot * scale_factor

    #Recompute:
    #m_dot = N_stacks_scaled * H2_rate(T_static, p, j, A_cell, N_cell, eta_f)
    #P = N_stacks_scaled * N_cell * P_1cell


    eta_kg_per_MWh = (m_dot / P_total) * 1000
    HHV_kWh_per_kg = 39.4
    eta_p = m_dot * HHV_kWh_per_kg / P_total
    #eta_HHV = eta_f*(3600*M_H2*U_HHV)/(z*F*U_cell)

    # 🔹 Linear fit analysis
    coeffs = np.polyfit(P_total, m_dot, 1)  # Linear fit: y = slope*x + intercept
    slope, intercept = coeffs
    m_dot_fit = np.polyval(coeffs, P_total)
    
    # Calculate R-squared
    residuals = m_dot - m_dot_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((m_dot - np.mean(m_dot))**2)
    r_squared = 1 - (ss_res / ss_tot)

    eta_tot, eta_f, eta_v = Eff_models(T_static, j, p, U_cell)
    
    # 🔹 Plot 1: Electrolyzer model with efficiency and linear fit
    fig1, ax1_1 = plt.subplots(figsize=(10, 6))
    
    # Primary y-axis: Hydrogen production
    color1 = 'tab:blue'
    ax1_1.set_xlabel('Power input [kW]')
    ax1_1.set_ylabel('Hydrogen production [kg/h]', color=color1)
    ax1_1.plot(P_total, m_dot, 'o-', color=color1, markersize=4, label='Electrolyzer model')
    ax1_1.plot(P_total, m_dot_fit, '--r', linewidth=2, label=f'Linear fit (R² = {r_squared:.6f})')
    ax1_1.tick_params(axis='y', labelcolor=color1)
    ax1_1.grid(True)
    
    # Secondary y-axis: Efficiency
    ax1_2 = ax1_1.twinx()
    color2 = 'tab:orange'
    ax1_2.set_ylabel('Total efficiency [-]', color=color2)
    ax1_2.plot(P_total, eta_tot, 's-', color=color2, markersize=4, label='Total efficiency')
    ax1_2.tick_params(axis='y', labelcolor=color2)
    
    # Title and combined legend
    plt.title('Electrolyzer model with efficiency and linear fit')
    
    # Create combined legend
    lines1_1, labels1_1 = ax1_1.get_legend_handles_labels()
    lines1_2, labels1_2 = ax1_2.get_legend_handles_labels()
    ax1_1.legend(lines1_1 + lines1_2, labels1_1 + labels1_2, loc='lower right')
    
    plt.show()
    
    temperatures_H2_prod = [50, 60, 70, 80, 90, 100, 110]   # three temperatures you want to compare
    colors = ["blue", "green", "red", "cyan", "magenta", "orange", "purple"]
    # 🔹 Plot 2: Electrolyzer model vs Simple model comparison
    plt.figure(figsize=(10, 6))

    for T_plot, c in zip(temperatures_H2_prod, colors):
        P_temp, m_dot_temp = electrolyzer_h2_prod_curve(
            T_plot, I_range, A_cell, p, p_out, N_stacks, N_cell
        )
        plt.plot(P_temp, m_dot_temp, '-o', color=c, markersize=4, linewidth=1.5,
                 label=f'Electrolyzer model T={T_plot} °C')

    # Add the hydrogen_production curve
    P_simple, h_simple = simple_h2_prod(eff_kWh_per_Nm3=3.86)
    plt.plot(P_simple, h_simple, '^-', color='tab:cyan', markersize=6, linewidth=2, label='Shiptown Data')

    plt.xlabel('Power input [kW]')
    plt.ylabel('Hydrogen production [kg/h]')
    plt.title('Electrolyzer model vs Shiptown data comparison')
    plt.grid(True)
    plt.legend(loc='upper left', fontsize='small')

    plt.show()

    plt.figure()
    plt.plot(P_total, eta_kg_per_MWh, 'o-')
    plt.xlabel("Power input [kW]")
    plt.ylabel("Efficiency [kg/MWh]")
    plt.title("Electrolyzer efficiency (kg/MWh)")
    plt.grid()
    plt.show()


    print("Total efficiency at max current:", eta_tot[-1])
    E_spec_kWh_per_Nm3 = (P_total / (m_dot / Nm3_to_kg))  # kWh/Nm3

    print("Specific energy at max load [kWh/Nm3]:", E_spec_kWh_per_Nm3[-1])
    print("U_cell max:", U_cell[-1])
    print("new slope:", m_dot[-1] / P_total[-1])
    