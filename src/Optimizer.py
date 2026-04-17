from pyomo.environ import *
#import hydrogen_production as hp
import numpy as np

class Expando:
    pass


class InputData:
    def __init__(self, T, S, I,
                 wind, price_DA, price_H2, price_FCR,
                 P_min, P_max, P_sb,
                 P_seg_min, P_seg_max,
                 A_s, B_s,
                 storage_max, storage_initial,
                 N_down):

        self.T = T              # time steps
        self.S = S              # segments
        self.I = I              # FCR blocks

        self.wind = wind
        
        self.price_DA = price_DA
        self.price_H2 = price_H2
        self.price_FCR = price_FCR

        self.P_min = P_min
        self.P_max = P_max
        self.P_sb = P_sb
        

        self.P_seg_min = P_seg_min
        self.P_seg_max = P_seg_max

        self.A_s = A_s
        self.B_s = B_s

        self.storage_max = storage_max
        self.storage_initial = storage_initial
        self.N_down = N_down
        self.P_min_bid = 1 # Minimum bid size for FCR [MW]

class ElectrolyzerModel:

    def __init__(self, data: InputData):
        self.data = data
        self.results = Expando()
        self._build_model()

    def _build_model(self):
        m = ConcreteModel()

        # --- Sets ---
        m.T = RangeSet(0, len(self.data.T) - 1)
        m.S = RangeSet(0, len(self.data.S) - 1) #Change this probably
        m.I = RangeSet(0, len(self.data.I) - 1)

        # --- Variables ---
        m.P_seg = Var(m.T, m.S, domain=NonNegativeReals)
        m.z_seg = Var(m.T, m.S, domain=Binary)

        m.z_on = Var(m.T, domain=Binary)
        m.z_sb = Var(m.T, domain=Binary)
        m.z_off = Var(m.T, domain=Binary)

        m.P_buy = Var(m.T, domain=NonNegativeReals)
        m.P_sell = Var(m.T, domain=NonNegativeReals)
        m.z_grid_buy = Var(m.T, domain=Binary)

        m.P_curt = Var(m.T, domain=NonNegativeReals)
        m.P_sb = Var(m.T, domain=NonNegativeReals)
        m.y_shutdown = Var(m.T, domain=Binary)

        m.h = Var(m.T, domain=NonNegativeReals)
        m.h_sell = Var(m.T, domain=NonNegativeReals)

        m.storage = Var(m.T, domain=NonNegativeReals)


        m.r_FCR_block = Var(m.I, domain=NonNegativeReals)
        m.z_FCR = Var(m.I, domain=Binary)
        m.r_FCR_t = Var(m.T, domain=NonNegativeReals)

        self.model = m

        self._build_constraints()
        self._build_objective()

    # --------------------------------------------------
    # CONSTRAINTS
    # --------------------------------------------------

    def _build_constraints(self):
        m = self.model
        d = self.data


        
        ### Map time steps to FCR blocks ##

        def map_t_to_i(t):
            block_size = 16  # 4 hours / 15 min
            return t // block_size

        # def fcr_constant_rule(m, t):
        #     if t == 0:
        #         return Constraint.Skip
        #     if map_t_to_i(t) == map_t_to_i(t-1):
        #         return m.r_FCR_t[t] == m.r_FCR_t[t-1]
        #     return Constraint.Skip

        # m.fcr_constant = Constraint(m.T, rule=fcr_constant_rule)

        # Link FCR power to block variables
        def fcr_link_rule(m, t):
            i = map_t_to_i(t)
            return m.r_FCR_t[t] == m.r_FCR_block[i]

        m.fcr_link = Constraint(m.T, rule=fcr_link_rule)

         #Piecewise bounds
        def seg_bounds_rule(m, t, s):
            return (d.P_seg_min[s] * m.z_seg[t, s],
                    m.P_seg[t, s],
                    d.P_seg_max[s] * m.z_seg[t, s])
        m.seg_bounds = Constraint(m.T, m.S, rule=seg_bounds_rule)

        #Only one segment active
        def one_seg_rule(m, t):
            return sum(m.z_seg[t, s] for s in m.S) == m.z_on[t]
        m.one_seg = Constraint(m.T, rule=one_seg_rule)

        #Only one state active
        def state_rule(m, t):
            return m.z_on[t] + m.z_sb[t] + m.z_off[t] == 1
        m.state = Constraint(m.T, rule=state_rule)

         #Total power
        def total_power_rule(m, t):
            return (
                sum(m.P_seg[t, s] for s in m.S)
                + m.P_sb[t]
                + m.r_FCR_t[t]
                <= d.P_max * m.z_on[t]
            )
        m.total_power = Constraint(m.T, rule=total_power_rule)

        # Minimum load constraint
        def min_load_rule(m, t):
            return (
                sum(m.P_seg[t, s] for s in m.S)
                + m.P_sb[t]
                - m.r_FCR_t[t]
                >= d.P_min * m.z_on[t]
            )
        m.min_load = Constraint(m.T, rule=min_load_rule)

        #Hydrogen production
        def h2_rule(m, t):
            return m.h[t] == sum(d.A_s[s] * m.P_seg[t, s] + d.B_s[s] * m.z_seg[t, s] for s in m.S)
        m.h2_prod = Constraint(m.T, rule=h2_rule)

                # --- Standby power --- #Consider removing this
        def standby_rule(m, t):
            return sum(m.P_seg[t, s] for s in m.S) == 0 if m.z_sb[t] == 1 else Constraint.Skip
        m.standby = Constraint(m.T, rule=standby_rule) #Consider removing this

        #Standby power
        def standby_power_rule(m, t):
            return m.P_sb[t] == d.P_sb * m.z_sb[t]
        m.standby_power = Constraint(m.T, rule=standby_power_rule)


        #shutdown
        def shutdown_def(m, t):
            if t == 0:
                return Constraint.Skip
            return m.y_shutdown[t] >= m.z_off[t] - m.z_off[t-1]
        m.shutdown_def = Constraint(m.T, rule=shutdown_def)

        # Minimum downtime
        def downtime_rule(m, t):
            return sum(
                m.z_off[k] for k in range(t, min(t + d.N_down, len(d.T)))
            ) >= d.N_down * m.y_shutdown[t]
        m.downtime = Constraint(m.T, rule=downtime_rule)

        # Storage dynamics 
        def storage_rule(m, t):
            if t == 0:
                return m.storage[t] == 0
            return m.storage[t] == m.storage[t-1] + m.h[t] - m.h_sell[t]
        m.storage_dyn = Constraint(m.T, rule=storage_rule)

        # Storage bounds 
        def storage_bound_rule(m, t):
            return m.storage[t] <= d.storage_max
        m.storage_bound = Constraint(m.T, rule=storage_bound_rule)

        #--- add storage start and end constraints ---
        #Initial storage
        def storage_init_rule(m):
            return m.storage[0] == d.storage_initial
        m.storage_init = Constraint(rule=storage_init_rule)

        #Terminal storage
        def storage_terminal_rule(m):
            return m.storage[len(d.T)-1] >= d.storage_initial
        m.storage_terminal = Constraint(rule=storage_terminal_rule)

        # Power balance
        def power_balance_rule(m, t):
            return (
                d.wind[t] + m.P_buy[t]
                ==
                sum(m.P_seg[t, s] for s in m.S)
                + m.P_sb[t]
                + m.P_sell[t]
                + m.P_curt[t]
            )
        m.power_balance = Constraint(m.T, rule=power_balance_rule)

        # Simultanous grid buy and sell
        def grid_buy_rule(m, t):
            return m.P_buy[t] <= d.P_max * m.z_grid_buy[t]
        m.grid_buy = Constraint(m.T, rule=grid_buy_rule)

        def grid_sell_rule(m, t):
            return m.P_sell[t] <= d.P_max * (1 - m.z_grid_buy[t])
        m.grid_sell = Constraint(m.T, rule=grid_sell_rule)


        def fcr_bounds_rule(m, i):
            return (
                d.P_min_bid * m.z_FCR[i] ,
                m.r_FCR_block[i],
                d.P_max * m.z_FCR[i]
            )
        m.fcr_bounds = Constraint(m.I, rule=fcr_bounds_rule)



        # FCR can only be provided when state is on / operating
        def fcr_on_rule(m, t):
            return m.r_FCR_t[t] <= d.P_max * m.z_on[t]
        m.fcr_on = Constraint(m.T, rule=fcr_on_rule)

        # ramping consraints
        def ramp_up_rule(m, t):
            dR_up = 20 # MW/15Min #PLACEHOLDER
            if t == 0:
                return Constraint.Skip
            return (
                sum(m.P_seg[t, s] for s in m.S)
                - sum(m.P_seg[t-1, s] for s in m.S)
                <= dR_up
            )
        m.ramp_up = Constraint(m.T, rule=ramp_up_rule)

        def ramp_down_rule(m, t):
            dR_down = 20 # MW/15min #PLACEHOLDER
            if t == 0:
                return Constraint.Skip
            return (
                sum(m.P_seg[t-1, s] for s in m.S)
                - sum(m.P_seg[t, s] for s in m.S)
                <= dR_down
            )
        m.ramp_down = Constraint(m.T, rule=ramp_down_rule)

    def _build_objective(self):
        m = self.model
        d = self.data

        def obj_rule(m):
            return sum(
                m.h[t] * d.price_H2[t]
                - m.P_buy[t] * d.price_DA[t]
                + m.P_sell[t] * d.price_DA[t]
                for t in m.T
            ) + sum(m.r_FCR_block[i] * d.price_FCR[i] * 4 
                    for i in m.I)
                    #Times four because FCR is paid per block and each block is 4 hours
        m.obj = Objective(rule=obj_rule, sense=maximize)


    def run(self):
        solver = SolverFactory('gurobi')
        solver.solve(self.model, tee=True)
        self._save_results()

    def _save_results(self):
        m = self.model
        self.results.h = [value(m.h[t]) for t in m.T]
        self.results.storage = [value(m.storage[t]) for t in m.T]
        
