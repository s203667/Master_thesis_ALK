from pyomo.environ import *
#import hydrogen_production as hp
import numpy as np
from pyomo.opt import TerminationCondition

class Expando:
    pass


class InputData:
    def __init__(self, T, S, I,
                 wind, wind_forecast, price_DA, price_H2, price_FCR, imbalance_price,
                 P_min, P_max, P_sb,
                 P_seg_min, P_seg_max,
                 A_s, B_s,
                 storage_max, storage_initial, grid_cap,
                 N_down, FCR_volume, z_on_prev=0, P_prev=0, is_DA = False, DA_export = None):

        self.T = T              # time steps
        self.S = S              # segments
        self.I = I              # FCR blocks

        self.wind = wind #Actual wind in real-time
        self.wind_forecast = wind_forecast
        self.price_DA = price_DA
        self.price_H2 = price_H2
        self.price_FCR = price_FCR
        self.imbalance_price = imbalance_price

        self.P_min = P_min
        self.P_max = P_max
        self.P_sb = P_sb
        

        self.P_seg_min = P_seg_min
        self.P_seg_max = P_seg_max

        self.A_s = A_s
        self.B_s = B_s

        self.storage_max = storage_max
        self.storage_initial = storage_initial
        self.grid_cap = grid_cap
        self.N_down = N_down
        self.P_min_bid = 1000 # Minimum bid size for FCR [kW]
        self.P_max_bid = FCR_volume  # array per block [kW], volumeUP from FCR data, adjust as needed


        self.z_on_prev = z_on_prev
        self.P_prev = P_prev

        self.is_DA = is_DA  # DA or real-time
        self.DA_export = DA_export

class ElectrolyzerModel:

    def __init__(self, data: InputData):
        self.data = data
        self.results = Expando()
        # Default values so model can build
  

        self._build_model()

    def set_initial_state(self, z_on_prev, P_prev):
        self.z_on_prev = z_on_prev
        self.P_prev = P_prev

    def _build_model(self):
        m = ConcreteModel() #?
        self.model = m
        d = self.data
        print("\n=== DEBUG: BUILDING MODEL ===")
        print("is_DA:", d.is_DA)

        if getattr(d, "wind", None) is not None:
            print("wind first 10:", d.wind[:10])
        else:
            print("NO d.wind (None)")

        if getattr(d, "wind_forecast", None) is not None:
            print("wind_forecast first 10:", d.wind_forecast[:10])
        else:
            print("NO d.wind_forecast (None)")

        print("price_DA first 10:", d.price_DA[:10])
        print("T:", d.T)
        print("I:", d.I)

        # --- Sets ---
        m.T = RangeSet(0, self.data.T - 1)
        m.S = RangeSet(0, self.data.S - 1)
        m.I = RangeSet(0, self.data.I - 1)

        # --- Variables ---
        m.P_seg = Var(m.T, m.S, domain=NonNegativeReals)
        m.z_seg = Var(m.T, m.S, domain=Binary)

        m.z_on = Var(m.T, domain=Binary)
        m.z_sb = Var(m.T, domain=Binary)
        m.z_off = Var(m.T, domain=Binary)

        m.P_buy = Var(m.T, domain=NonNegativeReals)
        m.P_DA = Var(m.T, domain=NonNegativeReals) #non-negative?
        m.P_grid = Var(m.T, domain=NonNegativeReals)#non-negative?


        m.deficit = Var(m.T, domain=NonNegativeReals)

        m.P_curt = Var(m.T, domain=NonNegativeReals)
        m.P_sb = Var(m.T, domain=NonNegativeReals)
        m.y_shutdown = Var(m.T, domain=Binary)

        m.h = Var(m.T, domain=NonNegativeReals)
        m.h_sell = Var(m.T, domain=NonNegativeReals)

        m.storage = Var(m.T, domain=NonNegativeReals)


        m.r_FCR_block = Var(m.I, domain=NonNegativeReals)
        m.z_FCR = Var(m.I, domain=Binary)


            # --- Parameters ---
                # wind: forecast in DA, actual in RT
        if d.is_DA:
            m.wind = Param(m.T, initialize=lambda m, t: d.wind_forecast[t])
    
        else:
            m.wind = Param(m.T, initialize=lambda m, t: d.wind[t])

        m.price_DA = Param(m.T, initialize=lambda m, t: d.price_DA[t])
        m.price_H2 = Param(m.T, initialize=lambda m, t: d.price_H2[t])
        m.price_FCR = Param(m.I, initialize=lambda m, i: d.price_FCR[i])

        if not d.is_DA:
            m.price_imb = Param(m.T, initialize=lambda m, t: d.imbalance_price[t])
            m.DA_export = Param(m.T, initialize=lambda m, t: d.DA_export[t])
                # P_grid[t] = wind_real[t] - electrolyzer_load[t]



        self.model = m

        self._build_constraints()
        self._build_objective()

    # --------------------------------------------------
    # CONSTRAINTS
    # --------------------------------------------------

    def _build_constraints(self):
        m = self.model
        d = self.data

        def map_t_to_i(t):
            block_size = 16  # 4 hours / 15 min
            return t // block_size

         #Piecewise bounds
        def seg_min_rule(m, t, s):
            return m.P_seg[t, s] >= d.P_seg_min[s] * m.z_seg[t, s]
        m.seg_min = Constraint(m.T, m.S, rule=seg_min_rule)

        def seg_max_rule(m, t, s):
            return m.P_seg[t, s] <= d.P_seg_max[s] * m.z_seg[t, s]
        m.seg_max = Constraint(m.T, m.S, rule=seg_max_rule)

        #Only one segment active when z_on=1. When z_on=0, all z_seg must be 0 and z_sb has its own constraint
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
                + m.r_FCR_block[map_t_to_i(t)]
                <= d.P_max * (m.z_on[t] + m.z_sb[t])
            )
        m.total_power = Constraint(m.T, rule=total_power_rule)

        # Minimum load constraint
        def min_load_rule(m, t):
            return (
                sum(m.P_seg[t, s] for s in m.S)
                + m.P_sb[t]
                -  m.r_FCR_block[map_t_to_i(t)]
                >= d.P_min * (m.z_on[t] + m.z_sb[t])
            )
        m.min_load = Constraint(m.T, rule=min_load_rule)

        #Hydrogen production
        def h2_rule(m, t):
            dt = 0.25
            return m.h[t] == sum(d.A_s[s] * m.P_seg[t, s] + d.B_s[s] * m.z_seg[t, s] for s in m.S)
        m.h2_prod = Constraint(m.T, rule=h2_rule)


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
                m.z_off[k] for k in range(t, min(t + d.N_down, d.T))
            ) >= d.N_down * m.y_shutdown[t]
        m.downtime = Constraint(m.T, rule=downtime_rule)

        # Storage dynamics 
        def storage_rule(m, t):
            if t == 0:
                return m.storage[t] == d.storage_initial
            return m.storage[t] == m.storage[t-1] + m.h[t] - m.h_sell[t]
        m.storage_dyn = Constraint(m.T, rule=storage_rule)

        # Storage bounds 
        def storage_bound_rule(m, t):
            return m.storage[t] <= d.storage_max
        m.storage_bound = Constraint(m.T, rule=storage_bound_rule)

        def h_sell_prod_limit(m, t):
            if t == 0:
                return m.h_sell[t] <= m.h[t] + d.storage_initial
            return m.h_sell[t] <= m.h[t] + m.storage[t-1]  # Can only sell what is produced plus what's in storage
        m.h_sell_prod_limit = Constraint(m.T, rule=h_sell_prod_limit)



        #--- add storage start and end constraints ---

        # --- WIND–ELECTROLYZER POWER BALANCE ---
        if d.is_DA:
            export_var = m.P_DA
        else:
            export_var = m.P_grid

        def wind_balance_rule(m, t):
            return (
                m.wind[t] + m.P_buy[t]
                ==
                sum(m.P_seg[t, s] for s in m.S)
                + m.P_sb[t]
                + m.P_curt[t]
                + export_var[t]
            )
        m.wind_balance = Constraint(m.T, rule=wind_balance_rule)

        # --- GRID BUY LOGIC ---
        # def grid_sell_rule(m, t):
        #     return m.P_DA[t] <= d.grid_cap
        # m.grid_sell = Constraint(m.T, rule=grid_sell_rule)


        def deficit_rule(m, t):
            return m.deficit[t] >= (
                sum(m.P_seg[t, s] for s in m.S)
                + m.P_sb[t]
                - m.wind[t]
            )
        m.deficit_con = Constraint(m.T, rule=deficit_rule)

        def buy_limit_deficit_rule(m, t):
            return m.P_buy[t] <= m.deficit[t]
        m.buy_limit_deficit = Constraint(m.T, rule=buy_limit_deficit_rule)

        def buy_limit_grid_rule(m, t):
            return m.P_buy[t] <= d.grid_cap
        m.buy_limit_grid = Constraint(m.T, rule=buy_limit_grid_rule)


        def fcr_min_rule(m, i):
            return m.r_FCR_block[i] >= d.P_min_bid * m.z_FCR[i]
        m.fcr_min = Constraint(m.I, rule=fcr_min_rule)

        def fcr_max_rule(m, i):
            upper_bound = min(d.P_max, d.P_max_bid[i])
            return m.r_FCR_block[i] <= upper_bound * m.z_FCR[i]
        m.fcr_max = Constraint(m.I, rule=fcr_max_rule)


        # def fcr_on_block_rule(m, i):
        #     return m.z_FCR[i] <= m.z_on[i*16]
        # m.fcr_on_block = Constraint(m.I, rule=fcr_on_block_rule)

        def fcr_on_block_rule(m, i):
            t = i * 16
            if t >= d.T:
                return Constraint.Skip
            return m.z_FCR[i] <= m.z_on[t]

        m.fcr_on_block = Constraint(m.I, rule=fcr_on_block_rule)

        # ramping consraints
        def ramp_up_rule(m, t):
            dR_up = 400000 # kW/15Min #PLACEHOLDER
            if t == 0:
                return (
                    sum(m.P_seg[0, s] for s in m.S)
                    - d.P_prev
                    <= dR_up
                )
            return (
                sum(m.P_seg[t, s] for s in m.S)
                - sum(m.P_seg[t-1, s] for s in m.S)
                <= dR_up
            )
        m.ramp_up = Constraint(m.T, rule=ramp_up_rule)

        def ramp_down_rule(m, t):
            dR_down = 400000 # kW/15min #PLACEHOLDER
            if t == 0:
                return (
                    d.P_prev
                    - sum(m.P_seg[0, s] for s in m.S)
                    <= dR_down
                )
            return (
                sum(m.P_seg[t-1, s] for s in m.S)
                - sum(m.P_seg[t, s] for s in m.S)
                <= dR_down
            )
        m.ramp_down = Constraint(m.T, rule=ramp_down_rule)

       
        def P_DA_rule(m, t):
            return m.P_DA[t] == m.wind[t] - (
                sum(m.P_seg[t, s] for s in m.S) + m.P_sb[t]
            )
        
        m.P_DA_con = Constraint(m.T, rule=P_DA_rule)

        def P_grid_rule(m, t):
            return m.P_grid[t] == m.wind[t] - (
                sum(m.P_seg[t, s] for s in m.S) + m.P_sb[t]
            )

        m.P_grid_con = Constraint(m.T, rule=P_grid_rule)


        #Rolling horizon constraints:
        def initial_on_rule(m):
            return m.z_on[0] == d.z_on_prev
        m.initial_on = Constraint(rule=initial_on_rule)

    def _build_objective(self):
        m = self.model
        d = self.data
        dt = 0.25
        dt_FCR = 16

        if d.is_DA:
            # DAY-AHEAD: forecast-based, no imbalance
            def obj_rule(m):
                return sum(
                    m.h_sell[t] * d.price_H2[t] * dt
                    - m.P_buy[t] * d.price_DA[t] * dt / 1000
                    + m.P_DA[t] * d.price_DA[t] * dt / 1000
                    for t in m.T
                ) + sum(
                    m.r_FCR_block[i] * d.price_FCR[i] * dt_FCR
                    for i in m.I
                )
        else:
            # REAL-TIME: actual-based, P_grid fixed, imbalance included
            def obj_rule(m):
                return sum(
                    m.h_sell[t] * d.price_H2[t] * dt
                    - m.P_buy[t] * d.price_DA[t] * dt / 1000
                    # DA revenue on fixed P_grid
                   + m.DA_export[t] * d.price_DA[t] * dt / 1000
                    # imbalance on deviation between actual export s(P_DA) and P_grid
                    - (m.DA_export[t] - m.P_grid[t]) * m.price_imb[t] * dt / 1000
                        for t in m.T
                ) + sum(
                    m.r_FCR_block[i] * d.price_FCR[i] * dt_FCR
                    for i in m.I
                )

        m.obj = Objective(rule=obj_rule, sense=maximize)


    def run(self):
        solver = SolverFactory('gurobi')
        solver.options['DualReductions'] = 0
        results = solver.solve(self.model, tee=True)

        tc = results.solver.termination_condition
        print("Termination condition:", tc)

        if tc == TerminationCondition.optimal:
            self._save_results()
            return

        print("Model is NOT optimal:", tc)

        # Try IIS
        try:
            solver = SolverFactory('gurobi')
            solver.options['IISMethod'] = 1
            solver.solve(self.model, tee=True, symbolic_solver_labels=True)
            
        except:
            pass

        # Debug constraints
        #self.debug_constraints()

    def _save_results(self):
        m = self.model

        self.results.P_DA = np.array([value(m.P_DA[t]) for t in m.T]) if hasattr(m, "P_DA") else None
        self.results.P_grid = np.array([value(m.P_grid[t]) for t in m.T]) if hasattr(m, "P_grid") else None
        self.results.h = [value(m.h[t]) for t in m.T]
        self.results.storage = [value(m.storage[t]) for t in m.T]
        self.results.z_on = [value(m.z_on[t]) for t in m.T]  # On/off status
        self.results.z_sb = [value(m.z_sb[t]) for t in m.T]  # Standby status
        self.results.z_off = [value(m.z_off[t]) for t in m.T]  # Off status
        self.results.P_buy = [value(m.P_buy[t]) for t in m.T]  # Electricity bought
        self.results.P_curt = [value(m.P_curt[t]) for t in m.T]  # Curtailed power
        self.results.r_FCR_block = [value(m.r_FCR_block[i]) for i in m.I]  # FCR block power
        self.results.h_sell = [value(m.h_sell[t]) for t in m.T]  # Hydrogen sold
        self.results.P_elec = np.array([
        sum(m.P_seg[t, s].value for s in m.S) + m.P_sb[t].value
        for t in m.T
    ])
        self.results.objective_value = value(m.obj)
        # print("Saved results: P_DA length =", len(self.results.P_DA) if self.results.P_DA is not None else None)
        # print("\n=== IIS CONSTRAINTS ===")
        # for c in m.component_objects(Constraint, active=True):
        #     for idx in c:
        #         if c[idx].active and c[idx].has_ub() and c[idx].has_lb():
        #             if c[idx].lbody() > c[idx].ub() + 1e-6 or c[idx].ub() < c[idx].lbody() - 1e-6:
        #                 print("VIOLATED:", c.name, idx, value(c[idx].body))
