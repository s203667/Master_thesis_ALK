from pyomo.environ import *
import numpy as np
from pyomo.opt import TerminationCondition


class Expando:
    pass


class StochasticInputData:
    def __init__(self, T, S_seg, I,
                 wind_forecast,
                 price_scenarios,    # (N_sc, T) DA price scenarios
                 wind_scenarios,     # (N_sc, T) wind scenarios
                 imb_scenarios,      # (N_sc, T) imbalance price scenarios
                 scenario_probs,     # (N_sc,)
                 price_H2, price_FCR,
                 P_min, P_max, P_sb,
                 P_seg_min, P_seg_max,
                 A_s, B_s,
                 storage_max, storage_initial,
                 h_min_daily, h_sell_max, grid_cap,
                 N_down, FCR_volume,
                 z_on_prev=0, P_prev=0):

        self.T = T
        self.S_seg = S_seg
        self.I = I
        self.N_sc = len(scenario_probs)

        self.wind_forecast   = wind_forecast
        self.price_scenarios = price_scenarios
        self.wind_scenarios  = wind_scenarios
        self.imb_scenarios   = imb_scenarios
        self.scenario_probs  = scenario_probs

        self.price_H2  = price_H2
        self.price_FCR = price_FCR

        self.P_min = P_min
        self.P_max = P_max
        self.P_sb  = P_sb
        self.P_seg_min = P_seg_min
        self.P_seg_max = P_seg_max
        self.A_s = A_s
        self.B_s = B_s

        self.storage_max     = storage_max
        self.storage_initial = storage_initial
        self.h_min_daily     = h_min_daily
        self.h_sell_max      = h_sell_max
        self.grid_cap        = grid_cap
        self.N_down          = N_down
        self.P_min_bid       = 1000
        self.P_max_bid       = FCR_volume

        self.z_on_prev = z_on_prev
        self.P_prev    = P_prev


class StochasticElectrolyzerModel:

    def __init__(self, data: StochasticInputData):
        self.data    = data
        self.results = Expando()
        self._build_model()

    # ------------------------------------------------------------------
    # BUILD
    # ------------------------------------------------------------------

    def _build_model(self):
        m = ConcreteModel()
        self.model = m
        d = self.data

        # --- Sets ---
        m.T   = RangeSet(0, d.T - 1)
        m.SEG = RangeSet(0, d.S_seg - 1)
        m.I   = RangeSet(0, d.I - 1)
        m.SC  = RangeSet(0, d.N_sc - 1)

        # --- Scenario parameters ---
        m.prob      = Param(m.SC, initialize=lambda m, sc: d.scenario_probs[sc])
        m.price_sc  = Param(m.SC, m.T,
                            initialize=lambda m, sc, t: d.price_scenarios[sc][t])
        m.wind_sc   = Param(m.SC, m.T,
                            initialize=lambda m, sc, t: d.wind_scenarios[sc][t])
        m.imb_sc    = Param(m.SC, m.T,
                            initialize=lambda m, sc, t: d.imb_scenarios[sc][t])

        # --- Known parameters ---
        m.wind_fc  = Param(m.T, initialize=lambda m, t: d.wind_forecast[t])
        m.price_H2 = Param(m.T, initialize=lambda m, t: d.price_H2[t])
        m.price_FCR = Param(m.I, initialize=lambda m, i: d.price_FCR[i])

        self._build_variables()
        self._build_constraints()
        self._build_objective()

    def _build_variables(self):
        m = self.model
        d = self.data

        # ---- STAGE 1: scenario-independent ----
        m.P_DA        = Var(m.T, bounds=(-d.P_min, None))
        m.r_FCR_block = Var(m.I, domain=NonNegativeReals)
        m.z_FCR       = Var(m.I, domain=Binary)
        m.z_on_commit = Var(m.I, domain=Binary)


        # ---- STAGE 2: scenario-indexed ----
        # Electrolyzer state
        m.z_on  = Var(m.SC, m.T, domain=Binary)
        m.z_sb  = Var(m.SC, m.T, domain=Binary)
        m.z_off = Var(m.SC, m.T, domain=Binary)
        m.y_shutdown = Var(m.SC, m.T, domain=Binary)

        # Piecewise
        m.z_seg = Var(m.SC, m.T, m.SEG, domain=Binary)
        m.P_seg = Var(m.SC, m.T, m.SEG, domain=NonNegativeReals)
        m.P_sb  = Var(m.SC, m.T, domain=NonNegativeReals)

        # Power flows
        m.P_grid  = Var(m.SC, m.T, bounds=(-d.grid_cap, None))
        #m.P_buy   = Var(m.SC, m.T, domain=NonNegativeReals)
        m.P_curt  = Var(m.SC, m.T, domain=NonNegativeReals)

        # Hydrogen
        m.h       = Var(m.SC, m.T, domain=NonNegativeReals)
        m.h_sell  = Var(m.SC, m.T, domain=NonNegativeReals)
        m.storage = Var(m.SC, m.T, domain=NonNegativeReals)

        # Imbalance deviation (for clarity in objective)
        m.dev_pos = Var(m.SC, m.T, domain=NonNegativeReals)
        m.dev_neg = Var(m.SC, m.T, domain=NonNegativeReals)

    # ------------------------------------------------------------------
    # CONSTRAINTS
    # ------------------------------------------------------------------

    def _build_constraints(self):
        m = self.model
        d = self.data

        def i_of_t(t):
            return t // 16

        # ==============================================================
        # STAGE 1 CONSTRAINTS
        # ==============================================================

        def fcr_min_rule(m, i):
            return m.r_FCR_block[i] >= d.P_min_bid * m.z_FCR[i]
        m.fcr_min = Constraint(m.I, rule=fcr_min_rule)

        def fcr_max_rule(m, i):
            return m.r_FCR_block[i] <= min(d.P_max, d.P_max_bid[i]) * m.z_FCR[i]
        m.fcr_max = Constraint(m.I, rule=fcr_max_rule)

        # ==============================================================
        # STAGE 2 CONSTRAINTS  (all indexed over m.SC)
        # ==============================================================

        # --- State ---
        def state_rule(m, sc, t):
            return m.z_on[sc,t] + m.z_sb[sc,t] + m.z_off[sc,t] == 1
        m.state = Constraint(m.SC, m.T, rule=state_rule)

        def one_seg_rule(m, sc, t):
            return sum(m.z_seg[sc,t,s] for s in m.SEG) == m.z_on[sc,t]
        m.one_seg = Constraint(m.SC, m.T, rule=one_seg_rule)

        def initial_on_rule(m, sc):
            return m.z_on[sc, 0] <= d.z_on_prev
        m.initial_on = Constraint(m.SC, rule=initial_on_rule)

        # --- Shutdown and downtime ---
        def shutdown_rule(m, sc, t):
            if t == 0:
                return Constraint.Skip
            return m.y_shutdown[sc,t] >= m.z_off[sc,t] - m.z_off[sc,t-1]
        m.shutdown_def = Constraint(m.SC, m.T, rule=shutdown_rule)

        def downtime_rule(m, sc, t):
            return sum(
                m.z_off[sc, k]
                for k in range(t, min(t + d.N_down, d.T))
            ) >= d.N_down * m.y_shutdown[sc, t]
        m.downtime = Constraint(m.SC, m.T, rule=downtime_rule)

        # --- Piecewise bounds ---
        def seg_min_rule(m, sc, t, s):
            return m.P_seg[sc,t,s] >= d.P_seg_min[s] * m.z_seg[sc,t,s]
        m.seg_min = Constraint(m.SC, m.T, m.SEG, rule=seg_min_rule)

        def seg_max_rule(m, sc, t, s):
            return m.P_seg[sc,t,s] <= d.P_seg_max[s] * m.z_seg[sc,t,s]
        m.seg_max = Constraint(m.SC, m.T, m.SEG, rule=seg_max_rule)

        def standby_power_rule(m, sc, t):
            return m.P_sb[sc,t] == d.P_sb * m.z_sb[sc,t]
        m.standby_power = Constraint(m.SC, m.T, rule=standby_power_rule)

        #--- Capacity with FCR headroom (COUPLING: r_FCR_block is Stage 1) ---
        def max_cap_rule(m, sc, t):
            return (
                sum(m.P_seg[sc,t,s] for s in m.SEG)
                + m.P_sb[sc,t]
                + m.r_FCR_block[i_of_t(t)]
                <= d.P_max * m.z_on[sc,t]
            )
        m.max_cap = Constraint(m.SC, m.T, rule=max_cap_rule)

        def min_load_rule(m, sc, t):
            return (
                sum(m.P_seg[sc,t,s] for s in m.SEG)
                - m.r_FCR_block[i_of_t(t)]
                >= d.P_min * m.z_on[sc,t]
            )
        m.min_load = Constraint(m.SC, m.T, rule=min_load_rule)

        # ALPHA = 0.8   # fraction of scenarios that must satisfy the headroom constraint
        #       # test: 1.0 (current), 0.9, 0.8, 0.7, 0.5

        # def max_cap_rule(m, sc, t):
        #     return (
        #         sum(m.P_seg[sc,t,s] for s in m.SEG)
        #         + m.P_sb[sc,t]
        #         + m.r_FCR_block[i_of_t(t)] * m.z_on[sc,t]  # headroom only required when on
        #         <= d.P_max * m.z_on[sc,t]
        #     )
        # m.max_cap = Constraint(m.SC, m.T, rule=max_cap_rule)

        # def min_load_rule(m, sc, t):
        #     return (
        #         sum(m.P_seg[sc,t,s] for s in m.SEG)
        #         - m.r_FCR_block[i_of_t(t)] * m.z_on[sc,t]  # only subtract when on
        #         >= d.P_min * m.z_on[sc,t]
        #     )
        # m.min_load = Constraint(m.SC, m.T, rule=min_load_rule)

        # # Add chance constraint: electrolyzer must be on in at least ALPHA fraction
        # # of scenarios for each timestep in a committed FCR block
        # def fcr_chance_rule(m, i, t_offset):
        #     t = i * 16 + t_offset
        #     if t >= d.T:
        #         return Constraint.Skip
        #     return (
        #         sum(m.z_on[sc, t] for sc in m.SC)
        #         >= ALPHA * d.N_sc * m.z_FCR[i]
        #     )
        # m.T_offset = RangeSet(0, 15)
        # m.fcr_chance = Constraint(m.I, m.T_offset, rule=fcr_chance_rule)

        # --- Ramping ---
        def ramp_up_rule(m, sc, t):
            dR = 0.95*d.P_max
            if t == 0:
                return sum(m.P_seg[sc,0,s] for s in m.SEG) - d.P_prev <= dR
            return (
                sum(m.P_seg[sc,t,s] for s in m.SEG)
                - sum(m.P_seg[sc,t-1,s] for s in m.SEG)
                <= dR
            )
        m.ramp_up = Constraint(m.SC, m.T, rule=ramp_up_rule)

        def ramp_down_rule(m, sc, t):
            dR = 0.95*d.P_max
            if t == 0:
                return d.P_prev - sum(m.P_seg[sc,0,s] for s in m.SEG) <= dR
            return (
                sum(m.P_seg[sc,t-1,s] for s in m.SEG)
                - sum(m.P_seg[sc,t,s] for s in m.SEG)
                <= dR
            )
        m.ramp_down = Constraint(m.SC, m.T, rule=ramp_down_rule)

        # --- Wind balance (uses scenario wind w_sc) ---
        # Note: P_grid >= 0, P_buy >= 0  (RT sign convention)
        # P_DA (Stage 1) appears here via imbalance only — not in balance
        def wind_balance_rule(m, sc, t):
            return (
                m.wind_sc[sc,t] #+ m.P_buy[sc,t]
                == sum(m.P_seg[sc,t,s] for s in m.SEG)
                + m.P_sb[sc,t]
                + m.P_curt[sc,t]
                + m.P_grid[sc,t]
            )
        m.wind_balance = Constraint(m.SC, m.T, rule=wind_balance_rule)

        # def buy_limit_rule(m, sc, t):
        #     return m.P_buy[sc,t] <= d.grid_cap
        # m.buy_limit = Constraint(m.SC, m.T, rule=buy_limit_rule)

        # def buy_use_rule(m, sc, t):
        #     return m.P_buy[sc,t] <= sum(m.P_seg[sc,t,s] for s in m.SEG)
        # m.buy_use = Constraint(m.SC, m.T, rule=buy_use_rule)

        # --- Imbalance deviation (COUPLING: P_DA is Stage 1) ---
        def dev_pos_rule(m, sc, t):
            return m.dev_pos[sc,t] >= m.P_grid[sc,t] - m.P_DA[t]
        m.dev_pos_con = Constraint(m.SC, m.T, rule=dev_pos_rule)

        def dev_neg_rule(m, sc, t):
            return m.dev_neg[sc,t] >= m.P_DA[t] - m.P_grid[sc,t]
        m.dev_neg_con = Constraint(m.SC, m.T, rule=dev_neg_rule)

        # --- Hydrogen ---
        def h2_rule(m, sc, t):
            return m.h[sc,t] == sum(
                d.A_s[s] * m.P_seg[sc,t,s] + d.B_s[s] * m.z_seg[sc,t,s]
                for s in m.SEG
            )
        m.h2_prod = Constraint(m.SC, m.T, rule=h2_rule)

        T_last = d.T - 1
        dt = 0.25

        def storage_rule(m, sc, t):
     #Could be change with an average for the initial storage instead of enforcing for every scenario 
            if t == 0:
                return m.storage[sc, t] == d.storage_initial + (m.h[sc,t] - m.h_sell[sc,t]) * dt #return m.storage[sc,t] == m.storage[sc, T_last] + (m.h[sc,t] - m.h_sell[sc,t]) * dt
            return m.storage[sc,t] == m.storage[sc,t-1] + (m.h[sc,t] - m.h_sell[sc,t]) * dt
        m.storage_dyn = Constraint(m.SC, m.T, rule=storage_rule)

        
   
        def storage_bound_rule(m, sc, t):
            return m.storage[sc,t] <= d.storage_max
        m.storage_bound = Constraint(m.SC, m.T, rule=storage_bound_rule)

        def h_sell_limit_rule(m, sc, t):
            if t == 0:
                return m.h_sell[sc,t]*dt <= m.h[sc,t]*dt + m.storage[sc, T_last]
            return m.h_sell[sc,t]*dt <= m.h[sc,t]*dt + m.storage[sc,t-1]
        m.h_sell_limit = Constraint(m.SC, m.T, rule=h_sell_limit_rule)

        def max_sell_rule(m, sc, t):
            return m.h_sell[sc,t] <= d.h_sell_max
        m.max_sell = Constraint(m.SC, m.T, rule=max_sell_rule)

        def min_daily_h2_rule(m, sc):
            return sum(m.h_sell[sc,t]*dt for t in m.T) >= d.h_min_daily
        m.min_daily_h2 = Constraint(m.SC, rule=min_daily_h2_rule)

        # --- FCR on-block (only participate when electrolyzer is on) ---
        # def fcr_on_block_rule(m, i, sc):
        #     t = i * 16
        #     if t >= d.T:
        #         return Constraint.Skip
        #     return m.z_FCR[i] <= m.z_on[sc, t]
        # m.fcr_on_block = Constraint(m.I, m.SC, rule=fcr_on_block_rule)


        # 
        # def fcr_on_block_rule(m, i, sc, t_offset):
        #     t = i * 16 + t_offset
        #     if t >= d.T:
        #         return Constraint.Skip
        #     return m.z_FCR[i] <= m.z_on[sc, t]

        m.T_offset = RangeSet(0, 15)
        # m.fcr_on_block = Constraint(m.I, m.SC, m.T_offset, rule=fcr_on_block_rule)


        # FCR requires a committed-on Stage 1 decision
        def fcr_needs_commit_rule(m, i):
            return m.z_FCR[i] <= m.z_on_commit[i]
        #m.fcr_needs_commit = Constraint(m.I, rule=fcr_needs_commit_rule)

        # All scenarios must honour the Stage 1 on-commitment (note: >= not <=)
        def honour_on_commit_rule(m, i, sc, t_offset):
            t = i * 16 + t_offset
            if t >= d.T:
                return Constraint.Skip
            return m.z_on[sc, t] >= m.z_on_commit[i]
        #m.honour_on_commit = Constraint(m.I, m.SC, m.T_offset, rule=honour_on_commit_rule)


        def P_DA_upper_rule(m, t):
            return m.P_DA[t] <= m.wind_fc[t]

        def P_DA_lower_rule(m, t):
            return m.P_DA[t] >= m.wind_fc[t] - d.P_max

        def P_DA_exp_link(m, t):
            expected_grid = sum(
             m.prob[sc] * m.P_grid[sc, t] for sc in m.SC
            )
            return m.P_DA[t] == expected_grid
        m.P_DA_link = Constraint(m.T, rule=P_DA_exp_link)

        #m.P_DA_upper = Constraint(m.T, rule=P_DA_upper_rule)
        #m.P_DA_lower = Constraint(m.T, rule=P_DA_lower_rule)



        # m.buy_or_curt = Var(m.SC, m.T, domain=Binary)
        M = d.grid_cap + 50000

        # def no_buy_if_curt_rule(m, sc, t):
        #     return m.P_buy[sc,t] <= M * m.buy_or_curt[sc,t]
        # m.no_buy_if_curt = Constraint(m.SC, m.T, rule=no_buy_if_curt_rule)

        # def no_curt_if_buy_rule(m, sc, t):
        #     return m.P_curt[sc,t] <= M * (1 - m.buy_or_curt[sc,t])
        # m.no_curt_if_buy = Constraint(m.SC, m.T, rule=no_curt_if_buy_rule)

        #Curtailment will be 0 in all scenarios if P_DA is negative:

        m.import_flag = Var(m.SC, m.T, domain=Binary)

        def no_import_if_curt(m, sc, t):
            return m.P_DA[t] >= -M * m.import_flag[sc, t]
        #m.no_import_if_curt = Constraint(m.SC, m.T, rule=no_import_if_curt)

        def no_curt_if_import(m, sc, t):
            return m.P_curt[sc, t] <= M * (1 - m.import_flag[sc, t])
        #m.no_curt_if_import = Constraint(m.SC, m.T, rule=no_curt_if_import)

        # m.binary = Var(m.SC, m.T, domain = Binary)
        # #No buy if export
        # def no_rt_imp_if_exp(m, sc, t):
        #     return m.P_grid[sc,t] <= M * m.binary [sc, t]
        # m.no_imp_if_exp = Constraint(m.SC, m.T, rule = no_rt_imp_if_exp)

        # def no_rt_exp_if_imp(m, sc, t):
        #     return m.P_buy[sc,t] <= M * (1- m.binary [sc, t])
        # m.no_exp_if_imp = Constraint(m.SC, m.T, rule = no_rt_exp_if_imp)
        
    # ------------------------------------------------------------------
    # OBJECTIVE
    # ------------------------------------------------------------------

    def _build_objective(self):
        m = self.model
        d = self.data
        dt      = 0.25
        dt_FCR  = 4
        penalty = 73.2

        def obj_rule(m):
            # Stage 1: certain FCR revenue
            fcr_revenue = sum(
                m.r_FCR_block[i] * m.price_FCR[i] * dt_FCR
                for i in m.I
            )

            # Stage 2: expected recourse value across scenarios
            expected_Q = sum(
                m.prob[sc] * (
                    # (ii) DA export revenue at realized price
                    sum(m.P_DA[t] * m.price_sc[sc,t] * dt / 1000
                        for t in m.T)


                    # (iii) Imbalance settlement
                     #- sum((m.dev_pos[sc,t] + m.dev_neg[sc,t])
                     #      *penalty * dt / 1000
                     #      for t in m.T)
                    # Imbalance settlement — signed, matches RT formulation exactly
                    + sum(
                        (m.P_grid[sc,t] - m.P_DA[t]) * m.imb_sc[sc,t] * dt / 1000 
                        for t in m.T
                    )

                    # (iv) Hydrogen revenue
                    + sum(m.h_sell[sc,t] * m.price_H2[t] * dt
                          for t in m.T)
                    # (v) Grid purchase cost
                    #- sum(m.P_buy[sc,t] * m.price_sc[sc,t] * dt / 1000
                    #      for t in m.T)
                )
                for sc in m.SC
            )

            return fcr_revenue + expected_Q

        m.obj = Objective(rule=obj_rule, sense=maximize)

    # ------------------------------------------------------------------
    # SOLVE
    # ------------------------------------------------------------------

    def run(self):
        from pyomo.opt import SolverFactory
        solver = SolverFactory('gurobi')
        result = solver.solve(self.model, tee=False)
        if result.solver.termination_condition != TerminationCondition.optimal:
            print("WARNING: Solver did not find optimal solution:",
                  result.solver.termination_condition)

        m = self.model
        d = self.data

        # Extract Stage 1 results
        self.results.P_DA         = np.array([value(m.P_DA[t]) for t in m.T])
        self.results.r_FCR_block  = np.array([value(m.r_FCR_block[i]) for i in m.I])
        self.results.z_FCR        = np.array([value(m.z_FCR[i]) for i in m.I])
        self.results.objective_value = value(m.obj)

        # Extract Stage 2 results averaged across scenarios (for main.py compatibility)
        self.results.h_sell = np.array([
            sum(value(m.h_sell[sc,t]) for sc in m.SC) / d.N_sc
            for t in m.T
        ])
        self.results.P_grid = np.array([
            sum(value(m.P_grid[sc,t]) for sc in m.SC) / d.N_sc
            for t in m.T
        ])
        self.results.P_curt = np.array([
            sum(value(m.P_curt[sc,t]) for sc in m.SC) / d.N_sc
            for t in m.T
        ])
        self.results.storage = np.array([
            sum(value(m.storage[sc,t]) for sc in m.SC) / d.N_sc
            for t in m.T
        ])
        self.results.z_on = np.array([
            sum(value(m.z_on[sc,t]) for sc in m.SC) / d.N_sc
            for t in m.T
        ])
        #self.results.P_buy = np.array([
        #    sum(value(m.P_buy[sc,t]) for sc in m.SC) / d.N_sc
        #    for t in m.T
        #])
        
        # Per-scenario results if needed for deeper analysis
        self.results.h_sell_sc = {
            sc: [value(m.h_sell[sc,t]) for t in m.T]
            for sc in m.SC
        }
        self.results.storage_sc = {
            sc: [value(m.storage[sc,t]) for t in m.T]
            for sc in m.SC
        }

        
    # def _save_results(self):
    #     m = self.model
    #     d = self.data
    #     if d.is_DA:
    #         self.results.P_DA = np.array([value(m.P_DA[t]) for t in m.T])
    #     else:
    #         self.results.P_DA = np.array([value(m.DA_export[t]) for t in m.T])        
    #     if not d.is_DA:
    #         self.results.P_grid = np.array([value(m.P_grid[t]) for t in m.T])
    #     else:
    #         self.results.P_grid = None
    #     self.results.h = [value(m.h[t]) for t in m.T]
    #     self.results.storage = [value(m.storage[t]) for t in m.T]
    #     self.results.z_on = [value(m.z_on[t]) for t in m.T]  # On/off status
    #     self.results.z_sb = [value(m.z_sb[t]) for t in m.T]  # Standby status
    #     self.results.z_off = [value(m.z_off[t]) for t in m.T]  # Off status
    #     self.results.P_buy = [value(m.P_buy[t]) for t in m.T]  # Electricity bought
    #     self.results.P_curt = [value(m.P_curt[t]) for t in m.T]  # Curtailed power
    #     self.results.r_FCR_block = [value(m.r_FCR_block[i]) for i in m.I]  # FCR block power
    #     self.results.h_sell = [value(m.h_sell[t]) for t in m.T]  # Hydrogen sold
    #     self.results.P_elec = np.array([
    #      sum(m.P_seg[t, s].value for s in m.S) + m.P_sb[t].value
    #      for t in m.T
    #  ])
        # self.results.P_elec = {
        #     t: sum(self.model.P_seg[t, s].value for s in self.model.S)
        #     + self.model.P_sb[t].value
        #     for t in self.model.T
        # }
        # self.results.objective_value = value(m.obj)
        # self.results.r_FCR_block = np.array([value(m.r_FCR_block[i]) for i in m.I])
        # print("Saved results: P_DA length =", len(self.results.P_DA) if self.results.P_DA is not None else None)
        # print("\n=== IIS CONSTRAINTS ===")
        # for c in m.component_objects(Constraint, active=True):
        #     for idx in c:
        #         if c[idx].active and c[idx].has_ub() and c[idx].has_lb():
        #             if c[idx].lbody() > c[idx].ub() + 1e-6 or c[idx].ub() < c[idx].lbody() - 1e-6:
        #                 print("VIOLATED:", c.name, idx, value(c[idx].body))
