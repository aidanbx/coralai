class EinParams:
    def __init__(self,
                 growth_efficiency=1.0,
                 capital_per_work_growth=10,
                 flow_cost=0.01,
                 capital_per_work_port=0.01,
                 capital_per_work_mine=0.01):
        self.growth_efficiency = growth_efficiency
        self.capital_per_work_growth = capital_per_work_growth
        self.flow_cost = flow_cost
        self.capital_per_work_port = capital_per_work_port
        self.capital_per_work_mine = capital_per_work_mine
