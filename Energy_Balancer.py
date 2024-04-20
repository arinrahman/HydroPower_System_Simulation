from Turbine import Turbine
from Solar import Solar


class Energy_Balancer:

    def __init__(self, power = 0):
        self.power = power
        self.turbine = Turbine()
        self.solar = Solar()
    
    def update_energy(self):
        self.power = self.solar.get_output() + self.solar.get_output()
        return self.power

    def get_energy_output(self):
        return self.power
    