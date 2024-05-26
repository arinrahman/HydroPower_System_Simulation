from Turbine import Turbine
from Solar import Solar


class Energy_Balancer:

    def __init__(self, power = 0):
        self.power = power
        self.turbine = Turbine()
        self.solar = Solar()
    
    def update_energy(self,temp,flow):
        self.solar.update_energy(temp)
        self.turbine.update(flow)
        self.power = self.turbine.get_output() + self.solar.get_output()
        return self.power

    def get_energy_output(self):
        return self.power
    
    def get_solar_energy_output(self):
        return self.solar.get_output()
    
    def get_hydro_energy_output(self):
        return self.turbine.get_output()
    