from Energy_Balancer import Energy_Balancer

class Reservoir:

    def __init__(self, length, width, height, current_volume, temperature):
        self.length = length
        self.width = width
        self.height = height
        self.max_volume = length * width * height
        #self.area = length * width
        self.current_volume = current_volume
        self.energy_generation = Energy_Balancer()
        self.temperature = temperature

    def release(self, release_amount):
        self.current_volume = max(0,self.current_volume - release_amount)
        self.energy_generation.update_energy(self.temperature,release_amount)
    
    def inflow(self, inflow_amount):
        self.current_volume = min(self.max_volume,self.current_volume + inflow_amount)
        self.energy_generation.update_energy(self.temperature,inflow_amount)

    def energy_output(self):
        return self.energy_generation.get_energy_output()

    


    
    