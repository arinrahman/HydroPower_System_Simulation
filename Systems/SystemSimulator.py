from Distribution import Distribution
from HydroRelease import HydroRelease
from SolarGeneration import SolarGeneration
from Notification import Notification
# Initialize our variables
energyDemand = 900
time = 12

panelEfficency = .9
panelAmount = 5

inflow = 1000
hydroStored = 1000
waterReleased = 0
hydroEnergy = 10

hydroWeight = .5
solarWeight = .5
# Stabilization Attempts
attempts = 0
satisfies = False
while attempts < 3 and not satisfies:
    # Generate Distribution
    distribution = Distribution(energyDemand, hydroWeight, solarWeight)
    hydroDemand, solarDemand = distribution.calculateRequired()

    # Determine System Generation Predictions
    hydroRelease = HydroRelease(inflow, time, hydroStored)
    waterReleased, hydroReleasePrediction = hydroRelease.releasePrediction()
    solarRelease = SolarGeneration(panelAmount, panelEfficency, time)
    solarGenerationPrediction = solarRelease.generatePrediction()
    print(hydroReleasePrediction, solarGenerationPrediction)

    # Evaluate
    evaluator = Notification(energyDemand, hydroReleasePrediction, solarGenerationPrediction)
    satisfies, hydroWeight, solarWeight = evaluator.evaulate(hydroWeight, solarWeight)
    attempts += 1

# Release System
if satisfies:
    hydroStored -= waterReleased
    print("System Met Required Demand")
else:
    print("System Cannot Meet Requirements")
