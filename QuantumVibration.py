import numpy as np


def temperature_factor(vibrationFrequencyCm: float, temperature: float = 298.15) -> float:
    h = 6.62607015e-34
    k = 1.380649e-23
    c = 299792458
    R = 8.31446261815324
    theta = h * c * vibrationFrequencyCm / k

    heatCapacity = R * (theta / temperature) ** 2 * (np.exp((-theta / (2 * temperature))) / (1 - np.exp(-theta / temperature))) ** 2
    return float(heatCapacity)


Cv = temperature_factor(1654.0, 2)
print(Cv)
waterFrequencies = [1654, 3280, 3490]
temperatures = [1, 5, 10, 20, 50, 100, 200, 300, 400]

i = []
for freq in waterFrequencies:
    j = []
    for temp in temperatures:
        j.append(temperature_factor(freq, temp))
    i.append(j)

print(i)
