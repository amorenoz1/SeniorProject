import numpy as np

def get_center_of_mass(adcs):
    weighted_sum = 0
    total_mass = 0
    
    for l, adc in enumerate(adcs):
        weighted_sum += adc * l
        total_mass += adc


adcs = [ -120, 160, 180, 200, 180, 160, 120]

print(adcs[get_center_of_mass(adcs)])
