import json
import matplotlib.pyplot as plt

# Load the data from the JSON file
with open('simulation_summary_20240812_190331.json', 'r') as file:
    data = json.load(file)

# Extract the daily population data for Deer
days = []
deer_population = []

for day_data in data['daily_animal_data']:
    days.append(day_data['day'])
    deer_population.append(day_data['population']['Deer'])

# Plot the population over days
plt.figure(figsize=(10, 6))
plt.plot(days, deer_population, marker='o')
plt.title('Deer Population Over Days')
plt.xlabel('Days')
plt.ylabel('Deer Population')
plt.grid(True)
plt.show()
