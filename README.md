# Reinforcement Learning in 2D Ecosystems Simulation

This project simulates species inside a realistic ecosystem using machine learning techniques to explore the dynamic interactions between organisms and their environments. The simulation focuses on predator-prey dynamics, competition for resources, and the evolution of animal intelligence over multiple generations.

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Modules Overview](#modules-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Limitations and Future Work](#limitations-and-future-work)
- [References](#references)

## Introduction

The project simulates complex ecological dynamics by combining the study of animal behavior and ecosystem interactions in a single model. The goal is to provide valuable insights into species survival strategies and ecosystem management, contributing to broader research in ecology, conservation, and environmental sustainability.

## Key Features

- **Realism**: Simulates real-world dynamics such as predator-prey interactions and resource depletion.
- **Modular Design**: Each component operates independently, allowing for easy updates without affecting the entire system.
- **Reinforcement Learning**: Uses neural networks and reward-based systems for simulating animal learning and adaptive behavior.
- **Visualization**: Graphical representation of population dynamics and health states using `matplotlib`.
- **Customizability**: Simulation parameters, animals, and resources can be easily modified through JSON files.

## Modules Overview

### Clock Module
Manages in-game time, independent of real-world time. Keeps track of simulation days and resets after each day.

### Entity Structures Module
Defines entities, primarily animals, and handles their creation, removal, and behavior.

### Reward Module
Handles animal behaviors, species identities, and interactions with resources, including predator-prey dynamics.

### Event Manager Module
Generates and manages daily events, like creating resources or introducing new entities into the ecosystem.

### Visual Display Module
Uses the `Arcade` library to display the simulation, offering a menu-driven interface and live updates.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ForgottenRat/Reinforcement-Learning-In-2D-Ecosystems
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the simulation:
    ```bash
    python visual_display.py
    ```

## Usage

- Adjust the simulation parameters in the provided JSON configuration files to create custom scenarios.
- The main menu allows starting or exiting the simulation, and you can observe the behavior and population dynamics in real-time.
- Data is logged into JSON files after each simulation cycle, which can be further visualized using `matplotlib` to analyze the results.

## Results

The simulation generates population dynamics, task distribution, and health state graphs for the species involved. These outputs can be used to study predator-prey relationships, resource availability, and adaptation strategies over time.

Example graphs:
- **Population Dynamics**: Tracks species populations over time.
- **Health State Dynamics**: Shows the health and energy levels of animals throughout the simulation.

## Limitations and Future Work

- **Lack of Altruism**: Currently, the simulation does not account for altruistic behaviors, which may affect ecosystem dynamics.
- **Performance**: The simulation's performance can be improved through parallelization to handle more complex ecosystems.
- **Future Enhancements**: Adding abiotic factors, seasons, and more complex interspecies relationships could significantly improve realism.

## References

- Carew, J.M. (2023). *What is Reinforcement Learning?*. TechTarget.
- Libretexts. (2022). *The Lotka-Volterra Predator-Prey Model*.
- Python Arcade. (n.d.). *Arcade Library Documentation*. GitHub.
- Bradbury, R.H., Green, D.G., & Snoad, N. (2010). *Are Ecosystems Complex Systems?* Cambridge Core.

