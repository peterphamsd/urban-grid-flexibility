# Urban Grid Flexibility Simulator

A neighborhood-scale energy system model for San Diego that simulates rooftop solar generation, battery storage dispatch, and grid interaction under variable climate conditions.

## What this model does

- Computes hourly solar generation using the pvlib irradiance model and temperature correction
- Tracks battery state of charge across dispatch decisions with realistic efficiency losses and physical constraints
- Optimizes hourly charge, discharge, and grid import decisions using a linear program to minimize electricity cost
- Quantifies system performance uncertainty using Monte Carlo simulation with temporally correlated weather scenarios
- Stress tests system resilience against marine layer events, heat waves, and grid outages

## Mathematical foundation

The core energy balance at every timestep:

    Solar(h) + Battery_out(h) + Grid_in(h) = Load(h) + Battery_in(h)

Solar generation follows the photovoltaic efficiency equation with temperature correction. Battery dynamics follow SOC difference equations with round-trip efficiency losses. Dispatch is optimized via linear programming across a 24-hour horizon.

## Data sources

- Climate: NASA POWER API (hourly, 2010-2023)
- Load profiles: NREL ResStock (San Diego climate zone)
- Electricity prices: CAISO time-of-use rates

## Stack

Python, pvlib, pyomo, pandas, numpy, scipy, plotly

## Project structure

    models/       solar, battery, load, and optimizer components
    simulation/   Monte Carlo engine, AR(1) weather persistence, stress tests
    tests/        unit tests validating physical behavior of each component
    data/         raw and processed climate and load inputs
    outputs/      figures and scenario result tables

## Status

- [x] Solar generation model
- [ ] Battery model
- [ ] Load model
- [ ] Dispatch optimizer
- [ ] Monte Carlo simulation
- [ ] Stress testing
- [ ] Output visualization