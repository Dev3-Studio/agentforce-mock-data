# Mining Sustainability Database Models

## Mine Model

The `Mine` model represents a mining operation with its basic characteristics and timeline.

| Column | Type | Description | Example Value |
|--------|------|-------------|--------------|
| mine_id | INTEGER | Unique identifier for each mine | 1 |
| mine_name | STRING | Name of the mine | "Copper Mountain Mine" |
| mine_type | STRING | Type of mine (Open Pit, Underground, etc.) | "Open Pit" |
| ore_type | STRING | Type of ore being mined | "Copper" |
| production_capacity | FLOAT | Annual production capacity in tonnes | 45000000.0 |
| mine_lifespan_months | INTEGER | Expected lifespan of the mine in months | 240 |
| initial_footprint | FLOAT | Initial land area in hectares | 350.0 |
| maximum_footprint | FLOAT | Maximum land area in hectares | 500.0 |
| pre_existing_biodiversity | FLOAT | Biodiversity index before mining (0-1) | 0.85 |
| construction_start | DATETIME | Date when construction begins | "2020-01-01" |
| production_start | DATETIME | Date when production begins | "2022-06-01" |
| peak_production | DATETIME | Date of peak production | "2030-01-01" |
| closure | DATETIME | Date when closure begins | "2040-01-01" |
| post_closure_monitoring | INTEGER | Years of post-closure monitoring | 30 |

## SustainabilityParameter Model

The `SustainabilityParameter` model stores environmental and social parameters associated with mining equipment.

| Column | Type | Description | Example Value |
|--------|------|-------------|--------------|
| param_id | INTEGER | Unique identifier for each parameter | 1 |
| equipment_id | INTEGER | Reference to the equipment | 1 |
| parameter_category | STRING | Category (Biodiversity, Carbon, Water, etc.) | "Carbon" |
| parameter_name | STRING | Name of the parameter | "emissions_factor" |
| parameter_value | TEXT | Value of the parameter (stored as text) | "85.5" |
| created_at | DATETIME | When the parameter was created | "2023-06-15T10:30:00" |

## MiningEvent Model

The `MiningEvent` model defines types of events that can occur during mining operations.

| Column | Type | Description | Example Value |
|--------|------|-------------|--------------|
| event_id | INTEGER | Unique identifier for each event type | 1 |
| event_name | STRING | Name of the mining event | "Mine Expansion Phase 1" |
| event_category | STRING | Category of event (Expansion, Technology, etc.) | "Expansion" |
| event_description | TEXT | Detailed description of the event | "Expansion of the mine footprint to access additional ore reserves" |
| typical_duration | INTEGER | Typical duration of the event in months | 24 |
| duration_variance | FLOAT | Variance in duration for simulations | 6.0 |
| prerequisite_events | TEXT | JSON array of event_ids that must occur before this | "[]" |
| mutually_exclusive_events | TEXT | JSON array of event_ids that cannot occur with this | "[7]" |
| probability_function | TEXT | Function to calculate probability of event occurring | "lambda year, mine_age: 0.2 if mine_age > 5 and mine_age < 15 else 0.05" |
| created_at | DATETIME | When the event was created | "2023-06-15T10:30:00" |

## EventImpact Model

The `EventImpact` model defines how mining events affect sustainability parameters.

| Column | Type | Description | Example Value |
|--------|------|-------------|--------------|
| impact_id | INTEGER | Unique identifier for each impact | 1 |
| event_id | INTEGER | Reference to the mining event | 1 |
| parameter_category | STRING | Category of sustainability parameter affected | "Biodiversity" |
| parameter_name | STRING | Specific parameter affected | "species_richness" |
| impact_type | STRING | Type of impact (Immediate, Gradual, Delayed) | "Immediate" |
| impact_mean | FLOAT | Mean impact value (positive or negative) | -0.25 |
| impact_std_dev | FLOAT | Standard deviation of impact | 0.05 |
| impact_distribution | STRING | Statistical distribution to use | "Normal" |
| recovery_rate_mean | FLOAT | Mean rate of recovery after impact | 0.05 |
| recovery_rate_std_dev | FLOAT | Standard deviation of recovery rate | 0.01 |
| max_recovery_percentage | FLOAT | Maximum percentage that can be recovered | 0.9 |
| created_at | DATETIME | When the impact was created | "2023-06-15T10:30:00" |

## SimulationRun Model

The `SimulationRun` model represents a Monte Carlo simulation run for sustainability assessment.

| Column | Type | Description | Example Value |
|--------|------|-------------|--------------|
| run_id | INTEGER | Unique identifier for each simulation run | 1 |
| mine_id | INTEGER | Reference to the mine | 1 |
| simulation_name | STRING | Name of the simulation | "Baseline Sustainability Simulation" |
| num_years | INTEGER | Number of years simulated | 20 |
| num_simulations | INTEGER | Number of Monte Carlo iterations | 100 |
| random_seed | INTEGER | Random seed for reproducibility | 42 |
| created_at | DATETIME | When the simulation was created | "2023-06-15T10:30:00" |
| description | TEXT | Description of the simulation purpose | "Baseline simulation to assess sustainability metrics over the mine's lifespan" |

## SimulationEvent Model

The `SimulationEvent` model records events that occurred during a simulation run.

| Column | Type | Description | Example Value |
|--------|------|-------------|--------------|
| sim_event_id | INTEGER | Unique identifier for each simulation event | 1 |
| run_id | INTEGER | Reference to the simulation run | 1 |
| event_id | INTEGER | Reference to the event type | 1 |
| simulation_number | INTEGER | Monte Carlo iteration number | 5 |
| start_year | INTEGER | Year when the event starts | 7 |
| end_year | INTEGER | Year when the event ends | 9 |
| actual_impact_values | TEXT | JSON object with actual impact values | '{"Biodiversity_species_richness": {"value": -0.27, "recovery_rate": 0.048}}' |
| created_at | DATETIME | When the simulation event was created | "2023-06-15T10:30:00" |

## SimulationResult Model

The `SimulationResult` model stores the results of sustainability simulations.

| Column | Type | Description | Example Value |
|--------|------|-------------|--------------|
| result_id | INTEGER | Unique identifier for each result | 1 |
| run_id | INTEGER | Reference to the simulation run | 1 |
| datetime | DATETIME | Date and time in simulation | "2027-06-15T00:00:00" |
| simulation_number | INTEGER | Monte Carlo iteration number | 5 |
| metric_name | STRING | Name of the sustainability metric | "biodiversity_score" |
| metric_value | FLOAT | Value of the sustainability metric | 72.5 |

## ScenarioDefinition Model

The `ScenarioDefinition` model defines predefined scenarios for sustainability planning.

| Column | Type | Description | Example Value |
|--------|------|-------------|--------------|
| scenario_id | INTEGER | Unique identifier for each scenario | 1 |
| scenario_name | STRING | Name of the scenario | "Eco-Friendly Operations" |
| scenario_description | TEXT | Description of the scenario | "A scenario with early renewable energy implementation and progressive rehabilitation" |
| created_at | DATETIME | When the scenario was created | "2023-06-15T10:30:00" |

## ScenarioEvent Model

The `ScenarioEvent` model defines events that make up a scenario.

| Column | Type | Description | Example Value |
|--------|------|-------------|--------------|
| scenario_event_id | INTEGER | Unique identifier for each scenario event | 1 |
| scenario_id | INTEGER | Reference to the scenario | 1 |
| event_id | INTEGER | Reference to the event type | 3 |
| start | DATETIME | Date when the event starts in this scenario | "2025-01-01T00:00:00" |
| probability_override | FLOAT | Override for the event probability (0-1) | 0.9 |
| event_parameters | TEXT | JSON object with custom parameters | '{"implementation_scale": "full", "technology": "solar_and_wind"}' |

## MachineryType Model

The `MachineryType` model defines types of machinery used in mining operations.

| Column | Type | Description | Example Value |
|--------|------|-------------|--------------|
| type_id | INTEGER | Unique identifier for each machinery type | 1 |
| type_name | STRING | Name of the machinery type | "Heavy Duty Haul Truck" |
| category | STRING | Category (Excavation, Hauling, Processing, etc.) | "Hauling" |
| description | TEXT | Description of the machinery type | "Large capacity haul truck for moving ore and waste rock" |
| fuel_type | STRING | Type of fuel used (Diesel, Electric, Hybrid, etc.) | "Diesel" |
| emissions_factor | FLOAT | CO2e emissions per hour of operation (kg) | 85.5 |
| energy_consumption | FLOAT | Energy units per hour of operation | 120.0 |
| water_usage | FLOAT | Water usage per hour of operation (m³) | 0.0 |
| maintenance_interval | INTEGER | Hours between maintenance | 500 |

## MachineryInstance Model

The `MachineryInstance` model represents specific machinery owned by a mine.

| Column | Type | Description | Example Value |
|--------|------|-------------|--------------|
| instance_id | INTEGER | Unique identifier for each machinery instance | 1 |
| mine_id | INTEGER | Reference to the mine | 1 |
| type_id | INTEGER | Reference to the machinery type | 1 |
| instance_name | STRING | Name of the machinery instance | "Haul Truck #HD-001" |
| operational_status | STRING | Status (Active, Maintenance, Retired, etc.) | "Active" |
| efficiency_factor | FLOAT | Efficiency relative to typical (1.0 = typical) | 0.95 |
| total_operating_hours | FLOAT | Total hours of operation | 2500.0 |

## MachinerySimulationResult Model

The `MachinerySimulationResult` model stores results of machinery simulations.

| Column | Type | Description | Example Value |
|--------|------|-------------|--------------|
| result_id | INTEGER | Unique identifier for each result | 1 |
| run_id | INTEGER | Reference to the simulation run | 1 |
| instance_id | INTEGER | Reference to the machinery instance | 1 |
| simulation_number | INTEGER | Monte Carlo iteration number | 5 |
| simulation_date | DATETIME | Date in simulation | "2027-06-15T00:00:00" |
| hours_operated | FLOAT | Hours operated that day | 18.5 |
| fuel_consumed | FLOAT | Fuel consumed (liters or kWh) | 2220.0 |
| emissions_produced | FLOAT | Emissions produced (kg CO2e) | 1581.75 |
| energy_consumed | FLOAT | Energy consumed (MJ) | 85470.0 |
| water_used | FLOAT | Water used (m³) | 0.0 |
| operational_efficiency | FLOAT | Operational efficiency (0-1) | 0.92 |


# Mining Sustainability Database Models (Continued)

## Sustainability Parameters Examples

The sustainability parameters are organized by category, with each parameter having a name, value, and description. Here are examples for each category:

### Biodiversity Parameters

| Parameter Name | Value | Description |
|----------------|-------|-------------|
| habitat_types | ["Forest", "Grassland", "Wetland"] | Types of habitats affected by mining operations |
| species_richness | 120 | Number of species in the mining area |
| endangered_species | 3 | Number of endangered species in the area |
| ecosystem_sensitivity | 6 | Sensitivity rating on a scale of 1-10 |
| biodiversity_offset_ratio | 1.5 | Ratio of restored area to disturbed area |
| restoration_effectiveness | 0.75 | Effectiveness of restoration efforts (0-1) |

### Carbon Parameters

| Parameter Name | Value | Description |
|----------------|-------|-------------|
| energy_mix | {"Grid": 0.6, "Diesel": 0.3, "Renewable": 0.1} | Proportion of energy sources used |
| equipment_efficiency | 0.85 | Efficiency factor of equipment (0-1) |
| process_emissions_factor | 0.4 | Emissions per tonne of ore processed (tCO2e) |
| carbon_intensity | 0.8 | Carbon intensity of operations (tCO2e/tonne) |
| decarbonization_rate | 0.03 | Annual rate of emissions reduction |
| carbon_offsets | 5000 | Annual carbon offsets purchased (tCO2e) |

### Water Parameters

| Parameter Name | Value | Description |
|----------------|-------|-------------|
| water_consumption | 1.2 | Water use per tonne of ore processed (m³) |
| water_recycling | 0.65 | Proportion of water recycled in operations |
| discharge_quality | 0.9 | Quality of discharged water (0-1) |
| watershed_impact | 0.4 | Impact on local watershed (0-1) |
| groundwater_drawdown | 2.5 | Annual groundwater level reduction (m) |

### Waste Parameters

| Parameter Name | Value | Description |
|----------------|-------|-------------|
| waste_rock_ratio | 3.2 | Ratio of waste rock to ore |
| tailings_generation | 0.85 | Tonnes of tailings per tonne of ore |
| hazardous_waste | 0.02 | Proportion of waste classified as hazardous |
| waste_treatment | 0.8 | Efficiency of waste treatment processes (0-1) |
| amd_potential | 0.3 | Potential for acid mine drainage (0-1) |

### Community Parameters

| Parameter Name | Value | Description |
|----------------|-------|-------------|
| local_employment | 0.6 | Proportion of employees from local communities |
| community_investment | 500000 | Annual investment in community projects ($) |
| resettlement | 15 | Number of households resettled due to mining |
| indigenous_overlap | 0.25 | Overlap with indigenous territories (0-1) |
| social_license | 6.5 | Social license to operate rating (1-10) |

## Mining Events Examples

Mining events represent significant activities or changes that can occur during a mine's operation. Here are examples of different types of events:

### Mine Expansion Event

| Field | Value | Description |
|-------|-------|-------------|
| event_id | 1 | Unique identifier |
| event_name | "Mine Expansion" | Name of the event |
| event_category | "Expansion" | Category of the event |
| event_description | "Expansion of the mine footprint to access additional ore reserves" | Detailed description |
| typical_duration | 3 | Typical duration in years |
| duration_variance | 1.0 | Variance in duration for simulations |
| prerequisite_events | "[]" | No prerequisite events required |
| mutually_exclusive_events | "[7]" | Cannot occur with event ID 7 (mine closure) |
| probability_function | "lambda year, mine_age: 0.2 if mine_age > 5 and mine_age < 15 else 0.05" | Function determining probability |

### Technology Upgrade Event

| Field | Value | Description |
|-------|-------|-------------|
| event_id | 2 | Unique identifier |
| event_name | "Processing Technology Upgrade" | Name of the event |
| event_category | "Technology" | Category of the event |
| event_description | "Implementation of more efficient ore processing technology" | Detailed description |
| typical_duration | 2 | Typical duration in years |
| duration_variance | 0.5 | Variance in duration for simulations |
| prerequisite_events | "[]" | No prerequisite events required |
| mutually_exclusive_events | "[7]" | Cannot occur with event ID 7 (mine closure) |
| probability_function | "lambda year, mine_age: 0.15 if mine_age > 3 else 0.05" | Function determining probability |

### Renewable Energy Implementation Event

| Field | Value | Description |
|-------|-------|-------------|
| event_id | 3 | Unique identifier |
| event_name | "Renewable Energy Implementation" | Name of the event |
| event_category | "Technology" | Category of the event |
| event_description | "Installation of renewable energy sources (solar, wind) to power mine operations" | Detailed description |
| typical_duration | 2 | Typical duration in years |
| duration_variance | 0.5 | Variance in duration for simulations |
| prerequisite_events | "[]" | No prerequisite events required |
| mutually_exclusive_events | "[7]" | Cannot occur with event ID 7 (mine closure) |
| probability_function | "lambda year, mine_age: 0.1 + (year - 2020) * 0.01" | Function determining probability |

## Event Impacts Examples

Event impacts define how mining events affect sustainability parameters. Here are examples of different types of impacts:

### Biodiversity Impact from Mine Expansion

| Field | Value | Description |
|-------|-------|-------------|
| impact_id | 1 | Unique identifier |
| event_id | 1 | Reference to Mine Expansion event |
| parameter_category | "Biodiversity" | Category of parameter affected |
| parameter_name | "species_richness" | Specific parameter affected |
| impact_type | "Immediate" | Type of impact |
| impact_mean | -0.25 | Mean impact value (25% reduction) |
| impact_std_dev | 0.05 | Standard deviation of impact |
| impact_distribution | "Normal" | Statistical distribution used |
| recovery_rate_mean | 0.05 | Mean recovery rate (5% per year) |
| recovery_rate_std_dev | 0.01 | Standard deviation of recovery rate |
| max_recovery_percentage | 0.9 | Maximum recovery possible (90%) |

### Water Impact from Mine Expansion

| Field | Value | Description |
|-------|-------|-------------|
| impact_id | 2 | Unique identifier |
| event_id | 1 | Reference to Mine Expansion event |
| parameter_category | "Water" | Category of parameter affected |
| parameter_name | "water_consumption" | Specific parameter affected |
| impact_type | "Immediate" | Type of impact |
| impact_mean | 0.3 | Mean impact value (30% increase) |
| impact_std_dev | 0.08 | Standard deviation of impact |
| impact_distribution | "Normal" | Statistical distribution used |
| recovery_rate_mean | 0.1 | Mean recovery rate (10% per year) |
| recovery_rate_std_dev | 0.02 | Standard deviation of recovery rate |
| max_recovery_percentage | 1.0 | Maximum recovery possible (100%) |

### Carbon Impact from Technology Upgrade

| Field | Value | Description |
|-------|-------|-------------|
| impact_id | 7 | Unique identifier |
| event_id | 2 | Reference to Technology Upgrade event |
| parameter_category | "Carbon" | Category of parameter affected |
| parameter_name | "carbon_intensity" | Specific parameter affected |
| impact_type | "Gradual" | Type of impact |
| impact_mean | -0.15 | Mean impact value (15% reduction) |
| impact_std_dev | 0.04 | Standard deviation of impact |
| impact_distribution | "Normal" | Statistical distribution used |
| recovery_rate_mean | 0.0 | Mean recovery rate (not applicable) |
| recovery_rate_std_dev | 0.0 | Standard deviation of recovery rate |
| max_recovery_percentage | 0.0 | Maximum recovery possible (not applicable) |

## Machinery Types Examples

Machinery types define the characteristics of different equipment used in mining operations:

### Haul Truck

| Field | Value | Description |
|-------|-------|-------------|
| type_id | 1 | Unique identifier |
| type_name | "Heavy Duty Haul Truck" | Name of machinery type |
| category | "Hauling" | Category of machinery |
| description | "Large capacity haul truck for moving ore and waste rock" | Detailed description |
| fuel_type | "Diesel" | Type of fuel used |
| emissions_factor | 85.5 | CO2e emissions per hour (kg) |
| energy_consumption | 120.0 | Diesel consumption per hour (liters) |
| water_usage | 0.0 | Water usage per hour (m³) |
| maintenance_interval | 500 | Hours between maintenance |

### Excavator

| Field | Value | Description |
|-------|-------|-------------|
| type_id | 2 | Unique identifier |
| type_name | "Hydraulic Excavator" | Name of machinery type |
| category | "Excavation" | Category of machinery |
| description | "Large hydraulic excavator for digging and loading material" | Detailed description |
| fuel_type | "Diesel" | Type of fuel used |
| emissions_factor | 95.0 | CO2e emissions per hour (kg) |
| energy_consumption | 140.0 | Diesel consumption per hour (liters) |
| water_usage | 0.0 | Water usage per hour (m³) |
| maintenance_interval | 400 | Hours between maintenance |

### Processing Plant

| Field | Value | Description |
|-------|-------|-------------|
| type_id | 3 | Unique identifier |
| type_name | "Ore Processing Plant" | Name of machinery type |
| category | "Processing" | Category of machinery |
| description | "Facility for crushing, grinding, and processing ore" | Detailed description |
| fuel_type | "Electric" | Type of fuel used |
| emissions_factor | 250.0 | CO2e emissions per hour (kg) |
| energy_consumption | 5000.0 | Electricity consumption per hour (kWh) |
| water_usage | 50.0 | Water usage per hour (m³) |
| maintenance_interval | 720 | Hours between maintenance |

# Additional Information

## Simulation Logic Flow

The typical flow for running a simulation:

1. Create a `Mine` record with basic parameters
2. Add `MachineryInstance` records for the mine's equipment
3. Define sustainability parameters for the mine/equipment
4. Create a `SimulationRun` record
5. For each simulation iteration:
   - Determine which events occur (based on probability functions)
   - Create `SimulationEvent` records
   - Calculate impacts on sustainability parameters
   - Generate daily machinery operation data
   - Calculate sustainability metrics
   - Store results in `SimulationResult` and `MachinerySimulationResult`

## Monte Carlo Simulation Parameters

Key parameters that affect simulation outcomes:

- **Random Seed**: Controls reproducibility of simulations
- **Number of Simulations**: More simulations provide better statistical confidence
- **Time Horizon**: Typically covers the full mine lifecycle (20-50 years)
- **Event Probabilities**: Defined by functions that consider mine age, year, etc.
- **Impact Distributions**: Statistical distributions for parameter impacts

## Sustainability Score Calculation

The overall sustainability score is typically calculated as a weighted average of category scores:

```
Overall Score = (Biodiversity Score × 0.25) + 
                (Carbon Score × 0.25) + 
                (Water Score × 0.2) + 
                (Waste Score × 0.15) + 
                (Community Score × 0.15)
```

Each category score is calculated from its parameters, with some parameters having more weight than others.

## Event Impact Mechanics

Three types of impacts with different behaviors:

1. **Immediate**: Full impact applied at the start of the event
2. **Gradual**: Impact increases linearly over the event duration
3. **Delayed**: Impact applied at the end of the event

Recovery from negative impacts follows an exponential curve with a rate parameter.

## Machinery Simulation Considerations

- **Operational Hours**: Typically follow daily/weekly patterns with maintenance downtime
- **Efficiency Degradation**: Equipment efficiency decreases with age/usage
- **Maintenance Effects**: Maintenance improves efficiency but causes downtime
- **Emissions Calculation**: Based on fuel consumption and emissions factors

## Scenario vs. Simulation

Important distinction:

- **Scenarios**: Predefined sets of events with specific timing and parameters (deterministic)
- **Simulations**: Monte Carlo runs with probabilistic event occurrence (stochastic)


