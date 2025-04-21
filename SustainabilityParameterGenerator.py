import random
import datetime
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

# SQLAlchemy imports
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from database.models import Base, Mine, MachineryType, MachineryInstance, SimulationRun, MachinerySimulationResult


class DistributionStrategy(ABC):
    """Base abstract class for different distribution strategies"""

    @abstractmethod
    def generate_values(self, n: int, **kwargs) -> List[float]:
        """Generate n values according to the distribution"""
        pass


class NormalDistribution(DistributionStrategy):
    """Normal (Gaussian) distribution strategy"""

    def generate_values(self, n: int, mean: float = 0.0, std: float = 1.0, **kwargs) -> List[float]:
        """Generate n values from a normal distribution"""
        # Ensure non-negative values where applicable (e.g., consumption, emissions)
        values = np.random.normal(mean, std, n)
        values[values < 0] = 0  # Clamp negative values to 0
        return list(values)


class PoissonDistribution(DistributionStrategy):
    """Poisson distribution strategy"""

    def generate_values(self, n: int, lam: float = 1.0, **kwargs) -> List[float]:
        """Generate n values from a Poisson distribution"""
        return list(np.random.poisson(lam, n))


class MonteCarloDistribution(DistributionStrategy):
    """Monte Carlo simulation strategy using custom probability functions"""

    def generate_values(self, n: int, min_val: float = 0.0, max_val: float = 1.0,
                        prob_func=None, **kwargs) -> List[float]:
        """
        Generate n values using Monte Carlo simulation

        Args:
            n: Number of values to generate
            min_val: Minimum value in the range
            max_val: Maximum value in the range
            prob_func: Custom probability function that takes a value and returns probability
                      If None, uniform distribution is used
        """
        if prob_func is None:
            # Default to uniform distribution
            return list(np.random.uniform(min_val, max_val, n))

        results = []
        for _ in range(n):
            # Keep sampling until we accept a value
            while True:
                x = random.uniform(min_val, max_val)
                y = random.uniform(0, 1)
                # Accept if random y is below the probability at x
                if y <= prob_func(x):
                    results.append(x)
                    break

        return results


class UniformDistribution(DistributionStrategy):
    """Uniform distribution strategy"""

    def generate_values(self, n: int, min_val: float = 0.0, max_val: float = 1.0, **kwargs) -> List[float]:
        """Generate n values from a uniform distribution"""
        return list(np.random.uniform(min_val, max_val, n))


class ExponentialDistribution(DistributionStrategy):
    """Exponential distribution strategy"""

    def generate_values(self, n: int, scale: float = 1.0, **kwargs) -> List[float]:
        """Generate n values from an exponential distribution"""
        return list(np.random.exponential(scale, n))


class SustainabilityParameterGenerator:
    """
    Generator for sustainability-related simulation data using SQLAlchemy models.
    Creates mines, machinery, simulation runs, and time-series results.
    """

    # Define parameter categories and possible names
    # Note: These are used to generate values, but only those mappable to
    # MachinerySimulationResult fields (emissions, energy, water usage) will be stored.
    PARAMETER_CATEGORIES = {
        "emissions": ["CO2", "CH4", "NOx", "SOx", "PM10", "PM2.5"],
        "water": ["pH", "turbidity", "dissolved_oxygen", "temperature", "conductivity", "total_suspended_solids", "water_usage"], # Added water_usage
        "energy": ["consumption_kwh", "renewable_percentage", "efficiency_rating", "peak_demand"],
        "waste": ["total_waste_tons", "recycled_percentage", "hazardous_waste", "tailings_volume"],
        "biodiversity": ["species_count", "habitat_quality_index", "revegetation_area", "protected_area_ratio"]
    }

    # Mapping from generated parameter names to MachinerySimulationResult fields
    PARAMETER_TO_MODEL_FIELD_MAP = {
        "CO2": "emissions_produced",
        "CH4": "emissions_produced",
        "NOx": "emissions_produced",
        "SOx": "emissions_produced",
        "PM10": "emissions_produced",
        "PM2.5": "emissions_produced",
        "consumption_kwh": "energy_consumed",
        "peak_demand": "energy_consumed", # Example mapping
        "water_usage": "water_used",
        "efficiency_rating": "operational_efficiency", # Example mapping
    }

    def __init__(self):
        """Initialize the generator with available distribution strategies"""
        self.distributions = {
            "normal": NormalDistribution(),
            "poisson": PoissonDistribution(),
            "montecarlo": MonteCarloDistribution(),
            "uniform": UniformDistribution(),
            "exponential": ExponentialDistribution()
        }

    def _get_parameter_config(self, category: str, parameter: str) -> Dict[str, Any]:
        """
        Get suitable distribution configuration for a given parameter.
        Provides sensible defaults for various environmental parameters.
        """
        configs = {
            # Emissions configs (units assumed consistent with model, e.g., kg/day or ppm/day)
            ("emissions", "CO2"): {"distribution": "normal", "mean": 500, "std": 100},
            ("emissions", "CH4"): {"distribution": "exponential", "scale": 2},
            ("emissions", "NOx"): {"distribution": "normal", "mean": 15, "std": 5},
            ("emissions", "SOx"): {"distribution": "normal", "mean": 10, "std": 3},
            ("emissions", "PM10"): {"distribution": "normal", "mean": 5, "std": 1.5},
            ("emissions", "PM2.5"): {"distribution": "normal", "mean": 2.5, "std": 1},

            # Water configs (only water_usage maps directly)
            ("water", "pH"): {"distribution": "normal", "mean": 7.5, "std": 0.5},
            ("water", "turbidity"): {"distribution": "exponential", "scale": 5},
            ("water", "dissolved_oxygen"): {"distribution": "normal", "mean": 8.5, "std": 1.5},
            ("water", "temperature"): {"distribution": "normal", "mean": 18, "std": 3},
            ("water", "conductivity"): {"distribution": "normal", "mean": 500, "std": 100},
            ("water", "total_suspended_solids"): {"distribution": "exponential", "scale": 50},
            ("water", "water_usage"): {"distribution": "normal", "mean": 1000, "std": 200}, # Added config for usage (e.g., Liters/day)

            # Energy configs (kWh/day or similar)
            ("energy", "consumption_kwh"): {"distribution": "normal", "mean": 2500, "std": 500},
            # Note: renewable_percentage doesn't map directly to MachinerySimulationResult
            ("energy", "renewable_percentage"): {"distribution": "normal", "mean": 30, "std": 10},
            # Mapping efficiency_rating to operational_efficiency (0-1 or 0-100 scale assumed)
            ("energy", "efficiency_rating"): {"distribution": "normal", "mean": 0.85, "std": 0.08},
            ("energy", "peak_demand"): {"distribution": "normal", "mean": 120, "std": 20},

            # Waste configs (No direct mapping to MachinerySimulationResult)
            ("waste", "total_waste_tons"): {"distribution": "normal", "mean": 50, "std": 10},
            ("waste", "recycled_percentage"): {"distribution": "normal", "mean": 40, "std": 15},
            ("waste", "hazardous_waste"): {"distribution": "exponential", "scale": 3},
            ("waste", "tailings_volume"): {"distribution": "normal", "mean": 1000, "std": 200},

            # Biodiversity configs (No direct mapping)
            ("biodiversity", "species_count"): {"distribution": "poisson", "lam": 25},
            ("biodiversity", "habitat_quality_index"): {"distribution": "normal", "mean": 70, "std": 10},
            ("biodiversity", "revegetation_area"): {"distribution": "normal", "mean": 150, "std": 30},
            ("biodiversity", "protected_area_ratio"): {"distribution": "normal", "mean": 0.25, "std": 0.1}
        }

        # Return config if exists, otherwise use default
        # Using a generic default might not be meaningful if it doesn't map to a model field
        return configs.get((category, parameter), {"distribution": "normal", "mean": 0, "std": 0})

    def generate_mine(self, session: Session, **kwargs) -> Mine:
        """Creates a Mine object and adds it to the session."""
        mine = Mine(**kwargs)
        session.add(mine)
        return mine

    def generate_machinery_types(self, session: Session, types_data: List[Dict]) -> List[MachineryType]:
        """Creates MachineryType objects from a list of dicts and adds them to the session."""
        machinery_types = []
        for type_data in types_data:
            machinery_type = MachineryType(**type_data)
            session.add(machinery_type)
            machinery_types.append(machinery_type)
        return machinery_types

    def generate_machinery_instances(self, session: Session, mine: Mine,
                                     machinery_types: List[MachineryType],
                                     num_instances_per_type: int = 1) -> List[MachineryInstance]:
        """Creates MachineryInstance objects for a given mine and types, adds to session."""
        machinery_instances = []
        instance_count = 1
        for m_type in machinery_types:
            for i in range(num_instances_per_type):
                instance = MachineryInstance(
                    mine_id=mine.mine_id,
                    type_id=m_type.type_id,
                    instance_name=f"{m_type.type_name} #{instance_count}",
                    operational_status="Active",
                    efficiency_factor=random.uniform(0.9, 1.1), # Add some variability
                    total_operating_hours=0.0 # Start at 0
                )
                session.add(instance)
                machinery_instances.append(instance)
                instance_count += 1 # Ensure unique names if multiple types exist
        session.flush() # Flush to get instance_ids if needed immediately
        return machinery_instances


    def create_simulation_run(self, session: Session, mine: Mine, **kwargs) -> SimulationRun:
        """Creates a SimulationRun object and adds it to the session."""
        sim_run = SimulationRun(mine_id=mine.mine_id, **kwargs)
        session.add(sim_run)
        session.flush() # Ensure run_id is available
        return sim_run

    def generate_machinery_simulation_results(self, session: Session, run: SimulationRun,
                                              machinery_instances: List[MachineryInstance],
                                              start_date: datetime.datetime, end_date: datetime.datetime,
                                              freq: str = 'D', categories: Optional[List[str]] = None,
                                              anomaly_percentage: float = 0.0, anomaly_factor: float = 5.0) -> None:
        """
        Generates time-series simulation results for machinery and adds them to the session.
        Maps generated parameters to MachinerySimulationResult fields where possible.
        Incorporates anomaly generation.

        Args:
            session: SQLAlchemy session.
            run: The SimulationRun object this data belongs to.
            machinery_instances: List of MachineryInstance objects to generate data for.
            start_date: Start date for the time series.
            end_date: End date for the time series.
            freq: Frequency of measurements (pandas offset string).
            categories: Optional list of parameter categories to include.
            anomaly_percentage: Percentage of values to convert to anomalies.
            anomaly_factor: Factor to multiply or divide values by for anomalies.
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        num_dates = len(date_range)

        if categories is None:
            valid_categories = list(self.PARAMETER_CATEGORIES.keys())
        else:
            valid_categories = [cat for cat in categories if cat in self.PARAMETER_CATEGORIES]

        print(f"Generating simulation results for {len(machinery_instances)} instances over {num_dates} steps.")

        for instance in machinery_instances:
            # Store generated values temporarily per instance before creating DB objects
            # Key: simulation_date, Value: Dict {model_field_name: value}
            instance_results_over_time = {date: {} for date in date_range}

            for category in valid_categories:
                parameters = self.PARAMETER_CATEGORIES[category]

                for param_name in parameters:
                    # Check if this parameter maps to a model field
                    model_field = self.PARAMETER_TO_MODEL_FIELD_MAP.get(param_name)
                    if not model_field:
                        # print(f"Skipping parameter '{param_name}' - no direct mapping to MachinerySimulationResult field.")
                        continue # Skip parameters that don't map

                    config = self._get_parameter_config(category, param_name)
                    dist_name = config.pop("distribution", "normal") # Default to normal
                    distribution = self.distributions.get(dist_name, self.distributions["normal"])

                    # Generate base values
                    values = distribution.generate_values(num_dates, **config)

                    # Apply anomalies directly to the generated list
                    num_anomalies = int(num_dates * anomaly_percentage)
                    anomaly_indices = np.random.choice(num_dates, num_anomalies, replace=False)
                    for idx in anomaly_indices:
                        current_value = values[idx]
                        if np.random.random() > 0.5:
                            values[idx] = current_value * anomaly_factor
                        else:
                            # Avoid division by zero if factor is large and value is small
                            if anomaly_factor != 0:
                                values[idx] = current_value / anomaly_factor
                            else:
                                values[idx] = 0 # Or handle as appropriate

                    # Store these values, potentially aggregating if multiple params map to the same field
                    for i, date in enumerate(date_range):
                        # Round the value
                        val = round(values[i], 3)
                        # Handle potential aggregation (e.g., multiple emission types mapping to 'emissions_produced')
                        # For simplicity, we overwrite here. A sum or average might be better depending on requirements.
                        instance_results_over_time[date][model_field] = val


            # Now create the MachinerySimulationResult objects for this instance
            for date, field_values in instance_results_over_time.items():
                if not field_values: # Skip if no mappable parameters were generated for this date
                    continue

                result = MachinerySimulationResult(
                    run_id=run.run_id,
                    instance_id=instance.instance_id,
                    simulation_number=1, # Assuming a single simulation trace per run for now
                    simulation_date=date,
                    # Populate fields based on generated data
                    emissions_produced=field_values.get("emissions_produced"),
                    energy_consumed=field_values.get("energy_consumed"),
                    water_used=field_values.get("water_used"),
                    operational_efficiency=field_values.get("operational_efficiency"),
                    # Other fields (hours_operated, fuel_consumed) are not generated here by default
                    # hours_operated=field_values.get("hours_operated"), # Example if generated
                    # fuel_consumed=field_values.get("fuel_consumed"),  # Example if generated
                )
                session.add(result)

        print("Finished generating simulation results.")


    @staticmethod
    def save_to_csv(dataframe: pd.DataFrame, filename: str) -> None:
        """Save a pandas DataFrame to a CSV file."""
        try:
            dataframe.to_csv(filename, index=False)
            print(f"DataFrame successfully saved to {filename}")
        except Exception as e:
            print(f"Error saving DataFrame to CSV: {e}")


# Example usage
if __name__ == "__main__":
    DB_FILE = "simulation_data.db"
    DATABASE_URL = f"sqlite:///{DB_FILE}"

    # --- Database Setup ---
    engine = create_engine(DATABASE_URL)
    # Create tables if they don't exist
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session: Session = SessionLocal()

    print(f"Database '{DB_FILE}' created/connected and tables ensured.")

    # --- Data Generation ---
    generator = SustainabilityParameterGenerator()

    # 1. Create a Mine
    mine_data = {
        "mine_name": "GigaMine Alpha",
        "mine_type": "Open Pit",
        "ore_type": "Copper",
        "production_capacity": 50000.0, # tons/day
        "mine_lifespan_months": 240,
        "construction_start": datetime.datetime(2023, 1, 1),
        "production_start": datetime.datetime(2024, 1, 1),
        "closure": datetime.datetime(2044, 1, 1),
    }
    try:
        mine = generator.generate_mine(session, **mine_data)
        session.flush() # Get the mine_id
        print(f"Created Mine: {mine.mine_name} (ID: {mine.mine_id})")

        # 2. Create Machinery Types
        machinery_types_data = [
            {"type_name": "Excavator XL", "category": "Excavation", "fuel_type": "Diesel", "emissions_factor": 150.5, "energy_consumption": 0}, # energy_consumption 0 for diesel?
            {"type_name": "Haul Truck HT500", "category": "Hauling", "fuel_type": "Diesel", "emissions_factor": 210.0, "energy_consumption": 0},
            {"type_name": "Electric Drill ED-1", "category": "Drilling", "fuel_type": "Electric", "emissions_factor": 5.0, "energy_consumption": 80.0}, # kWh/hr
            {"type_name": "Water Pump WP-H", "category": "Processing", "fuel_type": "Electric", "emissions_factor": 2.0, "energy_consumption": 50.0, "water_usage": 5000.0 } # L/hr
        ]
        machinery_types = generator.generate_machinery_types(session, machinery_types_data)
        session.flush()
        print(f"Created {len(machinery_types)} Machinery Types: {[mt.type_name for mt in machinery_types]}")

        # 3. Create Machinery Instances for the Mine
        # Create 2 of each type for this mine
        machinery_instances = generator.generate_machinery_instances(session, mine, machinery_types, num_instances_per_type=2)
        print(f"Created {len(machinery_instances)} Machinery Instances for Mine ID {mine.mine_id}")

        # 4. Create a Simulation Run
        sim_run_data = {
            "simulation_name": "Baseline Operation 2024",
            "num_years": 1, # For this example run
            "num_simulations": 1, # Number of simulation traces
            "random_seed": 123,
            "description": "Daily operational parameters for the first year."
        }
        simulation_run = generator.create_simulation_run(session, mine, **sim_run_data)
        print(f"Created Simulation Run: {simulation_run.simulation_name} (ID: {simulation_run.run_id})")

        # 5. Generate Machinery Simulation Results
        start_date = datetime.datetime(2024, 1, 1)
        end_date = datetime.datetime(2024, 12, 31) # Full year
        generator.generate_machinery_simulation_results(
            session=session,
            run=simulation_run,
            machinery_instances=machinery_instances,
            start_date=start_date,
            end_date=end_date,
            freq='D', # Daily data
            categories=["emissions", "energy", "water"], # Focus on mappable categories
            anomaly_percentage=0.02, # Add 2% anomalies
            anomaly_factor=10.0 # Anomalies are 10x or 1/10th
        )

        # --- Commit Changes ---
        session.commit()
        print("Successfully generated data and committed to the database.")

        # --- (Optional) Query and Save to CSV ---
        # Example: Query results for the first machine and save
        if machinery_instances:
            first_instance_id = machinery_instances[0].instance_id
            results_df = pd.read_sql(
                session.query(MachinerySimulationResult)
                       .filter(MachinerySimulationResult.instance_id == first_instance_id)
                       .statement,
                session.bind
            )
            print(f"Querying results for Machinery Instance ID: {first_instance_id}")
            print(results_df.head())
            # generator.save_to_csv(results_df, f"machinery_{first_instance_id}_results.csv")

    except Exception as e:
        print(f"An error occurred: {e}")
        session.rollback() # Roll back changes on error
    finally:
        session.close() # Close the session
        print("Session closed.")