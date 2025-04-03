import random
import datetime
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


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
        return list(np.random.normal(mean, std, n))


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
    """Generator for sustainability parameter data"""

    # Define parameter categories and possible names
    PARAMETER_CATEGORIES = {
        "emissions": ["CO2", "CH4", "NOx", "SOx", "PM10", "PM2.5"],
        "water": ["pH", "turbidity", "dissolved_oxygen", "temperature", "conductivity", "total_suspended_solids"],
        "energy": ["consumption_kwh", "renewable_percentage", "efficiency_rating", "peak_demand"],
        "waste": ["total_waste_tons", "recycled_percentage", "hazardous_waste", "tailings_volume"],
        "biodiversity": ["species_count", "habitat_quality_index", "revegetation_area", "protected_area_ratio"]
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
        Get suitable distribution configuration for a given parameter
        This provides sensible defaults for various environmental parameters
        """
        configs = {
            # Emissions configs (in kg or ppm)
            ("emissions", "CO2"): {"distribution": "normal", "mean": 5000, "std": 1000},
            ("emissions", "CH4"): {"distribution": "exponential", "scale": 20},
            ("emissions", "NOx"): {"distribution": "normal", "mean": 150, "std": 50},
            ("emissions", "SOx"): {"distribution": "normal", "mean": 100, "std": 30},
            ("emissions", "PM10"): {"distribution": "normal", "mean": 45, "std": 15},
            ("emissions", "PM2.5"): {"distribution": "normal", "mean": 25, "std": 10},

            # Water configs
            ("water", "pH"): {"distribution": "normal", "mean": 7.5, "std": 0.5},
            ("water", "turbidity"): {"distribution": "exponential", "scale": 5},
            ("water", "dissolved_oxygen"): {"distribution": "normal", "mean": 8.5, "std": 1.5},
            ("water", "temperature"): {"distribution": "normal", "mean": 18, "std": 3},
            ("water", "conductivity"): {"distribution": "normal", "mean": 500, "std": 100},
            ("water", "total_suspended_solids"): {"distribution": "exponential", "scale": 50},

            # Energy configs
            ("energy", "consumption_kwh"): {"distribution": "normal", "mean": 25000, "std": 5000},
            ("energy", "renewable_percentage"): {"distribution": "normal", "mean": 30, "std": 10},
            ("energy", "efficiency_rating"): {"distribution": "normal", "mean": 75, "std": 8},
            ("energy", "peak_demand"): {"distribution": "normal", "mean": 1200, "std": 200},

            # Waste configs
            ("waste", "total_waste_tons"): {"distribution": "normal", "mean": 500, "std": 100},
            ("waste", "recycled_percentage"): {"distribution": "normal", "mean": 40, "std": 15},
            ("waste", "hazardous_waste"): {"distribution": "exponential", "scale": 30},
            ("waste", "tailings_volume"): {"distribution": "normal", "mean": 10000, "std": 2000},

            # Biodiversity configs
            ("biodiversity", "species_count"): {"distribution": "poisson", "lam": 25},
            ("biodiversity", "habitat_quality_index"): {"distribution": "normal", "mean": 70, "std": 10},
            ("biodiversity", "revegetation_area"): {"distribution": "normal", "mean": 1500, "std": 300},
            ("biodiversity", "protected_area_ratio"): {"distribution": "normal", "mean": 0.25, "std": 0.1}
        }

        # Return config if exists, otherwise use default
        return configs.get((category, parameter), {"distribution": "normal", "mean": 100, "std": 20})

    def generate_parameter_data(self, equipment_ids: List[int], start_date: datetime.datetime,
                                end_date: datetime.datetime, freq: str = 'D', 
                                categories: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate sustainability parameter data for specified equipment over a time period

        Args:
            equipment_ids: List of equipment IDs to generate data for
            start_date: Start date for the time series
            end_date: End date for the time series
            freq: Frequency of measurements (D=daily, H=hourly, etc.)
            categories: Optional list of parameter categories to include (all if None)

        Returns:
            DataFrame containing the generated parameters
        """
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        num_dates = len(date_range)

        # Use all categories if none specified
        if categories is None:
            categories = list(self.PARAMETER_CATEGORIES.keys())
        else:
            # Validate that provided categories exist
            categories = [cat for cat in categories if cat in self.PARAMETER_CATEGORIES]

        data = []
        param_id = 1

        for equipment_id in equipment_ids:
            for category in categories:
                parameters = self.PARAMETER_CATEGORIES[category]

                for param_name in parameters:
                    config = self._get_parameter_config(category, param_name)
                    dist_name = config.pop("distribution")

                    # Get the distribution strategy
                    if dist_name not in self.distributions:
                        # Default to normal if distribution not found
                        dist_name = "normal"
                        config = {"mean": 100, "std": 20}

                    distribution = self.distributions[dist_name]

                    # Generate values using the distribution
                    values = distribution.generate_values(num_dates, **config)

                    # Create data points for each date
                    for i, date in enumerate(date_range):
                        data.append({
                            "param_id": param_id,
                            "equipment_id": equipment_id,
                            "parameter_category": category,
                            "parameter_name": param_name,
                            "parameter_value": str(round(values[i], 3)),
                            "timestamp": date
                        })
                        param_id += 1

        return pd.DataFrame(data)

    def save_to_csv(self, dataframe: pd.DataFrame, filename: str) -> None:
        """Save the generated data to a CSV file"""
        dataframe.to_csv(filename, index=False)

    def generate_anomalies(self, dataframe: pd.DataFrame, anomaly_percentage: float = 0.05,
                           anomaly_factor: float = 5.0) -> pd.DataFrame:
        """
        Add anomalies to the dataset by scaling values up or down significantly

        Args:
            dataframe: The original dataframe
            anomaly_percentage: Percentage of values to convert to anomalies
            anomaly_factor: Factor to multiply or divide values by

        Returns:
            DataFrame with anomalies
        """
        # Create a copy to avoid modifying the original
        df = dataframe.copy()

        # Number of rows to modify
        num_anomalies = int(len(df) * anomaly_percentage)

        # Randomly select rows to modify
        anomaly_indices = np.random.choice(len(df), num_anomalies, replace=False)

        for idx in anomaly_indices:
            # Get current value
            try:
                current_value = float(df.loc[idx, "parameter_value"])

                # Randomly decide whether to increase or decrease
                if np.random.random() > 0.5:
                    new_value = current_value * anomaly_factor
                else:
                    new_value = current_value / anomaly_factor

                # Update the value
                df.loc[idx, "parameter_value"] = str(round(new_value, 3))
            except ValueError:
                # Skip if value cannot be converted to float
                continue

        return df


# Example usage
if __name__ == "__main__":
    # Create the generator
    generator = SustainabilityParameterGenerator()

    # Generate data
    start_date = datetime.datetime(2024, 1, 1)
    end_date = datetime.datetime(2024, 1, 31)
    equipment_ids = [1001, 1002, 1003]

    data = generator.generate_parameter_data(
        equipment_ids=equipment_ids,
        start_date=start_date,
        end_date=end_date,
        freq='D',
        categories=["emissions", "water"]  # Only include these categories
    )

    # Add some anomalies
    data_with_anomalies = generator.generate_anomalies(data, anomaly_percentage=0.03)

    # Save to CSV
    generator.save_to_csv(data_with_anomalies, "sustainability_parameters.csv")

    # Display sample of the data
    print(data_with_anomalies.head(10))