# DataBase Models

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Mine(Base):

    __tablename__ = "mines"

    mine_id = Column(Integer, primary_key=True)
    mine_name = Column(String, nullable=False)
    mine_type = Column(String, nullable=False)
    ore_type = Column(String, nullable=False)
    production_capacity = Column(Float)
    mine_lifespan_months = Column(Integer)
    initial_footprint = Column(Float)
    maximum_footprint = Column(Float)
    pre_existing_biodiversity = Column(Float)
    construction_start = Column(DateTime)
    production_start = Column(DateTime)
    peak_production = Column(DateTime)
    closure = Column(DateTime)
    post_closure_monitoring = Column(Integer)

    # Relationships
    equipment = relationship("Equipment", back_populates="mine")
    simulation_runs = relationship("SimulationRun", back_populates="mine")

class Equipment(Base):

    __tablename__ = "equipment"

    equipment_id = Column(Integer, primary_key=True)
    equipment_name = Column(String, nullable=False)
    equipment_type = Column(String, nullable=False)
    equipment_start = Column(DateTime, nullable=False)
    equipment_lifespan_months = Column(Integer)

    # Relationships
    parameters = relationship("SustainabilityParameter", back_populates="equipment")

class SustainabilityParameter(Base):

    __tablename__ = "sustainability_parameters"

    param_id = Column(Integer, primary_key=True)
    equipment_id = Column(Integer, ForeignKey("equipment.equipment_id"))
    parameter_category = Column(String)
    parameter_name = Column(String)
    parameter_value = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    equipment = relationship("Equipment", back_populates="parameters")


class MiningEvent(Base):

    __tablename__ = "mining_events"

    event_id = Column(Integer, primary_key=True)
    event_name = Column(String)
    event_category = Column(String)
    event_description = Column(Text)
    typical_duration = Column(Integer)
    duration_variance = Column(Float)
    prerequisite_events = Column(Text)  # JSON array
    mutually_exclusive_events = Column(Text)  # JSON array
    probability_function = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    impacts = relationship("EventImpact", back_populates="event")
    simulation_events = relationship("SimulationEvent", back_populates="event")
    scenario_events = relationship("ScenarioEvent", back_populates="event")


class EventImpact(Base):

    __tablename__ = "event_impacts"

    impact_id = Column(Integer, primary_key=True)
    event_id = Column(Integer, ForeignKey("mining_events.event_id"))
    parameter_category = Column(String)
    parameter_name = Column(String)
    impact_type = Column(String)
    impact_mean = Column(Float)
    impact_std_dev = Column(Float)
    impact_distribution = Column(String)
    recovery_rate_mean = Column(Float)
    recovery_rate_std_dev = Column(Float)
    max_recovery_percentage = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    event = relationship("MiningEvent", back_populates="impacts")


class SimulationRun(Base):

    __tablename__ = "simulation_runs"

    run_id = Column(Integer, primary_key=True)
    mine_id = Column(Integer, ForeignKey("mines.mine_id"))
    simulation_name = Column(String)
    num_years = Column(Integer)
    num_simulations = Column(Integer)
    random_seed = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    description = Column(Text)

    # Relationships
    mine = relationship("Mine", back_populates="simulation_runs")
    events = relationship("SimulationEvent", back_populates="simulation_run")
    results = relationship("SimulationResult", back_populates="simulation_run")


class SimulationEvent(Base):

    __tablename__ = "simulation_events"

    sim_event_id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("simulation_runs.run_id"))
    event_id = Column(Integer, ForeignKey("mining_events.event_id"))
    simulation_number = Column(Integer)
    start_year = Column(Integer)
    end_year = Column(Integer)
    actual_impact_values = Column(Text)  # JSON object
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    simulation_run = relationship("SimulationRun", back_populates="events")
    event = relationship("MiningEvent", back_populates="simulation_events")


class SimulationResult(Base):

    __tablename__ = "simulation_results"

    result_id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("simulation_runs.run_id"))
    datetime = Column(DateTime)
    simulation_number = Column(Integer)
    metric_name = Column(String)
    metric_value = Column(Float)

    # Relationships
    simulation_run = relationship("SimulationRun", back_populates="results")


class ScenarioDefinition(Base):

    __tablename__ = "scenario_definitions"

    scenario_id = Column(Integer, primary_key=True)
    scenario_name = Column(String)
    scenario_description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    events = relationship("ScenarioEvent", back_populates="scenario")


class ScenarioEvent(Base):

    __tablename__ = "scenario_events"

    scenario_event_id = Column(Integer, primary_key=True)
    scenario_id = Column(Integer, ForeignKey("scenario_definitions.scenario_id"))
    event_id = Column(Integer, ForeignKey("mining_events.event_id"))
    start = Column(DateTime)
    probability_override = Column(Float)
    event_parameters = Column(Text)  # JSON object

    # Relationships
    scenario = relationship("ScenarioDefinition", back_populates="events")
    event = relationship("MiningEvent", back_populates="scenario_events")


class MachineryType(Base):
    """
    Defines types of machinery used in mining operations.
    """

    __tablename__ = "machinery_types"

    type_id = Column(Integer, primary_key=True)
    type_name = Column(String, nullable=False)
    category = Column(String, nullable=False)  # Excavation, Hauling, Processing, etc.
    description = Column(Text)
    fuel_type = Column(String)  # Diesel, Electric, Hybrid, etc.
    emissions_factor = Column(Float)  # CO2e per hour of operation
    energy_consumption = Column(Float)  # Energy units per hour
    water_usage = Column(Float)  # Water usage per hour (if applicable)
    maintenance_interval = Column(Integer)  # Hours between maintenance

    # Relationships
    machinery_instances = relationship(
        "MachineryInstance", back_populates="machinery_type"
    )


class MachineryInstance(Base):
    """
    Specific machinery instances owned by a mine.
    """

    __tablename__ = "machinery_instances"

    instance_id = Column(Integer, primary_key=True)
    mine_id = Column(Integer, ForeignKey("mines.mine_id"), nullable=False)
    type_id = Column(Integer, ForeignKey("machinery_types.type_id"), nullable=False)
    instance_name = Column(String, nullable=False)
    operational_status = Column(String)  # Active, Maintenance, Retired, etc.
    efficiency_factor = Column(
        Float, default=1.0
    )  # Relative to typical (1.0 = typical)
    total_operating_hours = Column(Float, default=0.0)

    # Relationships
    mine = relationship("Mine", back_populates="machinery")
    machinery_type = relationship("MachineryType", back_populates="machinery_instances")
    simulation_results = relationship(
        "MachinerySimulationResult", back_populates="machinery_instance"
    )


class MachinerySimulationResult(Base):
    """
    Results of machinery simulations.
    """

    __tablename__ = "machinery_simulation_results"

    result_id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("simulation_runs.run_id"), nullable=False)
    instance_id = Column(
        Integer, ForeignKey("machinery_instances.instance_id"), nullable=False
    )
    simulation_number = Column(Integer, nullable=False)
    simulation_date = Column(DateTime, nullable=False)
    hours_operated = Column(Float)
    fuel_consumed = Column(Float)
    emissions_produced = Column(Float)
    energy_consumed = Column(Float)
    water_used = Column(Float)
    operational_efficiency = Column(Float)

    # Relationships
    simulation_run = relationship("SimulationRun")
    machinery_instance = relationship(
        "MachineryInstance", back_populates="simulation_results"
    )
