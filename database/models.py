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
    mine_id = Column(Integer, ForeignKey("mines.mine_id"))
    parameter_category = Column(String)
    parameter_name = Column(String)
    parameter_value = Column(Text)

    # Relationships
    mine = relationship("Mine", back_populates="parameters")


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
