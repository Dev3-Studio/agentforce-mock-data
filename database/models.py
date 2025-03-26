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
    mine_lifespan = Column(Integer)
    initial_footprint = Column(Float)
    maximum_footprint = Column(Float)
    pre_existing_biodiversity = Column(Float)
    construction_start_year = Column(Integer)
    production_start_year = Column(Integer)
    peak_production_year = Column(Integer)
    closure_start_year = Column(Integer)
    post_closure_monitoring = Column(Integer)

    # Relationships
    parameters = relationship("SustainabilityParameter", back_populates="mine")
    simulation_runs = relationship("SimulationRun", back_populates="mine")


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
