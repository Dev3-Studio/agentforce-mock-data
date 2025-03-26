# DataBase Models

from datetime import datetime

from sqlalchemy import (Column, DateTime, Float, ForeignKey, Integer, MetaData,
                        String, Text)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Mine(Base):

    __tablename__ = 'mines'

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
    mine_id = Column(Integer, ForeignKey('mines.mine_id'))
    parameter_category = Column(String)
    parameter_name = Column(String)
    parameter_value = Column(Text)
    
    # Relationships
    mine = relationship("Mine", back_populates="parameters")
