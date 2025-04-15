from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import database_exists, create_database
import os
from dotenv import load_dotenv
from datetime import datetime

# Import the database models
from models import Base, Mine, SustainabilityParameter, MiningEvent, EventImpact, \
    SimulationRun, SimulationEvent, SimulationResult, ScenarioDefinition, \
    ScenarioEvent, MachineryType, MachineryInstance, MachinerySimulationResult

# Load environment variables from .env file (if present)
load_dotenv()

# Database connection configuration
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')

# Create database URL
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def setup_database():
    """
    Set up the database connection, create the database if it doesn't exist,
    and initialize all tables based on the defined models.
    """
    # Create an engine without specifying the database first
    temp_engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/")

    # Create the database if it doesn't exist
    if not database_exists(DATABASE_URL):
        try:
            with temp_engine.connect() as conn:
                conn.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
            print(f"Database '{DB_NAME}' created successfully.")
        except Exception as e:
            print(f"Error creating database: {e}")
            return None

    # Create engine with the database
    engine = create_engine(DATABASE_URL)

    # Create all tables defined in the models
    try:
        Base.metadata.create_all(engine)
        print("All tables created successfully.")
    except Exception as e:
        print(f"Error creating tables: {e}")
        return None

    # Create a session maker
    Session = sessionmaker(bind=engine)

    return engine, Session


def test_connection(Session):
    """Test the database connection by adding and retrieving a sample record."""
    session = Session()

    try:
        # Check if we have any mines in the database
        mine_count = session.query(Mine).count()

        if mine_count == 0:
            # Add a sample mine if none exists
            sample_mine = Mine(
                mine_name="Test Mine",
                mine_type="Open Pit",
                ore_type="Gold",
                production_capacity=5000.0,
                mine_lifespan_months=240,
                initial_footprint=100.0,
                maximum_footprint=500.0,
                pre_existing_biodiversity=0.75,
                construction_start=datetime.now(),
                production_start=datetime(2025, 6, 1),
                peak_production=datetime(2027, 1, 1),
                closure=datetime(2045, 1, 1),
                post_closure_monitoring=60
            )

            session.add(sample_mine)
            session.commit()
            print("Sample mine added successfully.")

        # Retrieve mines to verify connection
        mines = session.query(Mine).all()
        print(f"Connected successfully. Found {len(mines)} mines in the database.")

        for mine in mines:
            print(f"Mine ID: {mine.mine_id}, Name: {mine.mine_name}, Type: {mine.mine_type}")

    except Exception as e:
        print(f"Error testing connection: {e}")
    finally:
        session.close()


def main():
    """Main function to set up and test the database."""
    print("Setting up database connection...")
    result = setup_database()

    if result:
        engine, Session = result
        print("Database setup complete.")

        # Test the connection
        test_connection(Session)
    else:
        print("Failed to set up database.")


if __name__ == "__main__":
    main()