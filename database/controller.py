import logging

from database.db_config import DatabaseConnector
from database.models import Base

# Set up logging
logger = logging.getLogger(__name__)


def create_database_tables(connection_string: str):

    try:
        logger.info("Creating database tables")

        # Initialize database connector
        db = DatabaseConnector(connection_string=connection_string)

        # Connect to database
        if not db.connect():
            logger.error("Failed to connect to database")
            return False

        # Create tables
        result = db.create_tables(Base)

        if not result:
            logger.error("Failed to create database tables")
            return False

        logger.info("Database tables created successfully")
        return True

    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        raise


def delete_database_tables(connection_string: str):

    try:
        logger.info("Deleting database tables")

        # Initialize database connector
        db = DatabaseConnector(connection_string=connection_string)

        # Connect to database
        if not db.connect():
            logger.error("Failed to connect to database")
            return False

        # Create tables
        result = db.drop_all_tables()

        if not result:
            logger.error("Failed to delete database tables")
            return False

        logger.info("Database tables deleted successfully")
        return True

    except Exception as e:
        logger.error(f"Error deleting database tables: {str(e)}")
        raise


if __name__ == "__main__":
    connection_string = "sqlite:///test_db.db"

    _ = create_database_tables(connection_string=connection_string)
    _ = delete_database_tables(connection_string=connection_string)
