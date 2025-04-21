# DB connection config
import json
import logging
import os
from datetime import datetime

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import scoped_session, sessionmaker

from database.models import Base


class DatabaseConnector:
    """
    Handles database connection and session management for the mining sustainability database.
    """

    def __init__(self, connection_string: str):
        """
        Initialize the database connector with a connection string.

        Args:
            connection_string (str): SQLAlchemy connection string
                                    (e.g., 'postgresql://username:password@localhost:5432/mining_db')
        """
        self.connection_string = connection_string
        self.engine = None
        self.session_factory = None
        self.Session = None
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def connect(self):
        """
        Establish connection to the database and create session factory.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.logger.info(
                f"Connecting to database with connection string: {self.connection_string}"
            )
            self.engine = create_engine(self.connection_string, echo=False)

            # Test connection
            self.engine.connect()

            # Create session factory
            self.session_factory = sessionmaker(bind=self.engine)
            self.Session = scoped_session(self.session_factory)

            self.logger.info("Database connection established successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to database: {str(e)}")
            return False

    def get_session(self):
        """
        Get a new database session.

        Returns:
            Session: SQLAlchemy session object
        """
        if not self.Session:
            self.connect()
        return self.Session()

    def close_session(self, session):
        """
        Close a database session.

        Args:
            session: SQLAlchemy session to close
        """
        try:
            session.close()
        except Exception as e:
            self.logger.error(f"Error closing session: {str(e)}")

    def create_tables(self, base):
        """
        Create all tables defined in the SQLAlchemy Base.

        Args:
            base: SQLAlchemy declarative base containing model definitions

        Returns:
            bool: True if tables created successfully, False otherwise
        """
        try:
            self.logger.info("Creating database tables")
            base.metadata.create_all(self.engine)
            self.logger.info("Database tables created successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create tables: {str(e)}")
            return False

    def get_table_names(self):
        """
        Get a list of all table names in the database.

        Returns:
            list: List of table names
        """
        try:
            inspector = inspect(self.engine)
            return inspector.get_table_names()
        except Exception as e:
            self.logger.error(f"Failed to get table names: {str(e)}")
            return []

    def execute_query(self, query):
        """
        Execute a raw SQL query.

        Args:
            query (str): SQL query to execute

        Returns:
            result: Query result
        """
        try:
            with self.engine.connect() as connection:
                result = connection.execute(query)
                return result
        except Exception as e:
            self.logger.error(f"Failed to execute query: {str(e)}")
            return None

    def add_record(self, record):
        """
        Add a new record to the database.

        Args:
            record: SQLAlchemy model instance

        Returns:
            bool: True if record added successfully, False otherwise
        """
        session = self.get_session()
        try:
            session.add(record)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to add record: {str(e)}")
            return False
        finally:
            self.close_session(session)

    def add_records(self, records):
        """
        Add multiple records to the database.

        Args:
            records (list): List of SQLAlchemy model instances

        Returns:
            bool: True if records added successfully, False otherwise
        """
        session = self.get_session()
        try:
            session.add_all(records)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to add records: {str(e)}")
            return False
        finally:
            self.close_session(session)

    def update_record(self, record):
        """
        Update an existing record in the database.

        Args:
            record: SQLAlchemy model instance

        Returns:
            bool: True if record updated successfully, False otherwise
        """
        session = self.get_session()
        try:
            session.merge(record)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to update record: {str(e)}")
            return False
        finally:
            self.close_session(session)

    def delete_record(self, record):
        """
        Delete a record from the database.

        Args:
            record: SQLAlchemy model instance

        Returns:
            bool: True if record deleted successfully, False otherwise
        """
        session = self.get_session()
        try:
            session.delete(record)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to delete record: {str(e)}")
            return False
        finally:
            self.close_session(session)

    def drop_all_tables(self):
        """
        Drop all tables in the SQLite database.

        Returns:
            bool: True if tables dropped successfully, False otherwise
        """
        try:
            self.logger.info(
                f"Dropping all tables in SQLite database: {self.connection_string}"
            )

            # Get all table names
            table_names = self.get_table_names()

            if not table_names:
                self.logger.info("No tables found in the database")
                return True

            self.logger.info(
                f"Found {len(table_names)} tables to drop: {', '.join(table_names)}"
            )

            # Execute DROP TABLE statements
            with self.engine.begin() as connection:
                # Disable foreign key constraints
                connection.execute(text("PRAGMA foreign_keys = OFF"))

                # Drop each table
                for table_name in table_names:
                    connection.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                    self.logger.info(f"Dropped table: {table_name}")

                # Re-enable foreign key constraints
                connection.execute(text("PRAGMA foreign_keys = ON"))

            # Verify tables were dropped
            inspector = inspect(self.engine)
            remaining_tables = inspector.get_table_names()

            if remaining_tables:
                self.logger.warning(
                    f"Some tables remain after drop: {', '.join(remaining_tables)}"
                )
                return False

            self.logger.info("All tables dropped successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to drop tables: {str(e)}")
            return False
