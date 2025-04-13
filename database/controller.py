from database.db_config import DatabaseConnector


def test_db_connection(connection_string: str):
    db = DatabaseConnector(connection_string=connection_string)
    result = db.connect()

    return result


if __name__ == "__main__":
    connection_string = "sqlite:///test_db.db"
    print(test_db_connection(connection_string))
