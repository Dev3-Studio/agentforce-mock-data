import json
from typing import List, Any
import requests
from sqlalchemy.inspection import inspect


class SalesforceIngestor:
    def __init__(self, client_id, client_secret, username, password, auth_url, instance_url=None):
        self.client_id = client_id
        self.client_secret = client_secret
        self.username = username
        self.password = password
        self.auth_url = auth_url
        self.instance_url = instance_url
        self.access_token = None
        self.authenticate()

    def authenticate(self):
        data = {
            'grant_type': 'password',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'username': self.username,
            'password': self.password
        }
        response = requests.post(self.auth_url, data=data)
        response.raise_for_status()
        auth_data = response.json()
        self.access_token = auth_data['access_token']
        self.instance_url = auth_data['instance_url']
        print("[Salesforce] Authenticated successfully.")

    def serialize(self, obj: Any, depth: int = 1) -> dict:
        """
        Serialize SQLAlchemy model, including nested relationships up to given depth
        """
        if depth < 0 or obj is None:
            return None

        result = {}
        mapper = inspect(obj).mapper

        # Serialize column fields
        for column in mapper.column_attrs:
            val = getattr(obj, column.key)
            if hasattr(val, "isoformat"):
                val = val.isoformat()
            result[column.key] = val

        # Serialize relationships
        for rel in mapper.relationships:
            rel_val = getattr(obj, rel.key)
            if rel_val is None:
                result[rel.key] = None
            elif rel.uselist:
                result[rel.key] = [self.serialize(child, depth=depth - 1) for child in rel_val]
            else:
                result[rel.key] = self.serialize(rel_val, depth=depth - 1)

        return result

    def ingest(self, data_stream_api_name: str, records: List[Any], depth: int = 1):
        """
        Send serialized model data to the specified Salesforce Data Cloud stream
        """
        url = f"{self.instance_url}/services/data/v1/datalake/ingest/{data_stream_api_name}"
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        payload = {
            "data": [self.serialize(record, depth=depth) for record in records]
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        try:
            response.raise_for_status()
            print(f"[Salesforce] Successfully ingested {len(records)} record(s) into {data_stream_api_name}.")
        except requests.HTTPError as e:
            print(f"[Salesforce] Ingestion failed: {response.status_code} - {response.text}")
            raise e

# Example usage:
# if __name__ == "__main__":

    # ingestor = SalesforceIngestor(
    #     client_id='your-client-id',
    #     client_secret='your-client-secret',
    #     username='your-username',
    #     password='your-password+security_token',
    #     auth_url='https://login.salesforce.com/services/oauth2/token'
    # )
    #
    #
    # mines = session.query(Mine).all()
    # ingestor.ingest("mine_stream", mines, depth=2)
    #
    # events = session.query(MiningEvent).all()
    # ingestor.ingest("event_stream", events, depth=2)