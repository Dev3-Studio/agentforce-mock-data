from SalesforceDataCloudIngestor import SalesforceIngestor



ingestor = SalesforceIngestor(
        client_id='your-client-id',
        client_secret='your-client-secret',
        username='your-username',
        password='your-password+security_token',
        auth_url='https://login.salesforce.com/services/oauth2/token'
    )


mines = session.query(Mine).all()
ingestor.ingest("mine_stream", mines, depth=2)

events = session.query(MiningEvent).all()
ingestor.ingest("event_stream", events, depth=2)