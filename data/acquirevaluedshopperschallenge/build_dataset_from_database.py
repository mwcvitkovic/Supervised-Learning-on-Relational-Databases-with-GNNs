import os

from __init__ import data_root
from data.utils import create_datapoints_with_xargs, get_neo4j_db_driver

if __name__ == '__main__':
    db_name = 'acquirevaluedshopperschallenge'

    driver = get_neo4j_db_driver(db_name)
    with driver.session() as session:
        datapoint_ids = session.run('MATCH (h:History) RETURN h.history_id').value()

    base_query = 'MATCH (h:History)-[r]-(n)-[s:OFFER_TO_CATEGORY|OFFER_TO_COMPANY|OFFER_TO_BRAND' \
                 '|TRANSACTION_TO_CHAIN|TRANSACTION_TO_CATEGORY|TRANSACTION_TO_COMPANY|TRANSACTION_TO_BRAND*0..1]->(m) \
                              WHERE h.history_id = {}  \
                              RETURN h, r, n, s, m'

    target_dir = os.path.join(data_root, db_name, 'preprocessed_datapoints')
    os.makedirs(target_dir, exist_ok=False)

    n_jobs = 40

    create_datapoints_with_xargs(db_name, datapoint_ids, base_query, target_dir, n_jobs)
