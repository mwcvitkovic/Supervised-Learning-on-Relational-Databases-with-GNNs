import os

from __init__ import data_root
from data.utils import create_datapoints_with_xargs, get_neo4j_db_driver

if __name__ == '__main__':
    db_name = 'kddcup2014'

    driver = get_neo4j_db_driver(db_name)
    with driver.session() as session:
        datapoint_ids = session.run('MATCH (p:Project) RETURN p.project_id').value()

    base_query = 'MATCH r = (p:Project)--(n)-[*0..1]->(m) \
                             WHERE p.project_id = "{}" \
                             RETURN p, r, n, m'

    target_dir = os.path.join(data_root, db_name, 'preprocessed_datapoints')
    os.makedirs(target_dir, exist_ok=False)

    n_jobs = 20

    create_datapoints_with_xargs(db_name, datapoint_ids, base_query, target_dir, n_jobs)
