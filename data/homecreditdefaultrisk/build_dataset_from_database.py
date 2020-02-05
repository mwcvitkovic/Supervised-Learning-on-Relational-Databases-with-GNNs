import os

from __init__ import data_root
from data.utils import create_datapoints_with_xargs, get_neo4j_db_driver

if __name__ == '__main__':
    db_name = 'homecreditdefaultrisk'

    driver = get_neo4j_db_driver(db_name)
    with driver.session() as session:
        datapoint_ids = session.run('MATCH (a:Application) RETURN a.SK_ID_CURR').value()

    # Variable length match since not all applications have bureaus or previous
    base_query = 'MATCH r = (a:Application)-[*0..2]-(n)  \
                 where a.SK_ID_CURR = {} \
                 RETURN a, r, n'

    target_dir = os.path.join(data_root, db_name, 'preprocessed_datapoints')
    os.makedirs(target_dir, exist_ok=False)

    n_jobs = 40

    create_datapoints_with_xargs(db_name, datapoint_ids, base_query, target_dir, n_jobs)
