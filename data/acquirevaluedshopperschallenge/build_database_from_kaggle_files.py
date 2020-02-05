import os
import subprocess

from __init__ import data_root
from data.utils import get_db_container

db_name = 'acquirevaluedshopperschallenge'


def build_database_from_kaggle_files():
    # Add unique ids to the transactions.csv file
    local_transactions_csv_path = os.path.join(data_root, 'raw_data', db_name, 'transactions.csv')
    local_temp_filepath = os.path.join(data_root, 'raw_data', db_name, 'temp.csv')
    subprocess.run(
        'head -n 1 {} | sed "s|^id,|id,history,|g" > {}'.format(
            local_transactions_csv_path,
            local_temp_filepath),
        shell=True)
    subprocess.run(
        'tail -n +2 {} | pv | awk \'{{printf("%d,%s\\n", NR, $0)}}\' >> {}'.format(
            local_transactions_csv_path,
            local_temp_filepath),
        shell=True)

    # Load data into database
    container_id = get_db_container(db_name)
    cmd = 'docker exec -i {} cypher-shell < {}'.format(container_id,
                                                       os.path.join(data_root, db_name,
                                                                    '{}_neo4j_loader.cypher'.format(db_name)))
    print(cmd)
    subprocess.run(cmd, shell=True)


if __name__ == '__main__':
    # You have to download the kaggle files manually, after making an account
    build_database_from_kaggle_files()
