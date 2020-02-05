import os
import subprocess

from __init__ import data_root
from data.utils import get_db_container

db_name = 'kddcup2014'


def build_database_from_kaggle_files():
    # Fix errors in essays.csv file
    local_essays_dir = os.path.join(data_root, 'raw_data', db_name)
    local_essays_path = os.path.join(local_essays_dir, 'essays.csv')
    subprocess.run('unzip \'{dir}/*.zip\' -d {dir} '.format(dir=local_essays_dir), shell=True)
    subprocess.run('sed -i \'s/\\\\\\\\""//g\' {}'.format(local_essays_path), shell=True)
    subprocess.run('sed -i \'s/\\\\""//g\' {}'.format(local_essays_path), shell=True)
    subprocess.run('sed -i \'s/\\\\",/",/g\' {}'.format(local_essays_path), shell=True)
    subprocess.run('sed -i \'s/\\\\"$/"/g\' {}'.format(local_essays_path), shell=True)

    container_id = get_db_container(db_name)
    cmd = 'docker exec -i {} cypher-shell < {}'.format(container_id,
                                                       os.path.join(data_root, db_name,
                                                                    'kddcup2014_neo4j_loader.cypher'))
    print(cmd)
    subprocess.run(cmd, shell=True)


if __name__ == '__main__':
    # You have to download the kaggle files manually, after making an account
    build_database_from_kaggle_files()
