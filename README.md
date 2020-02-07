# Supervised Learning on Relational Databases with Graph Neural Networks

This is code to reproduce the results in the paper [Supervised Learning on Relational Databases with Graph Neural Networks](https://arxiv.org/abs/2002.02046).

## Install dependencies

The file `docker/whole_project/environment.yml` lists all dependencies you need to install to run this code.

You can follow the instructions
 [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) 
to automatically install a conda environment from this file.

You can also build a [docker](https://docs.docker.com/) container which contains all dependencies.   You'll need docker (or nvidia-docker if you want to use a GPU) installed to do this.
 The file `docker/whole_project/Dockerfile` builds a container that can run all experiments.
   

## Get datasets

I would love to have a link here where you could just download the prepared datasets.  But unfortunately that would violate the Kaggle terms of service.

So you either need to follow the instructions below and build them yourself, or reach out to me by email and I may be able to provide them to you.


### Preparing the datasets yourself

1) Set the `data_root` variable in `/__init__.py` to be the location where you'd like to install the datasets.  Default is `<HOME>/RDB_data`.

2) Download raw dataset files from Kaggle.  You need a Kaggle account to do this.  You only need to download the datasets you're interested in.

    a) Put the [Acquire Valued Shoppers Challenge](https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data) data in `data_root/raw_data/acquirevaluedshopperschallenge`. Extract any compressed files.
    
    b) Put the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data) data in `data_root/raw_data/homecreditdefaultrisk`. Extract any compressed files.

    c) Put the [KDD Cup 2014](https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose/data) data in `data_root/raw_data/kddcup2014`. Extract any compressed files.

3) Build the docker container specified in `docker/neo4j/Dockerfile`.  This creates a container with the [neo4j](https://neo4j.com/) graph database installed, which is used to build the datasets.

4) Start the database server(s) for the datasets you want to build:

   ```docker run -d -e "NEO4J_dbms_active__database=<db_name>.graph.db" --publish=7474:<port_for_browser> --publish=7687:<port_for_db> --mount type=bind,source=<path_to_code>/data/datasets/<db_name>,target=/data rdb-neo4j``` 
  
   where `<path_to_code>` is the location of this repo on your system, `<port_for_browser>` is an optional port for using the build-in neo4j data viewer (you can set it as `7474` if you don't care), and (`<db_name>`, `<port_for_db>`) are (`acquirevaluedshopperschallenge`, `9687`),  (`homecreditdefaultrisk`, `10687`), or (`kddcup2014`, `7687`), respectively.

5) Run `python -m data.<db_name>.build_database_from_kaggle_files` from the root directory of this repo.

6) (optional) To view the dataset in the built-in neo4j data viewer, navigate to `<your_machine's_ip_address>:7474` in a web browser, run `:server disconnect` to log off whatever your web browser thinks is the default neo4j server, and log into the right one by specifying `<port_for_browser>` in the web interface.

7) Run `python -m data.<db_name>.build_dataset_from_database` from the root directory of this repo.

8) Run `python -m data.<db_name>.build_db_info` from the root directory of this repo.

9) (optional) to create the tabular and DFS datasets used in the experiments, run `python -m data.<db_name>.build_DFS_features` from the root directory of this repo.  Then run `python -m data.<db_name>.build_tabular_datasets` from the root directory of this repo.


### Add your own datasets

If you have your own relational dataset you'd like to use this system with, you can copy and modify the code in one of the `data/acquirevaluedshopperschallenge`, `data/homecreditdefaultrisk`, or `data/kddcup2014` directories to suit your purposes.

The main thing you have to do is create the `.cypher` script to get your data into a neo4j database.  Once you've done that, nearly all the dataset building code is reusable.
You'll also have to add your dataset's name in a few places in the codebase, e.g. in the `__init__` method of the `DatabaseDataset` class.

## Run experiments

All experiments are started with the scripts in the `experiments` directory.

For example, to recreate the `PoolMLP` row in paper tables 3 and 4, you would run `python -m experiments.GNN.PoolMLP` from the root directory of this repo to start training, then run `python -m experiments.evaluate_experiments` when training is finished, and finally run `python -m experiments.GNN.print_and_plot_results`.

By default, experiments run in [tmux](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/) windows on your local machine.  But you can also change the argument in the `run_script_with_kwargs` command at the bottom of each experiment script to run them in a local docker container.
Or you can export the docker image built with `docker/whole_project/Dockerfile` to AWS ECR and modify the arguments in `experiments/utils/run_script_with_kwargs` to run all experiments on AWS Batch.


# License

The content of the notes linked above is licensed under the [Creative Commons Attribution 3.0 license](http://creativecommons.org/licenses/by/3.0/us/deed.en_US), and the code in this repo is licensed under the [MIT license](http://opensource.org/licenses/mit-license.php).
