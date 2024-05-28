
# Info

This is repo for Selected Topics in Data Science PhD course

Project is dedicated to the topic of Table Data Neural Architecture Search.

# Preparations

You need to download datasets from NIPS paper, mentioned in the report. 


`wget https://huggingface.co/datasets/puhsu/tabular-benchmarks/resolve/main/data.tar -O tabular-dl-tabr.tar.gz`

`tar -xvf tabular-dl-tabr.tar.gz`


In order to run experiment you would need to build container.

Write mounts pointing to cloned repo and downloaded datasets in compose.yaml 

`docker compose build`

`docker compose up -d`

`docker exec -it <container_name> bash`

`cd /home/table_nas`

Here you run all experiments, described in the report, via bash experiments/run_experiment_n.sh

Outputs will be in the directory /home/experiments

