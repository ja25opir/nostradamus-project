# 1) Research Plan

Can be taken from the project exposé (/documentation/expose.pdf). Deviations will be documented.

# 2) Definition of a "Statement about the Future"

## Definition of a Statement

"Statements are sentences that express a fact, idea or opinion. Statements do not ask questions, make requests or give
commands. They are also not exclamations.

Statement sentences can be simple, compound or complex sentences; a sentence always consists of at least one clause
containing a subject and a verb and nearly always ends in a full stop." \
source: https://www.theschoolrun.com/what-statement

## When do we classify a Statement as "about the Future"?

A sentence must match the previous described definition of a statement. Furthermore it must be related to the future,
e.g. what will happen at some point in the future.

An example statement from the candidate dataset created from our regex pre-filtering that wouldn't be classified as "
about the Future" is:
"USGPRU officials said this was the first fatality in the series in nine years.". This sentence refers to a period of
time in the past.

# 3) Data used for the future statements extraction model

``data/candidates_labeled.pkl`` contains 88 Statements about the Future and 112 sentences that don't match the
requirements described in 2). \
``data/candidates_unlabeled.pkl`` contains 7590 unlabeled candidate sentences.

# 4) WARC-DL configurations

## create config.ini

```
[s3]
BUCKET_NAMES = ["corpus-iwo-internet-archive-wide00001"]
AWS_ACCESS_KEY_ID = RVVS0V6VXYYSXS7PFJ25
AWS_SECRET = XoqNg1RdrTsRxEoKQWs7vjao7S9b745uUcmte4ab
ENDPOINT_URL = http://s3.dw.webis.de:7480/

[pyspark]
SPARK_INSTANCES = 5
enable_prebuilt_dependencies = yes

[tensorflow]
BATCHSIZE = 20

[profiler]
enable_logging = no
logging_delay_s = 120
logging_duration_s = 60
```

# 5) Setup the docker container

## Set your enroot credentials

```
touch $HOME/.config/enroot/.credentials
chmod 600 $HOME/.config/enroot/.credentials
echo "machine ghcr.io login USERNAME password ACCESS_TOKEN" >> $HOME/.config/enroot/.credentials
```

## Import the container from the registry

```
srun --mem=32g enroot import --output pipelineimg.sqsh docker://ghcr.io#ja25opir/nostradamus-project:main
```

# 6) Usage of data preprocessing scripts

## Extract candidates from WARC-Files with regular expressions

``` 
HADOOP_USER_NAME=$USER srun --export=ALL --pty --mem=50g --container-name=pipeline1 
--container-image=./pipelineimg.sqsh --container-mounts=/mnt/ceph:/mnt/ceph --container-writable 
--gres=gpu:1g.5gb bash -c " cd /mnt/ceph/storage/data-tmp/teaching-current/ja25opir/WARC-DL && 
PYTHONPATH=. HADOOP_CONF_DIR="./hadoop/" python3 ../nostradamus-project/scripts/future_regex_finder.py"
```

## Clean and merge candidates into one text-file

TODO

## Create .pkl from .txt file

```
python3 scripts/data_preprocessing.py --import_data data/NAME.txt OUTPUTNAME
```

## Shuffle candidates set and split into a labeled und unlabeled set + start manual labeling process

```
python3 scripts/data_preprocessing.py --start_labeling 200
```

## Information options
``scripts/data_preprocessing.py --show_data``: Prints the head of a given DataFrame. \
``scripts/data_preprocessing.py --count_classes``: Counts all candidates belonging to the classes 1 and 0 of a given DataFrame.