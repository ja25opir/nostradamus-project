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

We use the WARC-DL pipeline and pre-filtering methods from ``future_regex_finder.py`` to extract candidate sentences
that could include statements about the future.

## Training data

With a small manually labeled dataset and active learning we train a model for the classification of Statements about
the Future. \
``data/candidates_labeled.pkl`` contains 88 Statements about the Future and 112 sentences that don't match the
requirements described in 2). \
``data/candidates_unlabeled.pkl`` contains 7590 unlabeled candidate sentences.

## Data labeled by the trained model

``data/classification_set.pkl`` contains 20488 candidate sentences. Our model classified 9802 of these as Statements
about the Future (``data/future_classification.pkl``).

# 4) WARC-DL configurations

## create config.ini

```
[s3]
BUCKET_NAMES = ["corpus-iwo-internet-archive-wide00001"]
AWS_ACCESS_KEY_ID = ***REMOVED***
AWS_SECRET = ***REMOVED***
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

## create hadoop/yarn-site.xml

```
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>

<configuration>
    <property>
        <name>yarn.resourcemanager.hostname</name>
        <value>rm.yarn.webis.de</value>
    </property>
    <property>
        <name>yarn.nodemanager.remote-app-log-dir</name>
        <value>/app-logs</value>
    </property>
    <property>
        <name>yarn.nodemanager.remote-app-log-dir-suffix</name>
        <value>logs</value>
    </property>
    <property>
        <name>yarn.nodemanager.resource.memory-mb</name>
        <value>4096</value>
    </property>
    <property>
        <name>yarn.nodemanager.resource.cpu-vcores</name>
        <value>2</value>
    </property>
    <property>
        <name>yarn.scheduler.maximum-allocation-mb</name>
        <value>4096</value>
    </property>
    <property>
        <name>yarn.scheduler.maximum-allocation-vcores</name>
        <value>2</value>
    </property>

</configuration>
```

## add parameters to pipelines/pipeline.py

```
conf_list.append(("spark.driver.memory","4g"))
conf_list.append(("spark.yarn.executor.memoryOverhead","4g"))
conf_list.append(("spark.executor.memory","4g"))
conf_list.append(("spark.driver.cores","2"))
conf_list.append(("spark.executor.cores","2"))
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
srun --mem=32g enroot import --output nosimg.sqsh docker://ghcr.io#ja25opir/nostradamus-project:main
```

# 6) Usage of data preprocessing scripts

## Extract candidates from WARC-Files with regular expressions

``` 
HADOOP_USER_NAME=$USER srun --export=ALL --pty --mem=50g --container-name=nos1 
--container-image=./nosimg.sqsh --container-mounts=/mnt/ceph:/mnt/ceph --container-writable 
--gres=gpu:1g.5gb bash -c " cd /mnt/ceph/storage/data-tmp/teaching-current/USERNAME/WARC-DL && 
PYTHONPATH=. HADOOP_CONF_DIR="./hadoop/" python3 ../nostradamus-project/scripts/future_regex_finder.py"
```

## Clean and merge candidates into one text-file

```
python3 scripts/data_cleaning.py --clean_data INPUT_PATH OUTPUT_NAME
```

## Create .pkl from .txt file

```
python3 scripts/data_preprocessing.py --import_data data/NAME.txt OUTPUT_NAME
```

## Shuffle candidates set and split into a labeled und unlabeled set + start manual labeling process

```
python3 scripts/data_preprocessing.py --start_labeling 200
```

## Information options

``scripts/data_preprocessing.py --show_data``: Prints the head of a given DataFrame. \
``scripts/data_preprocessing.py --count_classes``: Counts all candidates belonging to the classes 1 and 0 of a given
DataFrame.

# 7) Usage of active learning script

TODO \
use ``--gres=gpu:ampere`` for torch with cuda-support!

# 8) Usage of Future Statements and Emotion classification models
TODO