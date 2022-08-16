# Research Plan

Can be taken from the project exposé (/documentation/expose.pdf). Deviations will be documented.

# Definition of a "Statement"

Statements are sentences that express a fact, idea or opinion. Statements do not ask questions, make requests or give
commands. They are also not exclamations.

Statement sentences can be simple, compound or complex sentences; a sentence always consists of at least one clause
containing a subject and a verb and nearly always ends in a full stop. 
https://www.theschoolrun.com/what-statement

# How to use the web-archive-pipeline

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

## execute python scripts

```
HADOOP_USER_NAME=$USER 
srun --export=ALL --pty --mem=50g --container-name=web-archive-keras \
--container-image=ghcr.io#niklasdeckers/web-archive-keras:master \
--container-mounts=/mnt/ceph:/mnt/ceph --container-writable \
--gres=gpu:1g.5gb bash -c " cd /mnt/ceph/storage/data-tmp/teaching-current/$USER/web-archive-keras \
&& PYTHONPATH=. HADOOP_CONF_DIR="./hadoop/" python3 examples/meme_classifier/meme_classifier_pipeline.py"
```
