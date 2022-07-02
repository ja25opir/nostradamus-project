# Research Plan

Can be taken from the project exposé. Deviations are listed here later.

# How to use the web-archive-pipeline
## create config.ini
```
[s3]
BUCKET_NAMES = ["corpus-iwo-internet-archive-wide00001"]
AWS_ACCESS_KEY_ID = ***REMOVED***
AWS_SECRET = ***REMOVED***
ENDPOINT_URL = https://s3.dw.webis.de/ <http://s3.dw.webis.de:7480/>

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