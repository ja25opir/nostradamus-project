- ``candidates.pkl``: contains 7790 sentences that are possibly statements about the future \
-> **train-test-data for active learning:** \
``candidates_labeled.pkl``: contains 200 manually labeled sentences\
``candidates_unlabeled.pkl``: contains 7590 unlabeled sentences\
-> see Datasheet.pdf for a detailed overview


- **input data for trained model** \
```classification_set.pkl```: contains 20488 unlabeled sentences


- **output data**\
``future_classification.pkl``: contains 20488 labeled sentences, 9802 classified as statement about the future \
``emotion_classificaiton.pkl``: contains 9802 labeled sentences with 7 emotion classes
