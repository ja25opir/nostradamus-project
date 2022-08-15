import collections
import json
import re
import threading
import time
import base64
import resiliparse.parse.lang
import numpy as np

from pipelines.text_pipeline import TextPipeline
from pipelines.tools.passthrough_model import PassthroughModelPipeline


class RegexCounterPipeline(PassthroughModelPipeline, CandidatePipeline):
    """
    This pipeline allows to search for regex occurrences within the texts from the text pipeline.
    The texts are discarded; only the count is reported.
    No GPU functionality is used.
    """

    def __init__(self, regex, out_dir):
        self.regex = regex
        max_content_length = 1000000000
        super().__init__(out_dir=out_dir, max_content_length=max_content_length)

    def get_distributed_filter(self):
        regex = self.regex
        acc_counter = self.acc_counter

        def distributed_filter(text):
            if len(text) < 1000:  # only extract long texts
                return False
            n_matches = len(regex.findall(text))
            if n_matches == 0:
                return False
            if not resiliparse.parse.lang.detect_fast(text)[0] == "en":  # only extract english texts
                return False
            acc_counter.add(collections.Counter({"n_regex_matches": n_matches}))
            return True

        return distributed_filter

    def tokenizer(self, text):
        reg_matches = self.regex.findall(text)
        sentences = re.split('[.!?\n]', text)   # split text up into sentences
        matches = " ###".join([s for s in sentences if any(r in s for r in reg_matches)])   # search sentences for regex matches
        return matches


    def export(self, prediction, export_text, url):
        prediction = np.reshape(prediction, ())
        #tokenizer = self.get_tokenizer()    # added tokenizer to save only candidate sentences
        print(url.decode("utf-8"), prediction)
        with open(f"{self.out_dir}/{base64.urlsafe_b64encode(url[:128]).decode('utf-8')}_{prediction:1.4f}.txt",
                  "w") as f:
            f.write(self.tokenizer(export_text.decode("utf-8")))  # call tokenizer when writing to file

    def start_threads(self):
        def save_stats():
            while True:
                time.sleep(60)
                with open(f"{self.out_dir}/stats.json", 'w') as f:
                    json.dump(self.acc_counter.value, f)

        threading.Thread(target=save_stats, daemon=True).start()

        super().start_threads()


if __name__ == "__main__":
    interesting_snippets = [
        "someday",
        "in the future",
        "upcoming (?:times?|months?|years?|decades?)",
        "(?:times?|months?|years?|decades?) [is|are|will]*\s?(?:comes?|coming)",
        "(?:times?|months?|years?|decades?) [is|are|will]*\s?[lays?|laying]*\s?ahead",
        "in (?:a|the next|several)\s?[few]*\s?(?:months?|years?|decades?)",
        "in [the next]*\s?[like|about|around|maybe|ca]*\s?a?\s?(?:one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|billion) (?:months?|years?|decades?)",
        "in [the next]*\s?[like|about|around|maybe|ca]*\s?a?\s?[1-9][0-9]?[0-9]?[0-9]?[0-9]?[0-9]? (?:months?|years?|decades?)"
    ]

    regex = "|".join(interesting_snippets)
    regex = re.compile(regex, re.IGNORECASE)    # precompile regexes
    out_dir = "data/regex_counter/out/"
    p = RegexCounterPipeline(regex, out_dir)
    p.run()
