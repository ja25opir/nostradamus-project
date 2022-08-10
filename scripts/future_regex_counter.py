import collections
import json
import re
import threading
import time

import resiliparse.parse.lang

from candidate_extraction.candidate_pipeline import CandidatePipeline
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
            n_matches = len(re.findall(regex, text))
            if n_matches == 0:
                return False
            if not resiliparse.parse.lang.detect_fast(text)[0] == "en":  # only extract english texts
                return False
            acc_counter.add(collections.Counter({"n_regex_matches": n_matches}))
            return True

        return distributed_filter

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
        "the future will be",
        "in the future we will",
        "in a year",
        "in one year",
        "in ten years",
        "in a hundred years",
        "in a few years",
        "the future will bring",
        "I think in the future"
    ]
    regex = "|".join(interesting_snippets)
    out_dir = "data/regex_counter/out/"
    p = RegexCounterPipeline(regex, out_dir)
    p.run()
