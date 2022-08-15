import collections
import json
import re
import threading
import time
import resiliparse.parse.lang
from bs4 import BeautifulSoup
from pipelines.tools.passthrough_model import PassthroughModelPipeline
import abc
import base64
import os
from collections import Counter
import numpy as np
import tensorflow as tf
from fastwarc.warc import ArchiveIterator
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse import detect_encoding
from resiliparse.parse.html import HTMLTree
from helpers import create_s3_client, get_file_stream
from pipelines.pipeline import Pipeline


class CandidatePipeline(Pipeline, abc.ABC):
    """
    This pipeline extracts texts from websites from the WARC files. It streams the following to the driver/GPU:
    An (optionally tokenized) version of the website text, which should be as clean as possible (useful for neural
    network input),
    an original version of the text as a string,
    the website url.
    """

    def __init__(self, out_dir, max_content_length):
        self.out_dir = out_dir
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
        self.max_content_length = max_content_length

        super().__init__()

    def get_signature(self):
        return (
            self.get_tokens_spec(),  # text for classification
            tf.TensorSpec(shape=(), dtype=tf.string),  # text for export
            tf.TensorSpec(shape=(), dtype=tf.string))  # url

    def get_distributed_filter(self):
        """
        Overridable method that provides a filter, which is executed on the pyspark cluster nodes.
        The returned distributed_filter must not use self. Needed attributes of self should be extracted into variables
        outside of the definition of distributed_filter, which may then use these variables.
        """

        def distributed_filter(text):
            return True

        return distributed_filter

    def get_tokens_spec(self):
        """
        Overridable method that returns a tf.TensorSpec which corresponds to the values returned by the tokenizer
        defined in get_tokenizer().
        """

        return tf.TensorSpec(shape=(), dtype=tf.string)

    def get_tokenizer(self):
        """
        Overridable method that provides a tokenizer, which is executed on the pyspark cluster nodes.
        The returned tokenizer must not use self. Needed attributes of self should be extracted into variables
        outside of the definition of tokenizer, which may then use these variables.
        """

        def tokenizer(text):
            return text

        return tokenizer

    def get_generator_factory(self):
        acc_counter = self.acc_counter
        max_content_length = self.max_content_length
        distributed_filter = self.get_distributed_filter()
        tokenizer = self.get_tokenizer()
        AWS_ACCESS_KEY_ID = self.AWS_ACCESS_KEY_ID
        AWS_SECRET = self.AWS_SECRET
        ENDPOINT_URL = self.ENDPOINT_URL

        def generator_factory(file_identifier):
            s3_client = create_s3_client(AWS_ACCESS_KEY_ID, AWS_SECRET, ENDPOINT_URL)
            stream = get_file_stream(s3_client, file_identifier)
            for record in ArchiveIterator(stream, max_content_length=max_content_length):
                try:
                    if record.headers is None:
                        acc_counter.add(Counter({"n_record_headers_none": 1}))
                        continue
                    if record.http_headers is None:
                        acc_counter.add(Counter({"n_http_headers_none": 1}))
                        continue
                    if record.headers['WARC-Type'] == 'response' and record.content_length >= 128:
                        content_type = str(record.http_content_type).lower()
                        if content_type.startswith("text/html"):
                            url = str(record.headers['WARC-Target-URI'])
                            html_bytes = record.reader.read()
                            try:
                                encoding = record.http_charset
                                if encoding is None:
                                    encoding = detect_encoding(html_bytes)
                                tree = HTMLTree.parse_from_bytes(html_bytes, encoding)
                            except:
                                acc_counter.add(Counter({"n_parsing_exception": 1}))
                                continue

                            prediction_text = extract_plain_text(tree, preserve_formatting=False,
                                                                 main_content=True, list_bullets=False,
                                                                 alt_texts=False, links=False,
                                                                 form_fields=False, noscript=False)

                            soup = BeautifulSoup(str(tree), 'html.parser')
                            for a in soup('a'):
                                a.decompose()
                            export_text = soup.get_text(strip=True, separator="\n")

                            if not distributed_filter(prediction_text):
                                acc_counter.add(Counter({"n_distributed_filter_not_passed": 1}))
                                continue

                            yield tokenizer(prediction_text), export_text, url
                            acc_counter.add(Counter({"n_node_results": 1}))

                        else:
                            acc_counter.add(Counter({"n_wrong_content_type": 1}))
                    else:
                        acc_counter.add(Counter({"n_wrong_warc_type": 1}))
                except:
                    acc_counter.add(Counter({"n_unhandled_record_exceptions": 1}))
                    continue
            acc_counter.add(Counter({"n_finished_warc_files": 1}))

        return generator_factory

    def export(self, prediction, export_text, url):
        prediction = np.reshape(prediction, ())
        print(url.decode("utf-8"), prediction)
        with open(f"{self.out_dir}/{base64.urlsafe_b64encode(url[:128]).decode('utf-8')}_{prediction:1.4f}.txt",
                  "w") as f:
            f.write(export_text.decode("utf-8"))


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
        "(?:hope|think|believe|wish) someday",
        "in the (?:upcoming times?|times? to come|times? [is|are]\s? coming|times? [is]\s?[lays?|laying]?\s?ahead|future)",
        "the (?:upcoming times?|times? to come|times? [is|are]\s? coming|times? [is]\s?[lays?|laying]?\s?ahead|future) will (?:be|bring|get|support)",
        "(?:good|better|best|bad|worse|worst|sad|sadder|saddest|funny|funnier|funniest|happy|happier|happiest|scary|scarier|scariest|hot|hotter|hottest|bright|horrible|terrible|crazy|cruel|cool|fantastic|fragile|glorious|messy|nice|perfect|strange|ugly|dirty|exciting|beautiful) (?:upcoming times?|times? to come|times? [is|are]\s? coming|times? [is]\s?[lays?|laying]?\s?ahead|future)",
        "In (?:a|the next)\s?[few]*\s?(?:months?|years?)",
        "in [the next][like|about|around|maybe|ca]\s?a?\s?(?:one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|billion) (?:months?|years?)",
        "in [the next]\s?[like|about|around|maybe|ca]\s?a?\s?[1-9][0-9]?[0-9]?[0-9]?[0-9]?[0-9]? (?:months?|years?)"
    ]

    regex = "|".join(interesting_snippets)
    regex = re.compile(regex, re.IGNORECASE)    # precompile regexes
    out_dir = "data/regex_counter/out/"
    p = RegexCounterPipeline(regex, out_dir)
    p.run()
