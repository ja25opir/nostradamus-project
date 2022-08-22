import re
import os


def read_sentence(file):
    with open(file, "r", encoding="utf-8") as uncleaned_sentence:
        sentence = uncleaned_sentence.read()
    return sentence


def clean_sentence(sentence, filter):
    sentence = sentence.replace(" ", " ")
    for regex in filter:
        sentence = re.sub(regex, "", sentence)
    return sentence


def write_data(sentence_set, path):
    cleaned_sentences = open(path+"output.txt", "w", encoding="utf-8")
    for sentence in sentence_set:
        cleaned_sentences.writelines(sentence+"\n")
    cleaned_sentences.close()


if __name__ == "__main__":
    input_dir = "../warc-dumps/test_dir/"
    output_dir = "../warc-dumps/test_dir/"

    leading_chars = r"^[^[a-zA-Z1-9(“'\"]*"
    # urls = r"\((?P<url>https?://[^\s]+)\)"
    html = r"<[^<]+?>"

    regex_filter = [leading_chars, html]
    sentences = []

    for input_file in os.listdir(input_dir):
        input_sentence = read_sentence(input_dir + input_file)
        sentences.append(clean_sentence(input_sentence, regex_filter))

    write_data(set(sentences), output_dir)
