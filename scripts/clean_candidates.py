import argparse
import re
import os


def read_sentence(file):
    try:
        with open(file, "r", encoding="utf-8") as uncleaned_sentence:
            sentence = uncleaned_sentence.read()
        return sentence
    except IOError as e:
        print(f"An error occurred while opening {file}: {e}")


def clean_sentence(sentence, filter):
    sentence = sentence.replace(" ", " ")
    for regex in filter:
        sentence = re.sub(regex, "", sentence)
    return sentence


def write_data(sentence_set, path):
    try:
        cleaned_sentences = open(path+"output.txt", "w", encoding="utf-8")
        for sentence in sentence_set:
            cleaned_sentences.writelines(sentence + "\n")
        cleaned_sentences.close()
    except IOError as e:
        print(f"An error occurred while writing \"output.txt\": {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean candidate sentences for further processing.")

    parser.add_argument("--input_dir", nargs=1, default="/",
                        help="Specify input directory containing .txt files with uncleaned sentences."
                             " Default is current directory.")
    parser.add_argument("--output_dir", nargs=1, default="/",
                        help="Specify output directory, where the cleaned and combined output.txt is saved to. "
                             "Default is current directory")
    args = parser.parse_args()

    input_dir = args.input_dir[0]
    output_dir = args.output_dir[0]

    leading_chars = r"^[^[a-zA-Z1-9(“'\"]*"
    # urls = r"\((?P<url>https?://[^\s]+)\)"
    html = r"<[^<]+?>"

    regex_filter = [leading_chars, html]
    sentences = []

    for input_file in os.listdir(input_dir):
        if input_file.endswith(".txt"):
            input_sentence = read_sentence(input_dir + input_file)
            sentences.append(clean_sentence(input_sentence, regex_filter))

    if sentences:
        write_data(set(sentences), output_dir)
