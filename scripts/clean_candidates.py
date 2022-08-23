import argparse
import re
import os


def clean_sentence(sentence):
    leading_chars = r"^[^[a-zA-Z1-9(“'\"]*"
    # urls = r"\((?P<url>https?://[^\s]+)\)"
    html = r"<[^<]+?>"
    regex_filter = [leading_chars, html]

    sentence = sentence.replace(" ", " ")
    for regex in regex_filter:
        sentence = re.sub(regex, "", sentence)
    return sentence


def read_sentence(file):
    try:
        cleaned_sentences = []
        with open(file, "r", encoding="utf-8") as uncleaned_text:
            text = uncleaned_text.readlines()
            for sentence in text:
                cleaned_sentences.append(clean_sentence(sentence))
        return cleaned_sentences
    except IOError as e:
        print(f"An error occurred while opening {file}: {e}")


def write_data(sentence_set, path):
    try:
        cleaned_sentences = open(path+"output.txt", "w", encoding="utf-8")
        for sentence in sentence_set:
            cleaned_sentences.writelines(sentence)
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

    sentences = []
    for input_file in os.listdir(input_dir):
        if input_file.endswith(".txt"):
            sentences = [*sentences, *read_sentence(input_dir+input_file)]
            # sentences.append(read_sentence(input_dir + input_file))
            # uncleaned_sentence = read_sentence(input_dir + input_file)
            # sentences.append(clean_sentence(input_sentence, regex_filter))

    # cleaned_sentences = clean_sentence(sentences, regex_filter)

    if sentences:
        write_data(set(sentences), output_dir)
