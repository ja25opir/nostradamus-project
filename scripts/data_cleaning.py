import argparse
import re
import os


def read_sentence(file):
    """
    Read textfile and pass every line to clean_sentence method.
    :param file: str
        path to textfile with sentences
    :return: list
        with cleaned sentences
    """
    try:
        cleaned_sentences = []
        with open(file, "r", encoding="utf-8") as uncleaned_text:
            text = uncleaned_text.readlines()
            for sentence in text:
                cleaned_sentences.append(clean_sentence(sentence))
        return cleaned_sentences
    except IOError as e:
        print(f"An error occurred while opening {file}: {e}")


def clean_sentence(sentence):
    """
    Cleans sentences from given strings or regexes.
    :param sentence: str
        sentence that should be cleaned
    :return: str
        cleaned sentence
    """
    nbsp = " "
    leading_chars = r"^[^[a-zA-Z1-9(“'\"]*"
    html = r"<[^<]+?>"

    sentence = sentence.replace(nbsp, " ")
    sentence = re.sub(leading_chars, "", sentence)
    sentence = re.sub(html, "", sentence)
    return sentence


def write_data(sentence_set, output):
    """
    Writes cleaned sentences into specified output file.
    :param sentence_set: set
        set of sentences without duplicates
    :param output:
        path where output file should be saved
    """
    try:
        cleaned_sentences = open(f"{output}.txt", "w", encoding="utf-8")
        for sentence in sentence_set:
            cleaned_sentences.writelines(sentence)
        cleaned_sentences.close()
    except IOError as e:
        print(f"An error occurred while writing {output}.txt: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean candidate sentences for further processing.")

    parser.add_argument("--clean_data", nargs=2,
                        help="Read .txt files from given path. Cleans and combines the content to output file.")
    args = parser.parse_args()

    input_dir = args.clean_data[0]
    output_file = args.clean_data[1]

    sentences = []
    for input_file in os.listdir(input_dir):
        if input_file.endswith(".txt"):
            sentences = [*sentences, *read_sentence(input_dir + input_file)]

    if sentences:
        write_data(set(sentences), input_dir + output_file)
