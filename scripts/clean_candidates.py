import re

special_chars = ["\)|\*|\+|\||,|/|;|:|@|]|}|<|>|_|%|$|~"]    # find all non ascii characters + special chars
non_ascii = [r"[^\x00-\x7F]+"]
other = "com|epl|htm|jpg|org|php"
html = ["<[^<]+?>"]
nbsp = [" "]

reg = "|".join(html+nbsp)
regex = "|".join(special_chars+non_ascii)

with open("../data/test.txt", "r", encoding='utf-8') as input_text:
    input = set(input_text.readlines())


with open("../data/finished.txt", "w", encoding='utf-8') as output_text:
    for line in input:
        if re.match(reg, line):
            output_text.writelines(re.sub(reg, "", line))
        elif re.match(other, line[:3]):
            output_text.writelines(line[3:])
        elif re.match(regex, line[:1]):
            output_text.writelines(line[1:])
        else:
            output_text.writelines(line)
