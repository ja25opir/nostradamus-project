import re

special_chars = ["\)|\*|\+|\||,|/|;|:|@|]|}|<|>|_|%|$|~"]    # find all non ascii characters + special chars
non_ascii = [r"[^\x00-\x7F]+"]
other = "com |epl |htm |jpg |org |php "
html = "<[^<]+?>"
nbsp = " "

regex = "|".join(special_chars+non_ascii)

with open("../data/all2.txt", "r", encoding='utf-8') as input_text:
    input = set(input_text.readlines())


output = []
append = True

for line in input:
    if nbsp in line:
        output.append(line.replace(nbsp, ""))
        append = False

    if re.match(regex, line[:1]):
        output.append(line[1:])
        append = False

    if re.match(html, line):
        output.append(re.sub(html, "", line))
        append = False

    if re.match(other, line[:3]):
        output.append(line[3:])
        append = False

    if append:
        output.append(line)

    append = True

with open("../data/candidates.txt", "w", encoding='utf-8') as output_text:
    output_text.writelines(output)
