import re

with open('narrative_text.txt', 'r') as file:
    file_content = file.read()

time_pattern = re.compile(r'(\d\d:\d\d:\d\d|\d\d:\d\d)')


print(time_pattern.findall(file_content))