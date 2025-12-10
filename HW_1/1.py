import re

with open('narrative_text.txt', 'r') as file:
    content = file.read()

phone_pattern = re.compile(pattern=r'(\+989\d{9} | 00989\d{9} | 09\d{9})', flags=re.VERBOSE)
phones = phone_pattern.findall(content)

date_pattern = re.compile(pattern=r'(\d{4}-\d{2}-\d{2} | \d{2}/\d{2}/\d{4})', flags=re.VERBOSE)
dates = date_pattern.findall(content)

time_pattern = re.compile(pattern=r'(\d{2}:\d{2}:\d{2} | \d{2}:\d{2})', flags=re.VERBOSE)
times = time_pattern.findall(content)


print("--- PHONE NUMBERS ---")
print(sorted(phones))
print(f"Count: {len(phones)}\n")

print("--- DATES ---")
print(sorted(dates))
print(f"Count: {len(dates)}\n")

print("--- TIMES ---")
print(sorted(times))
print(f"Count: {len(times)}\n")