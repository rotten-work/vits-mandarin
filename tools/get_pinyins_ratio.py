from pypinyin import Style, pinyin

all_pinyins_file = "bb_all_pinyins.txt"
filename = "lex_audio_text_train_filelist.txt"

all_pinyin_set = set()
with open(all_pinyins_file, encoding='utf-8') as f:
    for line in f:
        all_pinyin_set.add(line.strip())

with open(filename, encoding='utf-8') as f:
    text_lines = [line.strip().split('|')[1] for line in f]

# This set may contain items that are not pinyin.
# But if we get the intersection of all_pinyin_set and pinyin_set,
# we can get all valid pinyins in pinyin_set.
pinyin_set = set()
for line in text_lines:
    for py in pinyin(line, style=Style.TONE3):
        pinyin_set.add(py[0])

# https://realpython.com/python-sets/
valid_pinyin_set = all_pinyin_set & pinyin_set
print(f"Valid pinyins the file include are: {sorted(valid_pinyin_set)}")
print(f"Pinyins that the file doesn't include are: {sorted(all_pinyin_set - valid_pinyin_set)}")
print(f"The number of pinyins in the file is: {len(valid_pinyin_set)}")
print(f"The coverage ratio of pinyins in the file to all pinyins are {len(valid_pinyin_set) / len(all_pinyin_set)}")
