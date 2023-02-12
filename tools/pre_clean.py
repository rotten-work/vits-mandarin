import zhon.cedict

filename = "bb_audio_text_val_filelist.txt"

with open(filename, encoding='utf-8') as f:
    text_lines = [line.strip().split('|')[1] for line in f]

_punctuation_sc = '；：，。！？-“”《》、（）…— '
non_chinese_character_set = set()
for line in text_lines:
    for c in line:
        if (c not in zhon.cedict.all and
            c not in _punctuation_sc):
            non_chinese_character_set.add(c)

print(non_chinese_character_set)
