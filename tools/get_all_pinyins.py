from pypinyin import Style, pinyin

filename = "bb_audio_text_train_filelist.txt"

_non_pinyin_chars = '；：，。！？-“”《》、（）…— ＢＰ'

with open(filename, encoding='utf-8') as f:
    text_lines = [line.strip().split('|')[1] for line in f]

all_pinyin_set = set()
for line in text_lines:
    for py in pinyin(line, style=Style.TONE3):
        is_pinyin = True
        for c in py[0]:
            if c in _non_pinyin_chars:
                is_pinyin = False
                break
        if is_pinyin:
            all_pinyin_set.add(py[0])

all_pinyin_set = sorted(all_pinyin_set)
print(type(all_pinyin_set))
print(all_pinyin_set)
print(len(all_pinyin_set))

output = "bb_all_pinyins.txt"
with open(output, 'w', encoding='utf-8') as f:
    for py in all_pinyin_set:
        f.write(f"{py}\n")