import pandas as pd
import jieba

df = pd.read_csv('../resource/Simplified_Chinese_Multi-Emotion_Dialogue_Dataset.csv')
print(df.describe())
print(df.loc[:,'label'].value_counts())

input_keys = df.loc[:,"text"]
print(input_keys.shape)

def count_keywords(keys: pd.Series) -> int:
    input_cut_list = keys.apply(lambda x: jieba.lcut(x))
    word_set = set()
    for value in input_cut_list.values:
        for word in value:
            word_set.add(word)
    return len(word_set)

if __name__ == '__main__':
    print(count_keywords(input_keys))
