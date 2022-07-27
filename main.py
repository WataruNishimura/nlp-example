import glob
import re
from unittest import result

import MeCab
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import classification_report, confusion_matrix


class RulebasedEstimator(BaseEstimator, TransformerMixin):
  def __init__(self, label_encoder):
    self.le = label_encoder

  def fit(self, X, y):
    return self

  def predict(self, X):
    result = []
    for text in X:
      pred = 0
      if re.search(r"(スポーツ|サッカー)", text):
        pred = self.le.transform(["sports-watch"])[0]
      elif re.search(r"(独女|悩み|ブライド|サプライズ)", text):
        pred = self.le.transform(["dokujo-tsushin"])[0]
      elif re.search(r"(Mac|PC|IT|インテル|SSD|CPU)", text):
        pred = self.le.transform(["it-life-hack"])[0]
      elif re.search(r"(電力|冷蔵庫|扇風機|家電|熱中症)", text):
        pred = self.le.transform(["kaden-channel"])[0]
      elif re.search(r"(店|店舗|マラソン|スポーツ)", text):
        pred = self.le.transform(["livedoor-homme"])[0]
      elif re.search(r"(映画|芸能|リアクション|公開)", text):
        pred = self.le.transform(["movie-enter"])[0]
      elif re.search(r"(食べ物|菓子|ごはん|ワイン|カフェ)", text):
        pred = self.le.transform(["peachy"])[0]
      elif re.search(r"(ゲーム|スマホ)", text):
        pred = self.le.transform(["smax"])[0]
      elif re.search(r"(メダル|プレー)", text):
        pred = self.le.transform(["sports-watch"])[0]
      elif re.search(r"(テレビ|ジャーナリスト)", text):
        pred = self.le.transform(["topic-news"])[0]

      result.append(pred)
    return result

def parse_to_wakati(text):
  return tagger.parse(text).strip()

categories = [
  "sports-watch",
  "topic-news",
  "dokujo-tsushin",
  "peachy",
  "movie-enter",
  "kaden-channel",
  "livedoor-homme",
  "smax",
  "it-life-hack"
]

docs = []

for category in categories:
  for f in glob.glob(f"./text/{category}/{category}*.txt"):
    with open(f, "r") as fin:
      url = next(fin).strip()
      date = next(fin).strip()
      title = next(fin).strip()
      body = "\n".join([line.strip() for line in fin if line.strip()])

      docs.append((category, url, date, title, body))

df = pd.DataFrame(docs, columns=["category", "url", "date", "title", "body"])
df["date"] = pd.to_datetime(df["date"])

tagger = MeCab.Tagger("-Owakati")

df = df.assign(body_wakati=df.body.apply(parse_to_wakati))

le = LabelEncoder()
y = le.fit_transform(df.category)

X_train, X_test, y_train, y_test = train_test_split(df.body_wakati, y, test_size=0.2, random_state=42, shuffle=True)

rulebased = RulebasedEstimator(label_encoder=le)
rulebased_pred = rulebased.predict(X_test)

print(classification_report(y_test, rulebased_pred, target_names=le.classes_))