## Kaggle Competitions

### Tabular Web Data

1. [Personalized Web Search Challenge](https://www.kaggle.com/c/yandex-personalized-web-search-challenge) 2013

1. [Restaurant Revenue Prediction](https://www.kaggle.com/competitions/restaurant-revenue-prediction/overview) 2015

1. [Click-Through Rate Prediction](https://www.kaggle.com/competitions/avazu-ctr-prediction/overview) 2015

1. [Expedia Hotel Recommendations](https://www.kaggle.com/competitions/expedia-hotel-recommendations/data) 2016

1. [Instacart Market Basket Analysis](https://www.kaggle.com/competitions/instacart-market-basket-analysis/data?select=aisles.csv.zip) 2017

1. [Mercari Price Suggestion Challenge](https://www.kaggle.com/c/mercari-price-suggestion-challenge) 2018

1. [TalkingData AdTracking Fraud Detection Challenge](https://www.kaggle.com/competitions/talkingdata-adtracking-fraud-detection/overview) 2018

1. [Elo Merchant Category Recommendation](https://www.kaggle.com/competitions/elo-merchant-category-recommendation/overview) 2019

1. [M5 Forecasting - Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/overview) 2020

1. [M5 Forecasting - Uncertainty](https://www.kaggle.com/competitions/m5-forecasting-uncertainty/overview) 2020

1. [Shopee - Price Match Guarantee](https://www.kaggle.com/c/shopee-product-matching) 2021

1. [PetFinder.my - Pawpularity Contest](https://www.kaggle.com/c/petfinder-pawpularity-score) 2021

1. [H&M Personalized Fashion Recommendations](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations) 2022

### References

1. [The Most Comprehensive List of Kaggle Solutions and Ideas](https://farid.one/kaggle-solutions/) 2010~2022

1. [https://www.kaggle.com/code/sudalairajkumar/winning-solutions-of-kaggle-competitions/notebook] 2010~2022

1. [Kaggle Past Solutions](https://ndres.me/kaggle-past-solutions/) 2010~2019

### How to use Kaggle notebook

[Kaggle notebookの活用について](https://www.kaggle.com/docs/notebooks:)
* CPU/GPUの実行時間12時間、TPUの実行時間9時間
* 20ギガバイトの自動保存ディスクスペース ( /kaggle/working )
* GitHubを有効にする方法
1. Notebookを開く
1. File メニューで、Link to GitHub オプションを選択

（※初めてリンクする場合は、リンクの許可を明示的に求められます。それ以降の 新しいノートブックにリンクする場合は、この操作は自動的に行われます。自動的に行われます。）

* GPUとTPUの両方で使用できる一定の時間数を得ることができます; TPUの場合は30時間ですが、GPUの場合は週によって異なります。
* GPU/TPUを有効に利用するためには、初めにデータセットを最適化する必要がある。そのために、小さなデータセットで少しずつチューニングしながら実行時間の短縮をはかる必要がある。こうすることで処理時間を最適化し、効率よく利用することができる。

### How Kaggle competitions are run

1. 目標となる指標を検討するところから始める
コンペティションの概要ページの左メニューにあります。評価タブを選択すると、評価指標の詳細が表示されます。
評価指標の公式、それを再現するコード、評価指標の考察などがあります。また、投稿ファイルのフォーマットに関する説明を得ます。

[Meta Kaggleデータセット](https://www.kaggle.com/kaggle/meta-kaggle)
参考として、過去7年間のコンペティションで最も頻繁に使用された評価指標を把握するために使用できます。
（※トップ20の表が競技に使われるすべての指標をカバーしているわけではないことを考慮する必要があります。）

```py
import numpy as np
import pandas as pd
comps = pd.read_csv("/kaggle/input/meta-kaggle/Competitions.csv")
evaluation = ['EvaluationAlgorithmAbbreviation',
 'EvaluationAlgorithmName',
 'EvaluationAlgorithmDescription',]
compt = ['Title', 'EnabledDate', 'HostSegmentTitle']
df = comps[compt + evaluation].copy()
df['year'] = pd.to_datetime(df.EnabledDate).dt.year.values
df['comps'] = 1
time_select = df.year >= 2015
competition_type_select = df.HostSegmentTitle.isin(
['Featured', 'Research'])
pd.pivot_table(df[time_select&competition_type_select],
 values='comps',
 index=['EvaluationAlgorithmAbbreviation'],
 columns=['year'],
 fill_value=0.0,
 aggfunc=np.sum,
 margins=True
 ).sort_values(
 by=('All'), ascending=False).iloc[1:,:].head(20)
```

※自分なりに知識を得たいのであれば、評価関数を自分でコーディングして、不完全でもいいから実験してみることです。

2. 検証のデザイン
* モデリングと提出結果の順位に注目してしまうのは良くある間違いです。
* 結果に対する正しい検証の考えが重要です。
   * 過学習に注意すること
   * 検証を効果的に行う方法を決定すること
   * 良い性能の環境を使うことも試行回数を増やし良い成果を得られることに繋がるが、正しい方向で実験の検証を繰り返すことが重要  

3. 分析の作業段階
   1. データをどのように処理するか
   1. どのようなモデルを適用するか
   1. モデルのアーキテクチャをどう変えるか（特に深層学習モデル）
   1. モデルのハイパーパラメータをどのように設定するか
   1. 予測値の後処理方法 （評価指標の決定と使用）
      ※公開のリーダーボードが非公開のリーダーボードと完全に相関していたとしても、毎日の投稿数が限られているため（すべてのコンペティションに存在する制限）、前述のすべての領域で行うことができるテストの表面を削ることさえできません。
