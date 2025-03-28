readme_content = """
# 邮件分类项目 README

## 一、代码核心功能说明
本项目主要实现了邮件分类的功能，利用文本特征将邮件分为垃圾邮件和正常邮件两类。通过对邮件文本进行处理和特征提取，结合多项式朴素贝叶斯分类器完成分类任务，同时支持高频词特征和 TF - IDF 特征加权两种特征提取模式的切换。此外，针对样本量失衡问题进行了处理，并增加了模型评估指标。

### （一）算法基础
#### 1. 多项式朴素贝叶斯分类器
本仓库采用多项式朴素贝叶斯分类器进行邮件分类。该分类器基于条件概率的特征独立性假设，即假设在给定类别 $C$ 的条件下，各个特征 $x_1,x_2,\cdots,x_n$ 之间是相互独立的。用数学公式表示为：
$P(x_1,x_2,\cdots,x_n|C)=\prod_{i = 1}^{n}P(x_i|C)$
这个假设简化了计算过程，使得我们可以分别计算每个特征在给定类别下的条件概率，然后将它们相乘得到联合条件概率。

#### 2. 贝叶斯定理在邮件分类中的应用
贝叶斯定理的公式为：$P(C|x_1,x_2,\cdots,x_n)=\frac{P(x_1,x_2,\cdots,x_n|C)P(C)}{P(x_1,x_2,\cdots,x_n)}$
在邮件分类问题中，我们的目标是判断一封邮件属于垃圾邮件（$C_1$）还是正常邮件（$C_2$），也就是要比较 $P(C_1|x_1,x_2,\cdots,x_n)$ 和 $P(C_2|x_1,x_2,\cdots,x_n)$ 的大小。由于对于同一封邮件，$P(x_1,x_2,\cdots,x_n)$ 是固定的，所以我们只需要比较分子 $P(x_1,x_2,\cdots,x_n|C)P(C)$ 的大小。结合特征独立性假设，就可以将其转化为 $\prod_{i = 1}^{n}P(x_i|C)P(C)$ 进行计算。

### （二）数据处理流程
#### 1. 分词处理
代码中使用 `jieba.cut()` 方法对邮件文本进行分词处理。具体实现是逐行读取邮件文件，对每行文本进行清洗（去除无效字符）后，调用 `jieba.cut()` 方法将文本切分成单个的词语。例如：with open(filename, 'r', encoding='utf - 8') as fr:
    for line in fr:
        line = line.strip()
        line = re.sub(r'[.【】0 - 9、——。，！~\*]', '', line)
        line = cut(line)
#### 2. 停用词过滤
虽然原代码中没有显式的停用词过滤步骤，但在实际的文本处理中，停用词过滤是一个重要的预处理步骤。停用词是指在文本中频繁出现但对文本分类任务没有太大帮助的词语，如“的”“了”“是”等。可以通过定义一个停用词列表，在分词后过滤掉这些词语。示例代码如下：stopwords = set([line.strip() for line in open('stopwords.txt', 'r', encoding='utf - 8').readlines()])
words = [word for word in words if word not in stopwords]
### （三）特征构建过程
#### 1. 高频词特征选择
##### 数学表达形式
高频词特征选择是基于词频统计的。对于一个由 $N$ 封邮件组成的语料库，我们统计每个词语在所有邮件中出现的频率，选取出现频率最高的 $k$ 个词语作为特征。对于每封邮件，我们构建一个长度为 $k$ 的向量，向量的每个元素表示对应特征词在该邮件中出现的次数。设邮件 $d$ 中特征词 $w_i$ 出现的次数为 $tf_{i,d}$，则邮件 $d$ 的特征向量可以表示为 $\vec{v_d}=(tf_{1,d},tf_{2,d},\cdots,tf_{k,d})$。

##### 实现差异
实现过程中，首先遍历所有邮件文件，将所有词语汇总到一个列表中，然后使用 `collections.Counter()` 统计每个词语的出现次数，选取出现次数最多的 $k$ 个词语作为特征词。对于每封邮件，统计这些特征词在该邮件中出现的次数，构建特征向量。示例代码如下：def get_top_words(top_num, filename_list):
    all_words = []
    for filename in filename_list:
        all_words.extend(get_words(filename))
    freq = Counter(all_words)
    return [i[0] for i in freq.most_common(top_num)]

top_words = get_top_words(100)
vector = []
for words in all_words:
    word_map = list(map(lambda word: words.count(word), top_words))
    vector.append(word_map)
#### 2. TF - IDF 特征加权
##### 数学表达形式
TF - IDF（词频 - 逆文档频率）是一种常用的文本特征加权方法。TF（词频）表示词语在某篇文档中出现的频率，$tf_{i,d}=\frac{n_{i,d}}{\sum_{j}n_{j,d}}$，其中 $n_{i,d}$ 是词语 $w_i$ 在文档 $d$ 中出现的次数，$\sum_{j}n_{j,d}$ 是文档 $d$ 中所有词语出现的总次数。IDF（逆文档频率）表示词语在整个语料库中的普遍重要性，$idf_i=\log\frac{N}{df_i}$，其中 $N$ 是语料库中文档的总数，$df_i$ 是包含词语 $w_i$ 的文档数。TF - IDF 值为 $tfidf_{i,d}=tf_{i,d}\times idf_i$。对于每封邮件，我们构建一个向量，向量的每个元素表示对应特征词的 TF - IDF 值。

##### 实现差异
在代码中，使用 `sklearn.feature_extraction.text.TfidfVectorizer` 来实现 TF - IDF 特征的计算。首先将所有邮件的分词结果组合成一个语料库，然后调用 `TfidfVectorizer` 的 `fit_transform()` 方法计算每个词语的 TF - IDF 值，得到特征矩阵。示例代码如下：corpus = []
for filename in filename_list:
    words = get_words(filename)
    corpus.append(" ".join(words))
vectorizer = TfidfVectorizer()
vector = vectorizer.fit_transform(corpus)
### （四）两种特征选择方式对比
| 特征选择方式 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 高频词特征选择 | 实现简单，计算速度快 | 容易受到常用词的影响，对文本的区分能力可能不足 | 数据规模较小，文本主题相对单一 |
| TF - IDF 特征加权 | 能够突出重要的、具有区分性的词语，对文本的表达能力更强 | 计算复杂度相对较高，需要统计整个语料库的信息 | 数据规模较大，文本主题多样 |

## 二、高频词/TF - IDF 两种特征模式的切换方法
代码中通过 `extract_features` 函数的 `feature_method` 参数来实现两种特征模式的切换。具体如下：def extract_features(filename_list, feature_method='top_words', top_num=100):
   
# 使用 TF - IDF 特征
vector_tfidf, feature_names = extract_features(filename_list, feature_method='tfidf')
## 三、样本平衡处理
### 目标
缓解垃圾邮件（127 条）与普通邮件（24 条）的样本量失衡问题。

### 实现方法
在模型训练前，采用 `imblearn.over_sampling.SMOTE` 进行过采样处理。以下是相关代码示例：from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# 假设已经获取特征向量 X 和标签 y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 使用重采样后的数据进行模型训练
model = MultinomialNB()
model.fit(X_train_resampled, y_train_resampled)
## 四、增加模型评估指标
### 目标
在基础预测功能外，输出包含精度、召回率、F1 值的分类评估报告。

### 实现方法
通过 `sklearn.metrics.classification_report` 实现多维度的模型评估。以下是代码示例：from sklearn.metrics import classification_report
