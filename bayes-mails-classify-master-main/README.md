## 基于朴素贝叶斯的邮件分类系统


## 算法基础
#### 1. 多项式朴素贝叶斯分类器
多项式朴素贝叶斯是一种基于贝叶斯定理和特征条件独立性假设的分类方法。在文本分类任务中，它假设每个特征（词）之间是相互独立的，即在给定类别的条件下，一个词的出现与否不影响其他词的出现概率。

#### 2. 特征独立性假设
基于条件概率，假设我们有文本D由一系列词$w_1,w_2,\cdots,w_n$ 组成，类别为 C 。特征独立性假设意味着$P(w_1,w_2,\cdots,w_n |C)=\prod_{i = 1}^{n}P(w_i|C)$。也就是说，在已知类别的情况下，每个词的出现概率是相互独立的，这样可以大大简化计算。

#### 3. 贝叶斯定理在邮件分类中的应用
贝叶斯定理的公式为：$P(C|x_1,x_2,\cdots,x_n)=\frac{P(x_1,x_2,\cdots,x_n|C)P(C)}{P(x_1,x_2,\cdots,x_n)}$
在邮件分类问题中，我们的目标是判断一封邮件属于垃圾邮件（$C_1$）还是正常邮件（$C_2$），也就是要比较 $P(C_1|x_1,x_2,\cdots,x_n)$ 和 $P(C_2|x_1,x_2,\cdots,x_n)$ 的大小。由于对于同一封邮件，$P(x_1,x_2,\cdots,x_n)$ 是固定的，所以我们只需要比较分子 $P(x_1,x_2,\cdots,x_n|C)P(C)$ 的大小。结合特征独立性假设，就可以将其转化为 $\prod_{i = 1}^{n}P(x_i|C)P(C)$ 进行计算。


## 数据处理流程
1. **分词处理**  
    使用结巴分词实现中文文本切分，英文采用;保留长度≥2的有效词项（过滤"的"、"是"等单字噪声）
2. **停用词过滤**  
   加载哈工大停用词表，剔除"的"、"是"等无意义虚词，保留领域相关核心词汇
3. **标准化处理**  
   - 全角转半角字符
   - 统一小写转换
   - 正则去除HTML标签和特殊符号

```python
def get_words(filename):
    """读取文本并过滤无效字符和长度为1的词"""
    words = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            # 过滤无效字符
            line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
            # 使用jieba.cut()方法对文本切词处理
            line = cut(line)
            # 过滤长度为1的词
            line = filter(lambda word: len(word) > 1, line)
            words.extend(line)
    return words
```


## 特征构建过程
| 特征类型       | 数学表达                          | 实现差异                     |
|---------------|-------------------------------|------------------------------|
| **高频词特征** | $count(xᵢ)=ΣI(xᵢ∈文档)$         | `CountVectorizer`统计词频，选取Top-N高频词 |
|**TF-IDF特征** | $w(xᵢ)=tf(xᵢ)*log(N/df(xᵢ))$  | `TfidfVectorizer`计算逆文档频率，L2归一化处理|

**核心差异**：  
- 高频词侧重绝对词频，适合短文本快速分类
- TF-IDF抑制常见词影响，更强调类别区分性词汇

## 高频词/TF-IDF两种特征模式的切换方法
通过在 extract_features 函数和 train_model 函数中传入 feature_type 参数来实现特征模式的切换。当 feature_type 为 'high_frequency' 时，使用高频词特征提取方法；当 feature_type 为 'tfidf' 时，使用 TF-IDF 特征加权方法。在 predict 函数中，也根据 feature_type 参数选择相应的方法构建未知邮件的特征向量进行预测。

```python
def extract_features(feature_type='high_frequency'):
```
---


## 样本平衡处理
在 train_model 函数中，获取特征矩阵 vector 和标签数组 labels 后，使用 SMOTE 对数据进行过采样，生成新的特征矩阵 X_resampled 和标签数组 y_resampled。
使用过采样后的数据 X_resampled 和 y_resampled 来训练多项式朴素贝叶斯模
```python
 # 使用SMOTE进行过采样
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(vector, labels)
```
---

## 增加模型评估指标
提取特征并获取真实标签，接着对训练集中的每个邮件进行预测，最后使用 classification_report 函数生成包含精度、召回率、F1 值的分类评估报告
```python
def evaluate_model(model, top_words, feature_type='high_frequency'):
    vector = extract_features(feature_type)
    labels = np.array([1] * 127 + [0] * 24)
    predictions = []
    for i in range(151):
        filename = f'邮件_files/{i}.txt'
        pred = predict(filename, model, top_words, feature_type)
        predictions.append(pred)
    report = classification_report(labels, predictions)
    return report
```
---