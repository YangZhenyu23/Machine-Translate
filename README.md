# Machine-Translate

dataset.py下载了hugging face上一个3000条的数据集，对其进行分词，训练集、测试集划分。


augumentation.py 对原始数据集进行的NLP的数据增强。


model.py 使用了pytorch中无预训练参数的transformer架构。默认初始参数设置有所改变。


train.py 是常见的先练设置


evaluation 包含了10个样例的测试，以及一个可交互模块。
