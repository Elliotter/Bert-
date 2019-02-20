# Bidirectional Encode Representation From Transformers
## BERT模型介绍

  论文 | 备注
  --- | ---
  [Attention Is All Your Need](https://arxiv.org/abs/1706.03762) | Google Attention
  [BERT training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) | Bert
  [细讲Attention is all your need](https://cloud.tencent.com/developer/article/1377062) | 阿里云社区
  [一文读懂Attention](https://yq.aliyun.com/articles/342508?utm_content=m_39938) | 阿里云

## Transformer模型
* **总体模型**
   
  **模型主要分为Encoder和Decoder层**
  
  ![](https://github.com/Elliotter/Bert-/blob/master/pic/Model%20total%20one.png)
  
  **注：** 
  
  1.6层的Encoder与Decoder
  
  2.Encoder的最后一层输出作为每一层Decoder的输入
  
  **思考**
  
  1.为什么会考虑Encoder与Decoder这种结构？E-D结构最初是用来加密
  
  2.为什么是最后一层的Encoder作为Decoder的输入
  
  ![](https://github.com/Elliotter/Bert-/blob/master/pic/Model%20total%20two.png)

  
  **Encoder**: Encoder有N=6层，每层包括两个sub-layers:
  
  第一个multi-head self-attention
  
  第二个简单的全连接网络，LayerNorm(x + Sublayer(x))
  
  
  **Decoder**: Encoder有N=6层，每层包括三个sub-layers
  
  第一个是Masked multi-head self-attention
  
  第二个multi-head self-attention
  
  第三个简单的全连接网络，与Encoder一样
 
  **汇总来看几个点：**
  
  1.Self-Attention、Multi-head Attention、Masked Multi-head Attention
  
  2.Position-wise Feed-Forward Network
  
  3.ResNet、Norm
  
  4.Positional Encoding
  
  5.Encoder与Decoder连接处为什么这么连接
  
  6.Decoder中Multi-head Att两处不一样

* **Positional embedding**
  
  Position Embedding 的公式：
  
  ![](https://github.com/Elliotter/Bert-/blob/master/pic/Position%20embedding%20one.jpeg)
  
  其中 pos:单词在句子中的index， i:dimension index，demodel 表示位置向量的维度
  
  pos\i | 1 | 2 | 3 | 4 | 5
  -- | -- | -- | -- | -- | --
  **1** | .. | .. | .. | .. | ..
  **2** | .. | .. | .. | .. | ..
  **3** | .. | .. | .. | .. | ..
  
  将其分别进行sin cos计算
  
    pos\i | 1 | 2 | 3 | 4 | 5
    -- | -- | -- | -- | -- | --
    **1** | sin() | sin() | sin() | sin() | sin()
    **2** | .. | .. | .. | .. | ..
    **3** | .. | .. | .. | .. | ..
  
    pos\i | 1 | 2 | 3 | 4 | 5
    -- | -- | -- | -- | -- | --
    **1** | cos() | cos() | cos() | cos() | cos()
    **2** | .. | .. | .. | .. | ..
    **3** | .. | .. | .. | .. | ..
  
   拼接成10维的position Encoding
   
    pos\i | 1 | 5 | 6 | 10
    -- | -- | -- | -- | -- 
    **1** | sin() | .. | cos() | ..
    **2** | .. | .. | .. | .. 
    **3** | .. | .. | .. | ..
   
   为什么选用sin+cos形式：
   
   ![](https://github.com/Elliotter/Bert-/blob/master/pic/Position%20embedding%20two.jpeg)
   
   位置p+k的向量可以表示成位置p的向量的线性变换，这提供了表达相对位置信息的可能性。
   
   ![](https://github.com/Elliotter/Bert-/blob/master/pic/Position%20embedding%20three.jpeg)

* **Self Attention**

**1.自注意力是什么？**
    
  假设下面的句子是我们想要翻译的句子:
    
　   “The animal didn't cross the street because it was too tired”
  
  这句话中的“it”指代的是什么?是街道还是动物?这对人类来说是一个简单的问题，但对算法来说就不那么简单了。当模型处理每个单词(输入序列中的每个位置)时，自注意力能够捕捉该单词与输入序列中的其他位置上的单词的联系来寻找线索，以帮助更好地编码该单词。
  
  ![](https://github.com/Elliotter/Bert-/blob/master/pic/self%20attention%20one.png)
    
**2.如何计算自注意力？**
    
  1.将词嵌入与3个训练后的矩阵相乘得到一个Query向量、一个Key向量和一个Value向量。
  
  ![](https://github.com/Elliotter/Bert-/blob/master/pic/Model%20total%20two.png)
  
  2.向量 q1, k1 做点乘

  ![](https://github.com/Elliotter/Bert-/blob/master/pic/self%20attention%20three.jpeg)
  
  3.对该得分进行规范、softmax
  
  ![](https://github.com/Elliotter/Bert-/blob/master/pic/self%20attention%20four.jpeg)
  
  4.将带权重的各个value向量加起来产生在这个位置上self-attention层的输出。
  
  ![](https://github.com/Elliotter/Bert-/blob/master/pic/self%20attention%20five.png)
  
  5.整体流程：
   
  ![](https://github.com/Elliotter/Bert-/blob/master/pic/self%20attention%20six.png)
  
  ![](https://github.com/Elliotter/Bert-/blob/master/pic/self%20attention%20seven.png)
  
  6.如何理解？
  
  第一：Q * KT 是不同位置元素相乘的结果
  
  第二：Softmax() * V 是指利用第一步位置之间关系分布权重作用在原始的单词向量上，体现单词之间关系

* **3.Multi-head Attention**

  **1.Multi-head Attention是啥？**
  
  对于多头自注意力，我们有多组Query/Key/Value权重矩阵，每一个都是随机初始化的。然后经过训练，每个权重矩阵被用来将输入向量投射到不同的表示子空间。
  
  ![](https://github.com/Elliotter/Bert-/blob/master/pic/multi-head%20one.png)

  用不同的权重矩阵做8次不同的计算得到8个不同的Z矩阵。
  
  ![](https://github.com/Elliotter/Bert-/blob/master/pic/multi-head%20two.png)

  把8个矩阵拼接起来然后用一个额外的权重矩阵与之相乘
  
  ![](https://github.com/Elliotter/Bert-/blob/master/pic/multi-head%20three.png)
  
  总体上来看
  
  ![](https://github.com/Elliotter/Bert-/blob/master/pic/multi-head%20three.png)
   
  **2.Multi-head Attention作用**
  
  个人理解：如果一次attention是某个角度的挖掘单词之间的潜在关联，那么8个attention是从8个角度观察，有了8个“眼睛”，使得观察更加全面
  
  ![](https://github.com/Elliotter/Bert-/blob/master/pic/multi-head%20six.png)
  
  ![](https://github.com/Elliotter/Bert-/blob/master/pic/multi-head%20seven.png)

* **Resnet**
　　在每个编码器中的每个子层(self-attention, ffnn)在其周围都有一个残差连接，还伴随着一个规范化步骤。
  
  ![](https://github.com/Elliotter/Bert-/blob/master/pic/resnet%20one.png)
  
  ![](https://github.com/Elliotter/Bert-/blob/master/pic/resnet%20two.png)

  ![](https://github.com/Elliotter/Bert-/blob/master/pic/resnet%20three.png)
  
  思考：残差网络的作用？
  
  残差学习解决了随网络深度增加带来的退化问题，残差网络更容易优化，收敛更快，这里不具体讨论

* **Position-wise Feed-Forward Networks（FFN）**

在进行了Attention操作之后，encoder和decoder中的每一层都包含了一个全连接前向网络，对每个position的向量分别进行相同的操作，包括两个线性变换和一个ReLU激活输出：
  
  ![](https://raw.githubusercontent.com/Elliotter/Bidirectional-Encoder-Representation-From-Transformers/master/FFN.png)

## BERT模型
* **模型架构**
  
  BERT是一个多层的双向编码Transformer，在Bert中 层数（即transformer的块数）几位L，隐藏层大小为H,self-attention头数量记为A
  
  (1) BertBase: L = 12, H = 768, A = 12,总参数110M
  (2) BertLarge:L = 24, H = 1024, A = 16,总参数340M
  
  BertBase模型尺寸跟OpenAI GPT模型尺寸一样大，Bert使用双向结构，OpenAI GPT是单向（从左到右）的，前者常被称为Transformer编码器，后者常被称为Trasformer的解码器，由于文本生成。
  
  ![](https://raw.githubusercontent.com/Elliotter/Bidirectional-Encoder-Representation-From-Transformers/master/bert%20structure.png)
  
* **BERT的输入表示

    BERT的输入主要是由次块嵌入、段嵌入、位置嵌入求和组成，具体来说：
    (1) 词嵌入
    (2) 位置嵌入，最长支持512
    (3) 每一个序列的第一个词始终是特殊分类嵌入(CLS)，对应该词块的最后隐藏状态（Transformer的输出）被用做分类任务的输出聚合标志，对于非分类的任务，通常CLS会被忽略。
    (4) 句子被打包成单个序列，我们以两种方式来区分句子，首先用分割符分割句子，第二个是在第一个句子每个词块中添加句子A的嵌入，在第二个句子中添加句子B的嵌入
    (5) 对于单个句子的输入，我们用句子A嵌入
 * **预训练任务**
     我们使用两个新的无监督预测任务来预训练Bert
     (1) Masked LM
         Mask LM 是指随机的遮蔽一些输入的单词，类似完形填空任务，最后预测被遮蔽的词块，但是他带来两个问题：
         一 是Mask词块在微调的时候看不见，我们做了一些缓解，生成器并不总是用 MASK 去替换选择的词：
            80%的时间用Mask替换选择的词，比如我的狗是毛茸茸的！我的狗是MASK。
            10%的时间：用随机词体替换选择的词，比如我的狗是毛茸茸的! 我的狗是苹果。
            10%的时间：让选择的词保持不变。比如我的狗是毛茸茸的！我的狗是毛茸茸的！
            
            Transformer编码器不知道下一个它要预测的单词已经被随机单词替换，因此它会被迫保持每个输入词块的分布式的语境表示。另外，由于随机替换在所有             词块中只有1.5%的几率发生，着不会对模型的语言理解能力产生影响。
         二 在使用MLM之后，每个批次只会有15%词块被预测，为了让模型收敛，训练时间回增加
     （2）下一句的预测
          很多的任务侧重于句子之间的语义理解，我们预训练了一个二值化下一个句子的预测任务，比如选择句子A和B作为预训练样本：B由百分之五十的可能性是A实际的下一句，也有百分之五十的可能性是语料库中的随机句子。
          输入=[CLS] the man went to [MASK] store [ SEP] he bought a gallon [MASK] milk [SEP]
          标记=IsNext
          输入=[CLS] the man [MASK] to the store [SEP] penguin [MASK] are flight #less birds [SEP]
          标记=NotNext
 * **预训练过程
   
 * **微调过程**
   (1) 命名体识别
       在示例数据集中CoNLL命名体识别中，包含200k训练单词，他们被标记成人员、组织、地点、杂项或其他（非命名体）
       为了进行微调，我们最终将隐藏特征表示提供给每个词块i到NER标签集上的分类层。每个单词的预测并不依赖于周围的预测。为了使这个于WordPiece词块兼容，  我们将每个CoNLL词块化单词输入到我们的WordPiece词块化器，并且使用与第一个子标记相关的隐藏状态作为分类器的输入。比如：
       Jim Hen ##son was a puppet ##eer
       对X没有做预测，因为WordPiece词块边界是输入的已知部分，对训练和预测都做了类似的处理。可视化图在图3（d）中呈现。一种事例WordPiece模型被用于NER,而非事例模型被用于其他任务之中。
       
       
            
## 模型分析
## BERT应用场景
* **chinese sentences multiclass classficition**

  [bert-fine-tuning-for-chinese-multiclass-classification](https://github.com/maksna/bert-fine-tuning-for-chinese-multiclass-classification)
 
  label | text
  --- | ---
  0 | 黄蜂vs湖人首发：科比带伤战保罗
  1 | 王杰明：家博会未来将深入民心
  2 | 组图：威尼斯电影节开幕
  
  与其他模型的对比：
  
* **QA TASK**

  [BERT-for-Chinese-Question-Answering](https://github.com/xzp27/BERT-for-Chinese-Question-Answering)
  
* **NER TASK**

  [ner_bert](https://github.com/sberbank-ai/ner-bert)
  
* **Attribution Exaction**

  [BERT-AttributeExtraction](https://github.com/sakuranew/BERT-AttributeExtraction)
  
* **Relation Exaction** 
  
  暂缺
