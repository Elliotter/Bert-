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
  
  ![](https://github.com/Elliotter/Bert-/blob/master/pic/self%20attention%20one.png)
  
   这句话中的“it”指代的是什么?是街道还是动物?这对人类来说是一个简单的问题，但对算法来说就不那么简单了。当模型处理每个单词(输入序列中的每个位置)
   时，自注意力能够捕捉该单词与输入序列中的其他位置上的单词的联系来寻找线索，以帮助更好地编码该单词。
    
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

* **Multi-head Attention**

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

* **Masked Multi-head Attention**
  
  masked存在与Encoder与Decoder的协同工作有关，详见Decoder
  
* **Decoder**
  
  一、将上面编码器的输出转换成一组向量K和V
  
  ![](https://github.com/Elliotter/Bert-/blob/master/pic/decoder%20one.gif)
  
  
  二、每一个时间步，解码器会输出翻译后的一个单词。

  ![](https://github.com/Elliotter/Bert-/blob/master/pic/decoder%20two.gif)

  思考：
  1.为什么将输出单词要经过Masked multi-head attention而不直接与Encode模块一起输入？
  
  multi-head attention把原先word embedding矩阵通过引入注意力机制，转换成一个另外一个矩阵，在这个矩阵中，每个单词之间都是有不同的权重关系的，或者说单词之间的相互影响也考虑了进来，往深层次说就是关注了语义，使用Mask后，单词Q * K的权重（注意力）分布只会和之前一个单词有关，这样设计目的是为了让模型在适应在不知道未来信息前提下，根据现有信息做出判断
  
 ![](https://github.com/Elliotter/Bert-/blob/master/pic/Mask%20one.jpg)
  
  2.如何理解将输出编码转换成K V编码一起输入进Decoder
  
  Encoder的输出是包含着当前输入语句位置、语义等很多信息的映射关系，我们把输入与输出通过这个映射关系关联起来，前一个单词通过这层映射关系来预测下一个单词，与Rnn中的状态很相似。
  
  疑问：
  
  关于模型Decoder output作为输入部分细节具体细节部分仍然不够清楚，训练时候是结果语句单个单词一个个叠加输入还是整段word embedding作为输入？经过masked中细节呢？

* **Resnet**

　在每个编码器中的每个子层(self-attention, ffnn)在其周围都有一个残差连接，还伴随着一个规范化步骤。
  
  ![](https://github.com/Elliotter/Bert-/blob/master/pic/resnet%20one.png)
  
  ![](https://github.com/Elliotter/Bert-/blob/master/pic/resnet%20two.png)

  ![](https://github.com/Elliotter/Bert-/blob/master/pic/resnet%20three.png)
  
  思考：残差网络的作用？
  
  残差学习解决了随网络深度增加带来的退化问题，残差网络更容易优化，收敛更快，这里不具体讨论

* **Position-wise Feed-Forward Networks（FFN）**
  
  ![](https://github.com/Elliotter/Bert-/blob/master/pic/FFN%20one.png)
  
  疑问：FFN的作用是啥？详细的网络连接示例？
  
## 总结

   以上是Transformer的基本架构，基于Transformer有很多的变种，比如Bert，这些会单独列开介绍
   
## 参考

  论文 | 备注
  --- | ---
  [Attention Is All Your Need](https://arxiv.org/abs/1706.03762) | Google Attention
  [BERT training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) | Bert
  [细讲Attention is all your need](https://cloud.tencent.com/developer/article/1377062) | 阿里云社区
  [一文读懂Attention](https://yq.aliyun.com/articles/342508?utm_content=m_39938) | 阿里云
  [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Jay Alammar的博客
