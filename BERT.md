## BERT模型
* **BERT是啥**

      BERT基本上就是一个训练好的Transformer编码器栈，
  
  ![](https://raw.githubusercontent.com/Elliotter/Transformer/master/pic/bert%20one.png)
  
  1.**BERT BASE:**  
  
      和OpenAI Transformer模型的规模差不多，方便与其进行性能比较, L = 12, H = 768, A = 12,总参数110M
  
  2.**BERT LARGE:**  
  
      一个达到目前多个benchmark的SOTA的巨大的模型 ，BertLarge:L = 24, H = 1024, A = 16,总参数340M 
      
      其中：Bert中层数（即transformer的块数）为 L，隐藏层大小为 H,self-attention头数量为 A
  
  ![](https://raw.githubusercontent.com/Elliotter/Transformer/master/pic/bert%20two.png)
  
  **注意：只是运用Transformer的编码器，解码器并未使用**
  
---  
  
* **BERT的输入**
   
       就像Transformer的编码器一样，BERT以一串单词作为输入，这些单词不断地想编码器栈上层流动。
       
       每一层都要经过自注意力层和前馈网络，然后在将其交给下一个编码器
    
  ![](https://raw.githubusercontent.com/Elliotter/Transformer/master/pic/bert%20input%20one.png)
  
  ![](https://raw.githubusercontent.com/Elliotter/Transformer/master/pic/bert%20input%20two.png)
  
       BERT的输入主要是由次块嵌入、段嵌入、位置嵌入求和组成，具体来说：
       
       (1) 词嵌入
       
       (2) 位置嵌入，最长支持512
       
       (3) 每一个序列的第一个词始终是特殊分类嵌入(CLS)，对应该词块的最后隐藏状态（Transformer的输出）被用做分类任务的输出
       
           聚合标志，对于非分类的任务，通常CLS会被忽略。
           
       (4) 句子被打包成单个序列，我们以两种方式来区分句子，首先用分割符分割句子，第二个是在第一个句子每个词块中添加
       
           句子A的嵌入，在第二个句子中添加句子B的嵌入
           
       (5) 对于单个句子的输入，我们用句子A嵌入

---
	   
  ![](https://raw.githubusercontent.com/Elliotter/Transformer/master/pic/bert%20input%20three.png)
    
* **BERT的输出**

      每个位置对应地输出一个维度为hidden_size(BERT Base中为768)的向量。对于句子分类任务,只关注第一个位置的输出，
      
      也就是[CLS]符号代替的位置。
      
  ![](https://raw.githubusercontent.com/Elliotter/Transformer/master/pic/bert%20output%20one.png)
  
  ![](https://raw.githubusercontent.com/Elliotter/Transformer/master/pic/bert%20output%20two.png)

* **BERT训练与使用**
     
  1.完形填空任务
         
        Mask LM 是指随机的遮蔽一些输入的单词，类似完形填空任务，最后预测被遮蔽的词块，
         
   ![](https://raw.githubusercontent.com/Elliotter/Transformer/master/pic/bert%20train%20one.png)
   
        使用Mask带来两个问题：
        
        一 是Mask词块在微调的时候看不见问题
        
           解决：我们做了一些缓解，生成器并不总是用 MASK 去替换选择的词：
           
           80%的时间用Mask替换选择的词，比如我的狗是毛茸茸的！我的狗是MASK。
           
           10%的时间：用随机词体替换选择的词，比如我的狗是毛茸茸的! 我的狗是苹果。
           
           10%的时间：让选择的词保持不变。比如我的狗是毛茸茸的！我的狗是毛茸茸的！ 
           
        二 在使用MLM之后，每个批次只会有15%词块被预测，为了让模型收敛，**训练时间增加**
   
  2.给定两个句子（A和B）， B可能是接在A后面出现的句子吗
   
   ![](https://raw.githubusercontent.com/Elliotter/Transformer/master/pic/bert%20train%20two.png)
      
         示例：
         
         输入=[CLS] the man went to [MASK] store [ SEP] he bought a gallon [MASK] milk [SEP]
         
         标记=IsNext
         
         输入=[CLS] the man [MASK] to the store [SEP] penguin [MASK] are flight #less birds [SEP]
         
         标记=NotNext
   
  3.BERT在不同任务上的应用
   
   ![](https://raw.githubusercontent.com/Elliotter/Transformer/master/pic/bert%20train%20three.png)
   
        MNLI: 给出一对句子，目标是预测第二个句子和第一个句子相比是蕴含，矛盾，还是中立
        
        QQP Quora: 对是一个二值的单句子分类问题，目标是判断两个在Quora上的问题是否在语义上等价。
        
        QNLI: 问答数据集正例是指那些问题，句子对包含正确的答案，反例是指那些问题，句子对来自于同一段落，但不包含正确的答案。
        
        SST-2: 许多影评中的句子，这些句子中包含着人们对于他们情感的注释。
        
        COLA: 语言可接受性语料库是一个二值的单句子分类问题，它的目标是预测一个英文句子是否语法上是否有问题。
        
        STS-B: 语义文本相似性基准是一个句子对的集合,他们被标记为1到5分，来表示两个句子语义上是否相近。
        
        MRPC: 微软研究院解释语料库包含了从网上新闻中自动抽取的句子对，人工标注句子对是否语义上相等。
        
