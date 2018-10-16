
## 1 循环神经网络(RNN)

为了建模并捕捉序列信息，人们提出了循环神经网络结构。该网络的总体抽象模型如下： 
$$\mathbf{x}_t=F(\mathbf{x}_{t-1},\mathbf{u}_t,θ)$$
其中$\mathbf{x}_t$为状态向量，$\mathbf{u}_t$为输入向量, $t$表示离散的时间，$θ$表示模型所有参数。由于$\mathbf{x}_t$与$\mathbf{x}_{t-1}$有关，所以循环神经网络是一个递推模型，其初始状态$\mathbf{x}_0$需要事先取定，如取零向量。$F(⋅,⋅,⋅)$为一个递推函数，可以有不同的定义方案，决定着循环神经网络的性质和性能。$F(⋅,⋅,⋅)$其典型定义形式如下：
$$\mathbf{x}_t=\mathbf{W}_{rec} σ(\mathbf{x}_{t-1} )+\mathbf{W}_{in}\mathbf{u}_t+\mathbf{b}$$  
其中$\sigma(⋅)$为一个分别作用在向量各个元素上的非线性函数。$\mathbf{W}_{rec}$为循环部分的权重矩阵，$\mathbf{W}_{in}$为输入转换权重矩阵，$\mathbf{b}$为偏置向量。$\mathbf{W}_{rec}$、$\mathbf{W}_{in}$和$\mathbf{b}$为该循环神经网络需要学习的参数，用参数符号$θ$统一表示，并由后文介绍的BPTT算法求得。循环神经网络的总体性能使用一个代价函数$\varepsilon$决定。$\varepsilon$定义为各个时刻代价函数$\varepsilon_t$的和，既$\varepsilon=∑_{(1≤t≤T)}\varepsilon_t$ ，其中 $\varepsilon_t=\mathcal{L}(\mathbf{x}_t)$。
为了训练循环神经网络，我们需要计算梯度。循环神经网络的梯度算法被称为时间反向传播(Backpropagation Through Time,缩写为BPTT)，这是通过将循环神经网络展开成前向网络所得到的。  
![RNN](https://github.com/stikbuf/stikbuf.github.io/blob/master/images/RNN%E4%B8%8E%E6%A2%AF%E5%BA%A6%E7%88%86%E7%82%B8%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1/RNN.png?raw=true)
我们根据代价函数的和式将梯度展开：

$$\frac{∂E}{∂θ}=∑_{1≤t≤T}\frac{∂E_t}{∂θ}$$
$$\frac{∂E_t}{∂θ}=∑_{1≤k≤t}(\frac{∂E_t}{∂\mathbf{x}_t}  \frac{∂\mathbf{x}_t}{∂\mathbf{x}_k}\frac  {∂^+ \mathbf{x}_k}{∂θ}) $$
$$\frac{∂\mathbf{x}_t}{∂\mathbf{x}_k}=∏_{k<i≤t} \frac{∂\mathbf{x}_i}{∂\mathbf{x}_{i-1}}=∏_{k<i≤t} \mathbf{W}_{rec}^T diag(σ'{(\mathbf{x}_{i-1}})) $$

其中$diag(⋅)$表示一个向量$\mathbf{v}\in \mathbb{R}^n$到一个矩阵$\mathbf{A}\in R^{n×n}$的映射，其中$\mathbf{A}$的主对角线元素为$\mathbf{v}$的元素，其余元素为$0$。$\frac{∂^+\mathbf{x}_k}{∂θ}$表示将$\mathbf{x}_{k-1}$视为常数时$\mathbf{x}_k$对$θ$的梯度。直观的说，$\mathbf{x}_{k-1}$“阻断”了$\mathbf{x}_k$对$θ$在$k$时刻以前的梯度的传播。

对于循环神经网络，梯度消失和梯度爆炸是一个严重的问题。顾名思义，梯度爆炸是指梯度的范数随序列的时间成指数级别增长；梯度消失则是指梯度的范数随序列的时间成指数级别缩小。以梯度消失为例，对于循环神经网络中常见的非线性激活函数$\sigma(\cdot)$函数，都满足导数有界条件：
$$\exists γ \in \mathbb{R},‖diag(σ' (\mathbf{x}_k ))‖ \le γ$$
记矩阵$\mathbf{W}_{rec}$的最大特征值的绝对值为$λ_1$。若$λ_1<\frac{1}{\gamma}$,则$\frac{∂\mathbf{x}_{k+1}}{∂\mathbf{x}_k}$的雅可比矩阵$\mathbf{W}^{T}_{rec}  diag(σ' (\mathbf{x}_k ))$的二范数满足：
$$\forall k, ‖\frac{∂\mathbf{x}_{k+1}}{∂\mathbf{x}_k }‖≤‖\mathbf{W}_{rec}^T ‖‖diag(σ' (\mathbf{x}_k ))‖<\frac{1}{γ} γ<1$$
因此
$$\exists η \in R, \forall k, ‖\frac{∂\mathbf{x}_{k+1}}{∂\mathbf{x}_k }‖≤η<1$$
所以
$$\frac{∂\varepsilon_t}{∂\mathbf{x}_t}  \frac{∂\mathbf{x}_t}{∂\mathbf{x}_k}=(\prod_{i=k}^{t-1} \frac{∂\mathbf{x}_{i+1}}{∂\mathbf{x}_i }) \le \frac{∂\varepsilon_t}{∂\mathbf{x}_t} η^{t-k} $$
也就是说，梯度随着时间距离$t-k$呈指数衰减。使用对称的证明方式，我们可以得知循环神经网络存在梯度爆炸问题。从这个证明我们可以得知，没有在较长时间尺度上的梯度快速传递通道是导致梯度消失和梯度爆炸的原因，这为解决这两个问题提供了思路。
为了更好的捕捉长距离上下文信息，缓解梯度消失和梯度爆炸问题，往往需要同时使用多种解决方案。如2.节的梯度裁剪和3.节的长短时记忆网络。  

## 2. 梯度裁剪
在[Goodfellow, Ian, et al. _Deep learning_](https://www.deeplearningbook.org/)中，作者指出，由于循环网络的梯度计算如1.节所示，梯度存在范数非常大或非常小的情况。如下图左所示，梯度范数小的区域对应“平原”，梯度范数大的区域对应“悬崖”。在使用梯度下降法优化代价函数的过程中，如果遇到“悬崖”，则参数将更新到一个代价函数值较大的区域，且需要迭代多次才能恢复到代价函数较小的区域。为此  
![梯度裁剪原理](https://github.com/stikbuf/stikbuf.github.io/blob/master/images/RNN%E4%B8%8E%E6%A2%AF%E5%BA%A6%E7%88%86%E7%82%B8%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1/%E6%A2%AF%E5%BA%A6%E8%A3%81%E5%89%AA%E5%8E%9F%E7%90%86.png?raw=true)
我们可以使用梯度裁剪技术。梯度裁剪的思路为：如果遇到梯度爆炸的情况，就将梯度裁剪到一个比较小的值。使用以下梯度裁剪方法，可以保证优化方向和原梯度方向一致，同时参数优化向量的范数有上界，如上图右所示。其伪代码如下所示:
![梯度裁剪算法](https://github.com/stikbuf/stikbuf.github.io/blob/master/images/RNN%E4%B8%8E%E6%A2%AF%E5%BA%A6%E7%88%86%E7%82%B8%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1/%E6%A2%AF%E5%BA%A6%E8%A3%81%E5%89%AA%E7%AE%97%E6%B3%95.png?raw=true)
这个方法也可视为是一种根据梯度动态调节学习率的方法.

## 3. 长短时记忆网络
在1.节中定义的循环神经网络递推函数$F(⋅,⋅,⋅)$决定了循环神经网络的性能和性质，其中就包括对梯度爆炸和梯度消失的问题的解决能力。本小节介绍的长短时记忆网络(Long short term memory, LSTM) [S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural Computation, 9(8), 1997](https://www.bioinf.jku.at/publications/older/2604.pdf). 这一结构，可以在很大程度上缓解梯度爆炸和梯度消失问题。LSTM现今已在机器翻译领域和序列生成领域得到广泛应用。
LSTM的核心单元被称为细胞$\mathbf{c}$，$\mathbf{c}$编码了到当前时刻为止的输入信息。$\mathbf{c}$的行为和更新收到一系列“门”向量的控制。
![LSTM](https://github.com/stikbuf/stikbuf.github.io/blob/master/images/RNN%E4%B8%8E%E6%A2%AF%E5%BA%A6%E7%88%86%E7%82%B8%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1/LSTM.png?raw=true)
“门”这个名字来自于电路中的门控机制，但这里“门”向量通过和输入向量按元素相乘来控制信号的通过。例如乘以1表示让信号无保留的通过，0表示截断信号，0.5表示信号通过一半。在LSTM中，遗忘门(forget gate, $\mathbf{f}$)负责控制遗忘当前的$\mathbf{c}$向量，输入门(input gate, $\mathbf{i}$)负责控制输入信息到当前$\mathbf{c}$向量的流动，输出门(output gate, $\mathbf{o}$)负责控制$\mathbf{c}$向量向输出的流动。其具体运行方式如下所示

$$ \mathbf{i}_t=σ(\mathbf{W}_{ix} \mathbf{x}_t+\mathbf{W}_{im} \mathbf{m}_{t-1} )$$
$$ \mathbf{f}_t=σ(\mathbf{W}_{fx} \mathbf{x}_t+\mathbf{W}_{fm} \mathbf{m}_{t-1})$$
$$\mathbf{o}_t=σ(\mathbf{W}_{ox} \mathbf{x}_t+\mathbf{W}_{om} \mathbf{m}_{t-1} ) $$   
$$\mathbf{g}_t= tanh (\mathbf{W}_{cx} \mathbf{x}_t+\mathbf{W}_{cm} \mathbf{m}_{t-1} )$$
$$\mathbf{c}_t=\mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t\odot⁡ \mathbf{g}_t$$
$$\mathbf{m}_t=\mathbf{o}_t \odot \mathbf{c}_t$$
$$\mathbf{p}_{t+1}=softmax(\mathbf{m}_t)$$
其中$σ(⋅)$为sigmoid激活函数，$\odot$表示向量按元素相乘，$\mathbf{W}$为可训练的权重矩阵。从$$\mathbf{c}_t=\mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t\odot⁡ \mathbf{g}_t$$中我们注意到，$\mathbf{c}_t$到$\mathbf{c}_{t+1}$之间的更新法则是$\mathbf{c}_t$和$\mathbf{f}_t$的元素按元素相乘并接收一定的输入信息，其中没有连续的矩阵乘法的过程，就基本避免了梯度爆炸和梯度消失问题。这种通过“残差连接”“恒等变换”使得梯度可以快速通过深层神经网络结构的设计在如今的深度学习中已经非常普遍：
![残差](https://github.com/stikbuf/stikbuf.github.io/blob/master/images/RNN%E4%B8%8E%E6%A2%AF%E5%BA%A6%E7%88%86%E7%82%B8%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1/%E6%AE%8B%E5%B7%AE%E5%A5%97%E8%B7%AF.jpg?raw=true)

## \*4. 长时间依赖问题（探讨）
在LSTM网络中，由于
$$ \mathbf{f}_t=σ(\mathbf{W}_{fx} \mathbf{x}_t+\mathbf{W}_{fm} \mathbf{m}_{t-1})$$  
。由sigmoid激活函数的表达式$σ(x)=\frac{1}{1+e^{-x} }$可知，$f_t$的每个元素都小于$1$。考察LSTM的$c_t$更新规律
$$\mathbf{c}_t=\mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t\odot⁡ \mathbf{g}_t$$
，在形式上与一阶数字低通滤波器相似。也就是说，LSTM网络在迭代计算$\mathbf{c}_t$的过程中，较长时间以前的$\mathbf{c}$信息会逐渐消失，最近离散时刻的$\mathbf{i}
_t\odot \mathbf{g}_t$的比例会提高，因此LSTM网络趋向于提取输入序列最近的信息，不具有长时记忆能力。

所以，我认为“LSTM解决了长时间依赖问题”这个说法不准确，具有迷惑性。我认为准确的说法是，LSTM(部分)解决了长时间尺度上的梯度爆炸和梯度消失问题。但LSTM网络趋向于提取输入序列最近的信息，不能充分记忆较长时间之前的信息。在[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)的Figure 3中   
![长时间](https://github.com/stikbuf/stikbuf.github.io/blob/master/images/RNN%E4%B8%8E%E6%A2%AF%E5%BA%A6%E7%88%86%E7%82%B8%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1/%E9%95%BF%E6%97%B6%E9%97%B4%E8%AE%B0%E5%BF%86.png?raw=true)
我们可以观察到基于LSTM的解码器（RNNenc不带注意力机制）在解码的过程中，如果不使用attention，LSTM无法捕捉较长序列的全部信息。如果将来时间充裕，我可以设计一个实验，记录LSTM在处理长序列时$\mathbf{f}_t$的输出，将这些输出连乘，看看这个数值和长序列建模分数的相关性。
最后一节，欢迎拍砖！












