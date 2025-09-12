"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ 
    自定义的层归一化(Layer Normalization)模块，支持可选的偏置项。
    归一化有助于稳定训练过程，加速模型收敛。
    
    LayerNorm对于每个样本的特征维度进行归一化，计算公式：
    LN(x) = (x - mean) / sqrt(var + eps) * weight + bias
    
    PyTorch的标准LayerNorm不支持bias=False，所以这里自定义实现
    """

    def __init__(self, ndim, bias):
        """
        初始化层归一化模块
        
        参数:
            ndim: 特征维度，即需要归一化的维度大小
            bias: 是否使用偏置项，如果为False则不添加偏置（可以提升性能和速度）
        """
        super().__init__()
        # 可学习的缩放参数，初始化为全1向量
        # 允许网络学习每个特征维度的合适缩放
        self.weight = nn.Parameter(torch.ones(ndim))
        # 可学习的偏移参数，初始化为全0向量（可选）
        # 只有bias=True时才会创建，允许网络学习每个特征维度的偏移
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        """
        前向传播函数
        
        参数:
            input: 输入张量，形状为(..., ndim)
        
        返回:
            归一化后的张量，形状与输入相同
        """
        # 使用PyTorch内置的layer_norm函数，eps=1e-5用于数值稳定性
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """
    多头因果自注意力机制(Causal Self-Attention)模块
    这是Transformer架构的核心组件，允许模型在处理序列时关注序列中不同位置的信息
    
    因果性(Causal)意味着注意力只能看到当前位置及之前的信息，不能看到未来的信息
    这对于自回归语言模型至关重要，确保生成文本时不会"偷看"未来的词
    
    主要特点：
    1. 多头注意力：将特征分割成多个头，允许模型在不同子空间中学习不同的关系
    2. 因果掩码：确保位置i只能关注位置≤i的信息
    3. 支持Flash Attention优化（PyTorch>=2.0）
    """

    def __init__(self, config):
        """
        初始化因果自注意力模块
        
        参数:
            config: 模型配置对象，包含以下关键参数：
                - n_embd: 嵌入维度（embedding dimension）
                - n_head: 注意力头数（number of attention heads）
                - block_size: 最大序列长度（block size for causal mask）
                - dropout: dropout概率，用于正则化
                - bias: 是否在线性层中使用偏置项
        """
        super().__init__()
        # 确保嵌入维度能被头数整除，因为要在多个头之间平均分配特征
        assert config.n_embd % config.n_head == 0
        
        # 线性变换层：将输入映射为查询(Q)、键(K)、值(V)张量
        # 使用单个线性层同时生成QKV，然后分割，比三个独立的线性层更高效
        # 输出维度为3 * n_embd，因为要同时生成Q、K、V三个张量
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # 输出投影层：将多头注意力的输出重新映射到嵌入维度
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # 正则化层：防止过拟合
        self.attn_dropout = nn.Dropout(config.dropout)  # 注意力权重上的dropout
        self.resid_dropout = nn.Dropout(config.dropout)  # 残差连接上的dropout
        
        # 保存配置参数以供后续使用
        self.n_head = config.n_head  # 注意力头数
        self.n_embd = config.n_embd  # 嵌入维度
        self.dropout = config.dropout  # dropout概率
        
        # 检查是否支持Flash Attention（PyTorch>=2.0的优化实现）
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 创建因果掩码（下三角矩阵），确保位置i只能关注位置≤i的信息
            # 形状为 (1, 1, block_size, block_size)，支持广播到所有批次和头
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        """
        前向传播函数 - 执行多头因果自注意力计算
        
        参数:
            x: 输入张量，形状为 (B, T, C)
               B = batch size（批次大小）
               T = sequence length（序列长度）
               C = embedding dimensionality（嵌入维度，即n_embd）
        
        返回:
            注意力输出张量，形状为 (B, T, C)
        """
        B, T, C = x.size()  # 解构输入张量的维度

        # 计算Q、K、V：首先通过线性变换，然后将结果分割成三个部分
        # 这一步同时对所有注意力头进行计算，利用批处理提高效率
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        
        # 重塑Q、K、V张量以分离出头维度，并调整维度顺序以支持并行计算
        head_size = C // self.n_head  # 每个头的维度
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)  
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)

        # 执行注意力计算 - 两种实现方式：Flash Attention（高效）或标准实现
        if self.flash:
            # Flash Attention：使用优化的CUDA内核，内存效率更高
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # 标准注意力实现：手动计算（兼容性更好但效率较低）
            # 计算注意力分数：Q @ K^T / sqrt(head_size)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            
            # 应用因果掩码：确保每个位置只能关注之前的位置
            # 将掩码位置设为-inf，这样在softmax后权重就接近于0
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            
            # 对注意力分数进行softmax归一化，获得注意力权重
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)  # 应用dropout防止过拟合
            
            # 应用注意力权重到V：权重矩阵乘以值矩阵
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # 重新组装多头输出：将多个头的输出拼接起来
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        # 输出投影和dropout
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """
    多层感知机(MLP)模块 - Transformer中的前馈神经网络部分
    
    这是一个两层的全连接网络，通常被称为位置前馈网络(Position-wise Feed-Forward Network)
    MLP对每个位置的表示进行独立的非线性变换，增强模型的表达能力
    
    架构：输入 -> 线性层1 -> GELU激活 -> 线性层2 -> Dropout -> 输出
    维度变换：n_embd -> 4*n_embd -> n_embd
    """

    def __init__(self, config):
        """
        初始化MLP模块
        
        参数:
            config: 模型配置对象，包含以下关键参数：
                - n_embd: 输入和输出维度（嵌入维度）
                - dropout: dropout概率，用于正则化
                - bias: 是否在线性层中使用偏置项
        """
        super().__init__()
        # 第一层线性变换：将输入维度扩展4倍
        # 这样做可以增加模型的容量，让网络学习更复杂的表示
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        
        # GELU激活函数：Gaussian Error Linear Unit
        # 相比ReLU更平滑，有助于稳定训练，是GPT系列的首选激活函数
        self.gelu = nn.GELU()
        
        # 第二层线性变换：将维度压缩回原始的嵌入维度
        # 这样保证输入输出的维度一致，便于残差连接
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        
        # Dropout层：防止过拟合，随机丢弃一些神经元激活
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        前向传播函数 - 执行前馈网络计算
        
        参数:
            x: 输入张量，形状为 (B, T, C)
               B = batch size（批次大小）
               T = sequence length（序列长度）  
               C = embedding dimensionality（嵌入维度，即n_embd）
        
        返回:
            变换后的输出张量，形状为 (B, T, C)
        """
        # 第一层线性变换：扩展维度
        x = self.c_fc(x)
        
        # GELU激活：引入非线性
        x = self.gelu(x)
        
        # 第二层线性变换：压缩维度回原始大小
        x = self.c_proj(x)
        
        # Dropout：应用正则化
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """
    Transformer块(Block) - GPT模型的基本构建单元
    
    这是Transformer架构的核心组件，结合了自注意力机制和前馈神经网络
    每个块包含两个主要子层，都采用残差连接和预归一化(Pre-LayerNorm)结构：
    1. 多头自注意力子层：处理序列依赖关系
    2. 前馈神经网络子层：进行非线性变换
    
    架构：输入 -> LayerNorm1 -> Attention -> +残差 -> LayerNorm2 -> MLP -> +残差 -> 输出
    """

    def __init__(self, config):
        """
        初始化Transformer块
        
        参数:
            config: 模型配置对象，包含以下关键参数：
                - n_embd: 嵌入维度
                - bias: 是否在LayerNorm中使用偏置项
                - dropout: dropout概率
                - n_head, block_size等其他注意力参数
        """
        super().__init__()
        # 第一个层归一化：用于注意力子层的输入归一化
        # 采用Pre-LayerNorm结构，在子层输入前进行归一化
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        
        # 多头因果自注意力机制
        # 负责处理序列中不同位置之间的关系和依赖
        self.attn = CausalSelfAttention(config)
        
        # 第二个层归一化：用于MLP子层的输入归一化
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        
        # 前馈神经网络（多层感知机）
        # 对每个位置的特征进行非线性变换和特征提取
        self.mlp = MLP(config)

    def forward(self, x):
        """
        前向传播函数 - 执行Transformer块的所有计算
        
        每个子层都包含：归一化 -> 计算 -> 残差连接
        这种结构有助于梯度传播并稳定训练过程
        
        参数:
            x: 输入张量，形状为 (B, T, C)
               B = batch size（批次大小）
               T = sequence length（序列长度）
               C = embedding dimensionality（嵌入维度）
        
        返回:
            经过自注意力和前馈网络处理后的输出张量，形状为 (B, T, C)
        """
        # 第一个子层：多头自注意力
        # Pre-LayerNorm：在输入到注意力前先进行归一化
        # 残差连接：将注意力输出加到原始输入上
        x = x + self.attn(self.ln_1(x))
        
        # 第二个子层：前馈神经网络
        # Pre-LayerNorm：在输入到MLP前先进行归一化
        # 残差连接：将MLP输出加到子层输入上
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    """
    GPT模型配置类 - 使用Python的数据类(@dataclass)来存储模型超参数
    
    这个类包含了构建GPT模型所需的所有关键参数，使用数据类的好处是：
    1. 自动实现__init__方法
    2. 自动实现__repr__方法，便于调试
    3. 提供类型检查和默认值
    
    所有参数都有默认值，这些值对应于GPT-2模型的默认配置
    """
    block_size: int = 1024  # 最大序列长度（context length），模型能够处理的最长文本序列
    vocab_size: int = 50304  # 词汇表大小。GPT-2原始词汇表为50257，向上取整到64的倍数以提高效率
                             # 这样做是为了在现代GPU上获得更好的内存对齐和性能
    n_layer: int = 12        # Transformer层的数量，即堆叠的Block数量
    n_head: int = 12         # 多头注意力中的头数。每个头学习不同类型的关系
    n_embd: int = 768        # 嵌入维度，即模型内部表示的维度大小
    dropout: float = 0.0     # Dropout概率，用于正则化。预训练时通常为0，微调时可设置为0.1+
    bias: bool = True        # 是否在线性层和LayerNorm中使用偏置项。True：保持与GPT-2相同；False：略快且略好

class GPT(nn.Module):
    """
    GPT(Generative Pre-trained Transformer)语言模型
    
    这是完整的GPT实现，包含以下关键组件：
    1. 词嵌入(Word Token Embeddings)：将词汇索引映射到连续的向量空间
    2. 位置嵌入(Position Embeddings)：为序列中的每个位置添加位置信息
    3. Transformer块堆：多个Transformer块，每个包含注意力和MLP子层
    4. 输出层：将隐藏状态映射回词汇表以获得logits
    
    特点：
    - 支持预训练和微调两种模式
    - 权重绑定：Token Embedding和输出层的权重共享以减少参数量
    - 自定义初始化：遵循GPT-2论文的设置
    """

    def __init__(self, config):
        """
        初始化GPT模型
        
        参数验证：
            config: 模型配置对象，必须包含所有必要参数
        
        异常:
            AssertError: 如果vocab_size或block_size为None
        """
        super().__init__()
        # 验证必要参数不为None
        assert config.vocab_size is not None, "Vocabulary size cannot be None"
        assert config.block_size is not None, "Block size cannot be None"
        self.config = config  # 保存配置对象以供后续使用

        # Transformer核心组件字典
        # ModuleDict比普通的dict更适合PyTorch，可以正确处理参数注册
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # 词token嵌入矩阵
            wpe = nn.Embedding(config.block_size, config.n_embd), # 位置嵌入矩阵
            drop = nn.Dropout(config.dropout),                    # Dropout层
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # Transformer块堆
            ln_f = LayerNorm(config.n_embd, bias=config.bias),    # 最终的层归一化
        ))
        # 语言模型头部：将隐藏状态映射到词汇表的logits
        # bias=False可以减少参数并提高训练稳定性
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 权重绑定(Weight Tying)：共享token嵌入和输出层的权重
        # 理论依据：输入和输出空间共享相同的语义结构
        # 好处：1)减少参数量 2)提高训练效率 3)可能提高性能
        # 注意：使用torch.compile()时可能会产生警告，但目前看是良性的
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # 初始化所有权重参数
        self.apply(self._init_weights)
        
        # 特殊初始化：对残差连接的投影权重进行缩放初始化
        # 遵循GPT-2论文的设置：std = 0.02/sqrt(2*n_layer)
        # 这有助于保持残差连接的方差稳定
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # 报告模型参数总数，便于监控模型规模
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        计算模型的参数数量，可选是否包含嵌入层参数
        
        这个函数很有用，因为：
        1. 帮助了解模型规模（比如124M参数的GPT-2）
        2. 监控训练时的内存使用
        3. 比较不同配置的模型
        
        参数:
            non_embedding: 布尔值，默认True
                          - True: 只计算非嵌入参数（常用于报告实际模型容量）
                          - False: 计算所有参数
        
        返回:
            参数数量（整数）
        
        特别说明：
        由于token embedding和lm_head共享权重，token embedding参数实际上在最终层
        被使用，所以我们总是包含它们。只减去位置嵌入参数。
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()  # 减去位置嵌入参数数量
        return n_params

    def _init_weights(self, module):
        """
        自定义权重初始化函数
        
        这个函数会被apply()方法调用，遍历所有子模块
        按照GPT-2论文的设置进行初始化，确保训练稳定性
        
        参数:
            module: 要初始化的PyTorch模块
        """
        if isinstance(module, nn.Linear):
            # 线性层权重：正态分布初始化，均值为0，标准差为0.02
            # 这个较小的标准差有助于保持激活值在合理范围内
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                # 偏置项：初始化为零
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # 嵌入层权重：正态分布初始化，均值为0，标准差为0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        GPT模型的前向传播函数
        
        这是模型的核心计算流程，支持训练模式和推理模式：
        -> 训练模式：targets不为None，返回logits和loss
        -> 推理模式：targets为None，只返回最后位置的logits，无loss
        
        参数:
            idx: 输入的token索引，形状为 (B, T)
                 B = batch size，T = sequence length
            targets: 目标token索引（可选），形状为 (B, T)
                    如果提供，会计算交叉熵损失
        
        返回:
            (logits, loss)元组：
                - logits: 输出的logits，训练模式下形状为(B, T, vocab_size)，
                         推理模式下形状为(B, 1, vocab_size)
                - loss: 交叉熵损失，如果targets为None则返回None
        
        异常:
            AssertError: 如果输入序列长度超出模型支持的block_size
        """
        device = idx.device  # 获取输入设备信息（CPU或GPU）
        b, t = idx.size()    # 解构批次大小和序列长度
        
        # 检查序列长度是否在模型支持范围内
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        # 生成位置索引：0, 1, 2, ..., t-1
        # shape: (t)，所有样本共享相同的位置信息
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # ===== GPT核心计算流程 =====
        
        # 1. Token嵌入：将离散的token索引映射到连续的向量空间
        # shape: (B, T) -> (B, T, n_embd)
        tok_emb = self.transformer.wte(idx)
        
        # 2. 位置嵌入：为序列中的每个位置添加位置信息
        # shape: (T) -> (T, n_embd) -> 广播为 (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)
        
        # 3. 融合token和位置信息，应用dropout
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # 4. 通过所有Transformer块进行特征提取
        for block in self.transformer.h:
            x = block(x)  # 每个Block执行注意力和MLP计算
            
        # 5. 最终的层归一化
        x = self.transformer.ln_f(x)

        # ===== 输出层和损失计算 =====
        
        if targets is not None:
            # 训练模式：计算logits和loss
            logits = self.lm_head(x)  # 将所有隐藏状态映射到词汇表
            
            # 交叉熵损失：计算预测和目标的差异
            # view(-1, ...)将B*T展平为一维，便于计算每个位置的损失
            # ignore_index=-1允许过滤掉不需要计算损失的token
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推理模式的优化：只计算最后一个位置的logits
            # 在生成文本时，我们只需要知道下一个token的概率
            # 使用列表[-1]而不是-1是为了保持时间维度
            logits = self.lm_head(x[:, [-1], :])  # shape: (B, 1, vocab_size)
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        """
        模型手术：减少模型的block size（上下文长度）
        
        这个操作很有用，比如：
        1. 加载了GPT-2预训练模型（block size 1024）但想用小一些的block size
        2. 适应不同的硬件限制和应用场景
        3. 减少计算量和内存使用
        
        参数:
            block_size: 新的block size，必须 <= 原始block size
        
        注意:
            - 这是一个“手术式”操作，直接修改模型结构和参数
            - 只改变位置嵌入和因果掩码的相关维度
            - 其他模型参数保持不变
        """
        # 验证新的block size不超过原始大小
        assert block_size <= self.config.block_size, f"New block_size ({block_size}) must be <= original ({self.config.block_size})"
        
        # 更新配置
        self.config.block_size = block_size
        
        # 裁剪位置嵌入：只保留前block_size个位置的嵌入向量
        # 这会减少模型参数，因为位置嵌入矩阵变小了
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        self.transformer.wpe.num_embeddings = block_size  # 更新嵌入层的num_embeddings
        
        # 裁剪注意力层的因果掩码
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                # 只保留前block_size x block_size的子矩阵
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """
        从预训练的GPT-2模型加载权重（类方法）
        
        支持加载OpenAI发布的4个主要GPT-2变体：
        - gpt2: 最小版本，约124M参数
        - gpt2-medium: 中等大小，约350M参数  
        - gpt2-large: 大版本，约774M参数
        - gpt2-xl: 超大版本，约1558M参数
        
        参数:
            model_type: GPT-2模型类型，必须是{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}之一
            override_args: 可选字典，只能覆盖dropout参数
        
        返回:
            加载了预训练权重的GPT模型实例
        
        注意:
            - 需要安装transformers库: pip install transformers
            - 会强制设置vocab_size=50257, block_size=1024, bias=True（与原始GPT-2一致）
            - 主要用途：迁移学习、微调、性能对比
        """
        # 验证模型类型
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}, \
            f"model_type must be one of {{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}}, got {model_type}"
        
        # 处理覆盖参数，确保只覆盖dropout
        override_args = override_args or {}  # 默认为空字典
        assert all(k == 'dropout' for k in override_args), \
            "Only dropout parameter can be overridden in from_pretrained"
        
        from transformers import GPT2LMHeadModel  # 延迟导入，避免不必要的依赖
        print(f"Loading weights from pretrained gpt: {model_type}")

        # 预定义的GPT-2模型配置：层数、头数、嵌入维度
        # 这些配置对应于OpenAI发布的不同规模的GPT-2模型
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),   # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        
        # 强制设置与原始GPT-2兼容的参数
        print("Forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257  # GPT-2词汇表大小
        config_args['block_size'] = 1024   # GPT-2最大序列长度
        config_args['bias'] = True         # GPT-2使用偏差项
        
        # 可选：覆盖dropout参数（通常用于微调或测试）
        if 'dropout' in override_args:
            print(f"Overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        
        # 创建新的GPT实例（随机初始化）
        config = GPTConfig(**config_args)
        model = GPT(config)
        
        # 获取当前模型的状态字典
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # 忽略因果掩码缓冲区
        
        # 加载HuggingFace的GPT-2预训练模型
        print(f"Loading from HuggingFace Hub...")
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        配置和创建AdamW优化器 - 专门优化设置以提高训练效率
        
        采用了权重衰减分组策略，遵循GPT/Transformer论文的最佳实践：
        - 对矩阵乘法权重应用权重衰减（如：linear层权重、嵌入权重）
        - 不对偏置和层归一化参数应用权重衰减（这些是小参数或标准化参数）
        
        参数:
            weight_decay: 权重衰减率，通常设置为0.1
            learning_rate: 学习率，最大学习率峰值
            betas: AdamW的beta参数元组 (beta1, beta2)，通常(0.9, 0.95)
            device_type: 设备类型 ('cuda' 或 'cpu')，影响是否使用fused优化器
        
        返回:
            配置好的AdamW优化器实例
            
        技术细节：
        - 使用fused AdamW（如果可用且设备为CUDA）可提升训练速度
        - 按维度分组参数：2D及以上参数通常对应矩阵权重，应用权重衰减
        - 1D及0D参数通常对应偏置和标准化参数，不应用权重衰减
        """
        # 获取所有需要训练的参数（过滤参数名和参数值）
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        # 过滤掉不需要梯度的参数（如冻结的参数）
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # 创建优化器组：区分需要权重衰减和不需权重衰减的参数
        # 基于参数维度的启发式规则：2D及以上参数（如权重矩阵）需要权重衰减
        # 1D及0D参数（如偏置、LayerNorm参数）不需要权重衰减
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        # 构建优化器参数组
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},    # 应用权重衰减的组
            {'params': nodecay_params, 'weight_decay': 0.0}           # 不应用权重衰减的组
        ]
        
        # 统计参数数量以用于分析和监控
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # 检查是否支持fused AdamW优化器（CUDA专用优化）
        # fused版本在CUDA上能提供更好的性能
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        
        # 创建AdamW优化器：当前最优优化器，结合了Adam和权重衰减的优点
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        估算模型的FLOPs利用率（MFU）- 衡量计算效率的重要指标
        
        MFU表示模型实际达到的FLOPS与硬件理论峰值的比率。
        这对于：
        1. 评估训练/推理效率
        2. 对比不同实现方案
        3. 硬件性能调优
        4. 成本估算
        
        参数:
            fwdbwd_per_iter: 每次迭代的前向+反向传播次数
                             例如：梯度累积步骤数 * micro_batch_size
            dt: 每次迭代的耗时（秒），用于计算吞吐量
            
        返回:
            MFU值（0-1之间），表示相对于A100 GPU bfloat16峰值性能的利用率
            
        参考：PaLM论文附录B (https://arxiv.org/abs/2204.02311)
        """
        # 获取模型参数数量
        N = self.get_num_params()
        cfg = self.config
        
        # 解构关键超参数
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        # L: 层数, H: 头数, Q: 每个头的维度, T: 序列长度
        
        # 估算每个token的FLOPs计算量（基于PaLM论文公式）
        # 6N: 模型前向传播的基线计算 - 主要来自矩阵乘法
        # 12*L*H*Q*T: 注意力机制的额外计算成本
        flops_per_token = 6*N + 12*L*H*Q*T
        
        # 每个序列位置的FLOPs（乘以序列长度）
        flops_per_fwdbwd = flops_per_token * T
        
        # 每次迭代的总FLOPs（乘以每次迭代的传播次数）
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        
        # 计算实际达到的FLOPS性能（每秒浮点运算次数）
        flops_achieved = flops_per_iter * (1.0/dt)  # per second
        
        # A100 GPU的bfloat16理论峰值性能：312 TFLOPS
        flops_promised = 312e12  # 312 TFLOPS
        
        # 计算MFU：实际性能 / 理论峰值性能
        mfu = flops_achieved / flops_promised
        
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        自回归文本生成函数 - GPT的核心能力
        
        采用教师强制（Teacher Forcing）策略，每次生成一个新token并将其
        追加到输入序列中，继续生成下一个token。这是自回归语言模型的标准生成方式。
        
        参数:
            idx: 输入的token索引序列，形状 (B, T)
                 B = batch size，T = 当前序列长度
            max_new_tokens: 要生成的新token数量上限
            temperature: 温度参数，控制生成的随机性
                        - 1.0: 标准softmax分布
                        - <1.0: 更确定/保守的生成
                        - >1.0: 更随机的生成
            top_k: 可选的top-k采样参数，只保留概率最高的k个token
                  - None: 不进行top-k截断，使用完整分布
                  - int: 只从top-k个最可能的token中采样
        
        返回:
            完整的生成序列，形状 (B, T + max_new_tokens)，在末尾添加了新生成的token
            
        注意:
            - 应该在model.eval()模式下使用以获得确定性的生成
            - 使用@torch.no_grad()装饰器禁用梯度计算，提高生成速度
            - 生成的质量和多样性可以通过temperature和top_k来控制
            
        算法流程：
        1. 检查序列长度并裁剪（如果超出block_size）
        2. 前向传播获得最新位置的logits  
        3. 应用temperature缩放
        4. 可选的top-k截断
        5. 转换为概率分布并采样
        6. 追加采样结果并重复
        """
        # 设置模型为评估模式（禁用dropout等训练时行为）
        self.eval()
        for _ in range(max_new_tokens):
            # 步骤1：处理序列长度限制
            # 如果序列太长，只保留最后的block_size个token
            # 这是典型的滑动窗口机制，允许生成任意长度的文本
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # 步骤2：前向传播获得logits
            # 我们只关心最后一个位置的logits（下一个token的预测）
            logits, _ = self(idx_cond)
            
            # 步骤3：提取最后一个位置的logits并应用temperature缩放
            # temperature控制分布的“扁平”程度：
            # - 低temperature (<1): 更确定，选择高概率token
            # - 高temperature (>1): 更随机，探索更多可能性
            logits = logits[:, -1, :] / temperature  # shape: (B, vocab_size)
            
            # 步骤4：可选的top-k截断
            # 只保留概率最高的k个token，其他的设置为-inf（概率0）
            if top_k is not None:
                # 获取top-k的值和索引
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # 将低于top-k阈值的logits设置为-inf（这样softmax后概率接近0）
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # 步骤5：转换为概率分布并采样
            # 应用softmax从logits获得概率
            probs = F.softmax(logits, dim=-1)
            
            # 从概率分布中采样一个token
            # torch.multinomial：多项式分布采样
            idx_next = torch.multinomial(probs, num_samples=1)  # shape: (B, 1)
            
            # 步骤6：追加采样的token到序列中
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
