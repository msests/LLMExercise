# BPE多进程优化说明

## 概述

本项目对 `TokenizeBPE.Process` 方法进行了多进程优化，以提高BPE（Byte Pair Encoding）训练的速度。

## 主要改进

### 1. 新增多进程方法

- `ProcessMultiProcess(num_processes=None)`: 多进程版本的Process方法
- `CollectBasicTokensMultiProcess(num_processes=None)`: 多进程版本的CollectBasicTokens方法
- `FindMaxOccurrenceMultiProcess(num_processes=None)`: 多进程版本的FindMaxOccurrence方法
- `MergeIndicesMultiProcess(num_processes=None)`: 多进程版本的MergeIndices方法

### 2. 静态辅助方法

为了支持多进程，添加了以下静态方法：
- `_process_sample_chunk_static()`: 处理样本块的静态方法
- `_find_occurrences_chunk_static()`: 计算频率统计的静态方法
- `_merge_indices_chunk_static()`: 合并索引的静态方法

### 3. 修改的现有方法

- `Merge(use_multiprocess=False, num_processes=None)`: 添加了多进程选项

## 使用方法

### 基本用法

```python
from Tokenize.BPE import TokenizeBPE
from Dataset.WikiText.Loader import WikiTextDataset

# 加载数据集
train_set, val_set, test_set = WikiTextDataset()

# 创建BPE实例
bpe = TokenizeBPE([train_set],
                  option_wrap_sentence=True,
                  option_preserve_tokens=True,
                  preserve_tokens=[".", "?", "!", ",", ":", ";"])

# 使用多进程版本（自动检测CPU核心数）
bpe.ProcessMultiProcess()

# 或者指定进程数
bpe.ProcessMultiProcess(num_processes=4)
```

### 性能测试

运行测试脚本比较单进程和多进程性能：

```bash
python test_multiprocess.py
```

## 性能优化原理

### 1. 数据分块处理

- 将大量数据分成多个块，每个进程处理一个块
- 减少进程间的数据传递开销

### 2. 并行计算

- `CollectBasicTokens`: 并行处理不同的文本样本
- `FindMaxOccurrence`: 并行计算不同部分的频率统计
- `MergeIndices`: 并行处理不同的索引列表

### 3. 结果合并

- 使用进程池（Pool）管理多个进程
- 自动合并各进程的计算结果

## 注意事项

### 1. 内存使用

- 多进程会增加内存使用量
- 建议根据可用内存调整进程数

### 2. 进程数选择

- 默认使用 `mp.cpu_count()` 个进程
- 可以根据实际情况手动指定进程数
- 通常设置为CPU核心数或略少于核心数

### 3. 兼容性

- 保持与原有单进程版本的完全兼容
- 可以通过参数选择使用单进程或多进程版本

## 预期性能提升

在典型的多核CPU环境下，多进程版本可以带来：

- **2-4倍的速度提升**（取决于CPU核心数和数据规模）
- **更好的CPU利用率**
- **更快的BPE训练完成时间**

## 示例输出

```
使用 4 个进程进行BPE训练
Merge 5 tokens in 0.15 seconds
Merge 6 tokens in 0.12 seconds
...
Merge 1400 tokens in 0.08 seconds
```

## 故障排除

### 常见问题

1. **内存不足**: 减少进程数或增加系统内存
2. **进程启动失败**: 检查Python版本和multiprocessing模块
3. **性能提升不明显**: 检查数据规模是否足够大，CPU是否多核

### 调试模式

可以通过设置较小的进程数来测试：

```python
bpe.ProcessMultiProcess(num_processes=2)  # 使用2个进程测试
``` 