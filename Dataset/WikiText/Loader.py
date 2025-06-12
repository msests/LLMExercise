from datasets import load_dataset
import torch
from torch.utils.data import Dataset

def WikiTextDataset():
    """
    创建WikiTextDataset实例的函数
    
    Args:
        ds: WikiText数据集，如果为None则使用默认加载的数据集
        
    Returns:
        WikiTextDatasetClass: 派生自torch.utils.data.Dataset的WikiText数据集实例
    """
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    
    # 默认返回训练集，用户可以根据需要选择其他分割
    return ds['train'], ds['validation'], ds['test']






