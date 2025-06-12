#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>
#include <random>
#include <cmath>
#include <chrono>
#include <variant>
#include <algorithm>

// 数据集的简单实现，替代Python的Dataset
struct DataSet {
    std::vector<std::string> texts;
    int num_rows;
    
    DataSet(const std::vector<std::string>& data) : texts(data), num_rows(data.size()) {}
    
    std::string operator[](int index) const {
        if (index < 0 || index >= num_rows) return "";
        return texts[index];
    }
};

// Token类型，可以是字符串或者两个int的pair
using Token = std::variant<std::string, std::pair<int, int>>;

class TokenizeBPE {
private:
    std::vector<DataSet> data_sets;
    std::vector<Token> token_list;
    std::unordered_map<std::string, int> token_index_map;
    
    // 仅在训练时使用
    std::unordered_map<std::string, int> alphabet_map;
    std::vector<std::vector<int>> indices;
    std::unordered_map<std::string, int> occurrences;  // 使用字符串作为key
    std::pair<int, int> most_freq_pair;
    int most_freq;
    int vocab_size;
    int vocab_size_limit;
    int sample_count;
    std::vector<std::string> special_tokens;
    
    std::mt19937 rng;  // 随机数生成器

public:
    TokenizeBPE(const std::vector<DataSet>& datasets) 
        : data_sets(datasets), most_freq_pair(0, 0), most_freq(0), vocab_size(0), 
          vocab_size_limit(4000), sample_count(32000), 
          special_tokens({"<s>", "<pad>", "</s>", "<unk>"}),
          rng(std::chrono::steady_clock::now().time_since_epoch().count()) {
        
        // 初始化特殊tokens
        for (const auto& token : special_tokens) {
            alphabet_map[token] = vocab_size;
            token_list.push_back(token);
            vocab_size++;
        }
    }
    
    // 计算token长度
    int LengthOfToken(const Token& token) const {
        if (std::holds_alternative<std::string>(token)) {
            return 1;
        } else {
            auto pair = std::get<std::pair<int, int>>(token);
            return LengthOfToken(token_list[pair.first]) + LengthOfToken(token_list[pair.second]);
        }
    }
    
    // 获取token字符串
    std::string GetToken(const Token& token) const {
        if (std::holds_alternative<std::string>(token)) {
            return std::get<std::string>(token);
        } else {
            auto pair = std::get<std::pair<int, int>>(token);
            return GetToken(token_list[pair.first]) + GetToken(token_list[pair.second]);
        }
    }
    
    // 打印token列表
    void PrintTokenList() const {
        for (const auto& token : token_list) {
            std::cout << GetToken(token) << std::endl;
        }
    }
    
    // 保存到文件
    void SaveToFile(const std::string& filename) const {
        std::ofstream file(filename, std::ios::out);
        if (!file.is_open()) {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return;
        }
        
        for (const auto& token : token_list) {
            file << GetToken(token) << std::endl;
        }
        file.close();
    }
    
    // 从文件加载
    void LoadFromFile(const std::string& filename) {
        token_list.clear();
        token_index_map.clear();
        
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\n') {
                line.pop_back();  // 移除换行符
            }
            token_list.push_back(line);
            token_index_map[line] = token_list.size() - 1;
        }
        file.close();
    }
    
    // 分词
    std::vector<int> Tokenize(const std::string& text) const {
        std::vector<int> indices;
        std::string token = "";
        
        for (char c : text) {
            std::string temp_token = token + c;
            auto it = token_index_map.find(temp_token);
            if (it != token_index_map.end()) {
                token = temp_token;
                continue;
            } else {
                auto token_it = token_index_map.find(token);
                if (token_it != token_index_map.end()) {
                    indices.push_back(token_it->second);
                }
                token = std::string(1, c);
            }
        }
        
        auto token_it = token_index_map.find(token);
        if (token_it != token_index_map.end()) {
            indices.push_back(token_it->second);
        }
        
        return indices;
    }
    
    // 合并token
    void Merge() {
        int length = LengthOfToken(std::make_pair(most_freq_pair.first, most_freq_pair.second));
        std::cout << "合并 (" << most_freq_pair.first << ", " << most_freq_pair.second 
                  << ") 长度为 " << length << std::endl;
        
        token_list.push_back(most_freq_pair);
        vocab_size++;
        MergeIndices();
    }
    
    // 合并索引
    void MergeIndices() {
        for (auto& sequence : indices) {
            std::vector<int> new_indices;
            int j = 0;
            while (j < sequence.size()) {
                if (j < sequence.size() - 1 && 
                    sequence[j] == most_freq_pair.first &&
                    sequence[j + 1] == most_freq_pair.second) {
                    // 合并这对token
                    new_indices.push_back(vocab_size - 1);  // 使用新的合并token ID
                    j += 2;  // 跳过下一个token
                } else {
                    new_indices.push_back(sequence[j]);
                    j++;
                }
            }
            sequence = new_indices;
        }
    }
    
    // 收集基础tokens
    void CollectBasicTokens() {
        if (data_sets.empty()) return;
        
        const DataSet& data_set = data_sets[0];
        int sample_total = data_set.num_rows - data_set.num_rows % sample_count;
        int sample_range = sample_total / sample_count;
        
        std::uniform_int_distribution<int> dist(0, sample_range - 1);
        
        for (int i = 0; i < sample_count; i++) {
            int lower = i * sample_range;
            int sample_index = -1;
            
            while (sample_index == -1 || data_set[sample_index].empty()) {
                sample_index = lower + dist(rng);
            }
            
            std::string text = data_set[sample_index];
            std::vector<int> sequence_indices;
            
            for (char c : text) {
                std::string char_str(1, c);
                if (alphabet_map.find(char_str) == alphabet_map.end()) {
                    alphabet_map[char_str] = vocab_size;
                    token_list.push_back(char_str);
                    vocab_size++;
                }
                sequence_indices.push_back(alphabet_map[char_str]);
            }
            indices.push_back(sequence_indices);
        }
    }
    
    // 将pair转换为字符串key
    std::string PairToString(const std::pair<int, int>& p) const {
        return std::to_string(p.first) + "," + std::to_string(p.second);
    }
    
    // 找到最大出现次数
    void FindMaxOccurrence() {
        occurrences.clear();
        most_freq = 0;
        most_freq_pair = std::make_pair(0, 0);
        
        for (const auto& sequence : indices) {
            int length = sequence.size();
            for (int j = 0; j < length - 1; j++) {
                std::pair<int, int> token_pair = std::make_pair(sequence[j], sequence[j + 1]);
                std::string key = PairToString(token_pair);
                
                occurrences[key]++;
                int current_count = occurrences[key];
                
                if (current_count > most_freq) {
                    most_freq = current_count;
                    most_freq_pair = token_pair;
                }
            }
        }
    }
    
    // 主处理流程
    void Process() {
        CollectBasicTokens();
        
        while (vocab_size < vocab_size_limit) {
            auto start_time = std::chrono::steady_clock::now();
            FindMaxOccurrence();
            Merge();
            auto end_time = std::chrono::steady_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "合并 " << vocab_size << " 个tokens，耗时 " 
                      << duration.count() / 1000.0 << " 秒" << std::endl;
        }
    }
    
    // 获取词汇表大小
    int GetVocabSize() const {
        return vocab_size;
    }
    
    // 设置词汇表大小限制
    void SetVocabSizeLimit(int limit) {
        vocab_size_limit = limit;
    }
    
    // 设置样本数量
    void SetSampleCount(int count) {
        sample_count = count;
    }
};

// 使用示例的main函数
int main() {
    // 创建一些示例数据
    std::vector<std::string> sample_texts = {
        "Hello world, this is a test.",
        "BPE tokenization is useful for NLP.",
        "This is another example text.",
        "Machine learning and deep learning are interesting topics."
    };
    
    DataSet dataset(sample_texts);
    std::vector<DataSet> datasets = {dataset};
    
    // 创建BPE tokenizer
    TokenizeBPE bpe(datasets);
    bpe.SetVocabSizeLimit(100);  // 设置较小的词汇表用于测试
    bpe.SetSampleCount(4);       // 使用所有样本
    
    std::cout << "开始BPE训练..." << std::endl;
    bpe.Process();
    
    std::cout << "\n最终词汇表大小: " << bpe.GetVocabSize() << std::endl;
    
    // 保存词汇表
    bpe.SaveToFile("vocab.txt");
    std::cout << "词汇表已保存到 vocab.txt" << std::endl;
    
    // 测试分词
    std::string test_text = "Hello world";
    auto tokens = bpe.Tokenize(test_text);
    std::cout << "\n分词结果 '" << test_text << "': ";
    for (int token : tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;
    
    return 0;
} 