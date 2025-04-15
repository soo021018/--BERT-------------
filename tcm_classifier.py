import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
import json
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

# 设置随机种子，确保结果可重复
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 加载数据
def load_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = []
    options = []
    labels = []
    
    for item in data:
        question = item['question']
        option_a = item['A']
        option_b = item['B']
        option_c = item['C']
        option_d = item['D']
        correct_label = item['label']
        
        # 将问题和选项组合
        questions.append(question)
        options.append([option_a, option_b, option_c, option_d])
        labels.append(ord(correct_label) - ord('A'))  # 将A,B,C,D转换为0,1,2,3
    
    return questions, options, labels

# 定义数据集类
class TCMDataset(Dataset):
    def __init__(self, questions, options, labels, tokenizer, max_length=256):
        self.questions = questions
        self.options = options
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        option_list = self.options[idx]
        label = self.labels[idx]
        
        # 将问题和每个选项组合
        texts = [f"{question} [SEP] {option}" for option in option_list]
        
        # 对每个组合进行编码
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 定义模型
class TCMClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-chinese', dropout_rate=0.3):
        super(TCMClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        # 增加一个隐藏层
        self.hidden = nn.Linear(self.bert.config.hidden_size, 256)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(256, 1)  # 每个选项输出一个分数
        
    def forward(self, input_ids, attention_mask):
        # 获取BERT输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # 使用[CLS]标记的输出
        
        # 通过隐藏层和dropout
        pooled_output = self.dropout(pooled_output)
        hidden_output = self.hidden(pooled_output)
        hidden_output = self.activation(hidden_output)
        hidden_output = self.dropout(hidden_output)
        
        # 最终分类
        logits = self.classifier(hidden_output)
        return logits.squeeze(-1)  # 返回形状为 [batch_size] 的张量

# 准备示例数据
def prepare_sample_data():
    questions = [
        "中医理论中，气的运动形式主要有",
        "以下哪项属于阴阳学说的基本内容",
        "关于五行生克制化的表述，正确的是",
        "以下哪项不属于八纲辨证的内容",
        "中医诊断的\"四诊\"不包括",
        "以下哪项不属于舌诊的内容",
        "以下哪项不属于中医病因",
        "在中医理论中，\"神\"的概念不包括",
        "在五脏的生理功能中，\"藏神\"指的是",
        "以下哪项不属于五行相生的关系",
        "在中医理论中，心的主要功能不包括",
        "以下关于脾的生理功能，错误的是",
        "在中医理论中，肺的主要功能不包括",
        "关于肾的生理功能，以下哪项是错误的",
        "关于肝的生理功能，以下说法错误的是"
    ]
    
    options = [
        ["升、降、出、入", "升、降、聚、散", "升、降、开、合", "升、降、收、放"],
        ["阴阳对立", "阴阳互根", "阴阳消长", "阴阳转化"],
        ["金生水", "火克金", "木生火", "土生金"],
        ["表证与里证", "寒证与热证", "实证与虚证", "痛证与痒证"],
        ["望诊", "闻诊", "问诊", "摸诊"],
        ["舌质", "舌苔", "舌形", "舌声"],
        ["内因", "外因", "不内外因", "食物因素"],
        ["精神意识", "思维活动", "生命活力", "语言能力"],
        ["心", "肺", "脾", "肝"],
        ["木生火", "火生土", "土生金", "水生木"],
        ["主血脉", "主神明", "主汗", "主藏血"],
        ["主运化", "主升清", "主统血", "主藏精"],
        ["主气", "主宣发肃降", "主行水", "主藏血"],
        ["主生长发育", "主水液代谢", "主纳气", "主封藏"],
        ["主疏泄", "主藏血", "主筋", "主情志"]
    ]
    
    # 正确答案索引（从0开始）
    labels = [1, 0, 2, 3, 3, 3, 3, 3, 0, 0, 3, 3, 3, 2, 3]
    
    # 增加数据集大小，数据增强
    augmented_questions = questions * 2
    augmented_options = options * 2
    augmented_labels = labels * 2
    
    return augmented_questions, augmented_options, augmented_labels

def predict(model, tokenizer, question, options, device):
    """
    使用训练好的模型进行预测
    Args:
        model: 训练好的模型
        tokenizer: BERT分词器
        question: 问题文本
        options: 选项列表 [A, B, C, D]
        device: 设备（CPU/GPU）
    Returns:
        predicted_label: 预测的选项（A/B/C/D）
        probabilities: 每个选项的概率
    """
    model.eval()
    
    # 将问题和每个选项组合
    texts = [f"{question} [SEP] {option}" for option in options]
    
    # 对每个组合进行编码
    encodings = tokenizer(
        texts,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        # 输出形状为 [4]，即每个选项的分数
        probabilities = torch.softmax(outputs, dim=0)  # 使用维度0而不是1
        predicted = torch.argmax(probabilities, dim=0)
    
    # 将预测结果转换为选项（A/B/C/D）
    predicted_label = chr(ord('A') + predicted.item())
    probabilities = probabilities.cpu().tolist()
    
    return predicted_label, probabilities

def train(model, train_loader, val_loader, optimizer, criterion, device, scheduler=None, num_epochs=20):
    best_val_accuracy = 0
    patience = 6
    patience_counter = 0
    
    # 创建保存模型的目录
    os.makedirs('models', exist_ok=True)
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)  # [batch_size, 4, seq_len]
            attention_mask = batch['attention_mask'].to(device)  # [batch_size, 4, seq_len]
            labels = batch['labels'].to(device)  # [batch_size]
            
            batch_size = input_ids.size(0)  # 获取批次大小
            options_count = input_ids.size(1)  # 选项数量，通常是4
            
            # 重塑输入，保留批次信息
            reshaped_input_ids = input_ids.view(batch_size, options_count, -1)
            reshaped_attention_mask = attention_mask.view(batch_size, options_count, -1)
            
            optimizer.zero_grad()
            
            # 分批处理每个样本的所有选项
            all_outputs = []
            for i in range(batch_size):
                sample_input_ids = reshaped_input_ids[i].view(options_count, -1)
                sample_attention_mask = reshaped_attention_mask[i].view(options_count, -1)
                
                outputs = model(sample_input_ids, sample_attention_mask)
                all_outputs.append(outputs)
            
            # 合并所有样本的输出
            outputs = torch.stack(all_outputs)
            
            # 使用交叉熵损失函数
            loss = criterion(outputs.view(batch_size, -1), labels)
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                batch_size = input_ids.size(0)
                options_count = input_ids.size(1)
                
                # 重塑输入，保留批次信息
                reshaped_input_ids = input_ids.view(batch_size, options_count, -1)
                reshaped_attention_mask = attention_mask.view(batch_size, options_count, -1)
                
                # 分批处理每个样本的所有选项
                all_outputs = []
                for i in range(batch_size):
                    sample_input_ids = reshaped_input_ids[i].view(options_count, -1)
                    sample_attention_mask = reshaped_attention_mask[i].view(options_count, -1)
                    
                    outputs = model(sample_input_ids, sample_attention_mask)
                    all_outputs.append(outputs)
                
                # 合并所有样本的输出
                outputs = torch.stack(all_outputs)
                
                loss = criterion(outputs.view(batch_size, -1), labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.view(batch_size, -1), 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.2f}%')
        
        # 保存最佳模型，基于验证准确率而不是损失
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'新的最佳验证准确率: {best_val_accuracy:.2f}%，保存模型')
            
            # 额外保存当前epoch的模型
            torch.save(model.state_dict(), f'models/model_epoch_{epoch+1}_acc_{val_accuracy:.2f}.pth')
        else:
            patience_counter += 1
            print(f'验证准确率未提高，耐心计数: {patience_counter}/{patience}')
            if patience_counter >= patience:
                print('早停触发，停止训练')
                break

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 准备数据
    questions, options, labels = prepare_sample_data()
    
    # 划分训练集和验证集
    train_questions, val_questions, train_options, val_options, train_labels, val_labels = train_test_split(
        questions, options, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"训练集大小: {len(train_questions)}")
    print(f"验证集大小: {len(val_questions)}")
    
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 创建数据集
    train_dataset = TCMDataset(train_questions, train_options, train_labels, tokenizer)
    val_dataset = TCMDataset(val_questions, val_options, val_labels, tokenizer)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    # 初始化模型
    model = TCMClassifier().to(device)
    
    # 设置优化器和学习率
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # 添加学习率调度器
    num_training_steps = len(train_loader) * 20  # 20个epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_training_steps
    )
    
    # 设置损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 训练模型
    train(model, train_loader, val_loader, optimizer, criterion, device, scheduler, num_epochs=20)

if __name__ == '__main__':
    main() 