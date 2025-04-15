import torch
from transformers import BertTokenizer
from tcm_classifier import TCMClassifier

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 初始化tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = TCMClassifier().to(device)
    
    try:
        # 加载模型
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        print("模型加载成功")
    except FileNotFoundError:
        print("错误：找不到模型文件 'best_model.pth'")
        print("请先运行 tcm_classifier.py 训练模型")
        return
    
    # 设置模型为评估模式
    model.eval()
    
    # 示例问题和选项
    question = "中医理论中，气的运动形式主要有"
    options = [
        "升、降、出、入",
        "升、降、聚、散",
        "升、降、开、合",
        "升、降、收、放"
    ]
    
    print(f"\n问题: {question}")
    print("选项:")
    for i, option in enumerate(options):
        print(f"{chr(65+i)}: {option}")
    
    # 将问题和选项组合
    texts = [f"{question} [SEP] {option}" for option in options]
    
    # 对输入进行编码
    encodings = tokenizer(
        texts,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # 将数据移到设备上
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    # 进行预测
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        # 对输出进行 softmax 处理，得到概率分布
        probabilities = torch.softmax(outputs, dim=0)
        # 获取最大概率的索引作为预测结果
        predicted = torch.argmax(probabilities, dim=0).item()
    
    # 打印结果
    print("\n预测结果：")
    for i, (option, prob) in enumerate(zip(options, probabilities.cpu().tolist())):
        print(f"{chr(65+i)}: {option} (概率: {prob:.4f})")
    print(f"\n预测答案: {chr(65+predicted)}")
    
    # 提供交互式预测
    while True:
        try:
            print("\n是否要尝试自定义问题？(y/n)")
            choice = input()
            if choice.lower() != 'y':
                break
                
            print("请输入问题:")
            custom_question = input()
            print("请输入选项A:")
            option_a = input()
            print("请输入选项B:")
            option_b = input()
            print("请输入选项C:")
            option_c = input()
            print("请输入选项D:")
            option_d = input()
            
            custom_options = [option_a, option_b, option_c, option_d]
            
            # 进行自定义预测
            texts = [f"{custom_question} [SEP] {option}" for option in custom_options]
            
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
                probabilities = torch.softmax(outputs, dim=0)
                predicted = torch.argmax(probabilities, dim=0).item()
            
            print("\n预测结果：")
            for i, (option, prob) in enumerate(zip(custom_options, probabilities.cpu().tolist())):
                print(f"{chr(65+i)}: {option} (概率: {prob:.4f})")
            print(f"\n预测答案: {chr(65+predicted)}")
            
        except KeyboardInterrupt:
            print("\n退出预测模式")
            break
        except Exception as e:
            print(f"发生错误: {e}")

if __name__ == '__main__':
    main() 