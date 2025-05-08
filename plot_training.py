import os
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_results_file(file_path):
    epochs = []
    train_losses = []
    learning_rates = []
    dice_scores = []
    
    with open(file_path, 'r') as f:
        content = f.read()
        
    # 使用正则表达式提取数据
    epoch_blocks = re.findall(r'\[epoch: (\d+)\]\n.*?train_loss: ([\d.]+)\n.*?lr: ([\d.]+)\n.*?dice coefficient: ([\d.]+)', content)
    
    for epoch, loss, lr, dice in epoch_blocks:
        epochs.append(int(epoch))
        train_losses.append(float(loss))
        learning_rates.append(float(lr))
        dice_scores.append(float(dice))
    
    return epochs, train_losses, learning_rates, dice_scores

def plot_training_curves(results_file):
    epochs, train_losses, learning_rates, dice_scores = parse_results_file(results_file)
    
    # 创建子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # 绘制训练损失
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss over Time')
    ax1.grid(True)
    ax1.legend()
    
    # 绘制学习率
    ax2.plot(epochs, learning_rates, 'r-', label='Learning Rate')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate over Time')
    ax2.grid(True)
    ax2.legend()
    
    # 绘制Dice系数
    ax3.plot(epochs, dice_scores, 'g-', label='Dice Coefficient')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Dice Coefficient')
    ax3.set_title('Dice Coefficient over Time')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

if __name__ == '__main__':
    # 获取最新的results文件
    results_files = [f for f in os.listdir('.') if f.startswith('results') and f.endswith('.txt')]
    if not results_files:
        print("No results files found!")
        exit()
    
    latest_file = max(results_files, key=os.path.getctime)
    print(f"Plotting training curves from {latest_file}")
    plot_training_curves(latest_file) 