import os
import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def parse_results_file(file_path):
    epochs = []
    train_losses = []
    learning_rates = []
    dice_scores = []
    global_correct = []
    mean_ious = []
    ious = []
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 使用正则表达式提取每个epoch的数据块
    epoch_blocks = re.findall(r'\[epoch: (\d+)\]\n.*?train_loss: ([\d.]+)\n.*?lr: ([\d.]+)\n.*?dice coefficient: ([\d.]+)\n.*?global correct: ([\d.]+)\n.*?average row correct: \[(.*?)\]\n.*?IoU: \[(.*?)\]\n.*?mean IoU: ([\d.]+)', content)
    
    for epoch, loss, lr, dice, g_correct, row_correct, iou_str, mean_iou in epoch_blocks:
        epochs.append(int(epoch))
        train_losses.append(float(loss))
        learning_rates.append(float(lr))
        dice_scores.append(float(dice))
        global_correct.append(float(g_correct))
        mean_ious.append(float(mean_iou))
        
        # 解析IoU列表，处理带引号的值
        iou_values = [float(x.strip("'")) for x in iou_str.split(', ')]
        ious.append(iou_values)
    
    return epochs, train_losses, learning_rates, dice_scores, global_correct, mean_ious, ious

def plot_training_curves(results_file):
    epochs, train_losses, learning_rates, dice_scores, global_correct, mean_ious, ious = parse_results_file(results_file)
    
    # 创建子图
    fig = plt.figure(figsize=(15, 20))
    
    # 1. 训练损失
    ax1 = plt.subplot(4, 2, 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss over Time')
    ax1.grid(True)
    ax1.legend()
    
    # 2. 学习率
    ax2 = plt.subplot(4, 2, 2)
    ax2.plot(epochs, learning_rates, 'r-', label='Learning Rate')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate over Time')
    ax2.grid(True)
    ax2.legend()
    
    # 3. Dice系数
    ax3 = plt.subplot(4, 2, 3)
    ax3.plot(epochs, dice_scores, 'g-', label='Dice Coefficient')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Dice Coefficient')
    ax3.set_title('Dice Coefficient over Time')
    ax3.grid(True)
    ax3.legend()
    
    # 4. 全局准确率
    ax4 = plt.subplot(4, 2, 4)
    ax4.plot(epochs, global_correct, 'c-', label='Global Correct')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Global Correct (%)')
    ax4.set_title('Global Correct over Time')
    ax4.grid(True)
    ax4.legend()
    
    # 5. 平均IoU
    ax5 = plt.subplot(4, 2, 5)
    ax5.plot(epochs, mean_ious, 'm-', label='Mean IoU')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Mean IoU (%)')
    ax5.set_title('Mean IoU over Time')
    ax5.grid(True)
    ax5.legend()
    
    # 6. 各类别IoU
    ax6 = plt.subplot(4, 2, 6)
    ious_array = np.array(ious)
    ax6.plot(epochs, ious_array[:, 0], 'y-', label='Background IoU')
    ax6.plot(epochs, ious_array[:, 1], 'k-', label='Foreground IoU')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('IoU (%)')
    ax6.set_title('Class-wise IoU over Time')
    ax6.grid(True)
    ax6.legend()
    
    # 7. 训练损失和Dice系数的组合图
    ax7 = plt.subplot(4, 2, 7)
    ax7_twin = ax7.twinx()
    ax7.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax7_twin.plot(epochs, dice_scores, 'g-', label='Dice Coefficient')
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Loss', color='b')
    ax7_twin.set_ylabel('Dice Coefficient', color='g')
    ax7.set_title('Loss and Dice Coefficient')
    ax7.grid(True)
    lines1, labels1 = ax7.get_legend_handles_labels()
    lines2, labels2 = ax7_twin.get_legend_handles_labels()
    ax7.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # 8. 最终指标总结
    ax8 = plt.subplot(4, 2, 8)
    ax8.axis('off')
    final_metrics = f"""Final Metrics:
    Best Dice Coefficient: {max(dice_scores):.4f}
    Best Global Correct: {max(global_correct):.1f}%
    Best Mean IoU: {max(mean_ious):.1f}%
    Best Background IoU: {max(ious_array[:, 0]):.1f}%
    Best Foreground IoU: {max(ious_array[:, 1]):.1f}%
    Final Loss: {train_losses[-1]:.4f}"""
    ax8.text(0.1, 0.5, final_metrics, fontsize=12, va='center')
    
    plt.tight_layout()
    
    # 保存图像
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'training_curves_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印最佳指标
    print("\n最佳指标:")
    print(f"最佳Dice系数: {max(dice_scores):.4f}")
    print(f"最佳全局准确率: {max(global_correct):.1f}%")
    print(f"最佳平均IoU: {max(mean_ious):.1f}%")
    print(f"最佳背景IoU: {max(ious_array[:, 0]):.1f}%")
    print(f"最佳前景IoU: {max(ious_array[:, 1]):.1f}%")

if __name__ == '__main__':
    # 获取最新的results文件
    results_files = [f for f in os.listdir('.') if f.startswith('results') and f.endswith('.txt')]
    if not results_files:
        print("No results files found!")
        exit()
    
    latest_file = max(results_files, key=os.path.getctime)
    print(f"Plotting training curves from {latest_file}")
    plot_training_curves(latest_file) 