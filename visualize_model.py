import torch
import torch.nn as nn
from graphviz import Digraph
import os

def create_unet_architecture_diagram():
    # 创建有向图
    dot = Digraph(comment='UNet Architecture', format='png')
    dot.attr(rankdir='TB', size='12,8', dpi='300')
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue', fontname='Arial', fontsize='14')
    dot.attr('edge', fontname='Arial', fontsize='12')
    
    # 设置节点样式
    input_style = {'shape': 'box', 'style': 'filled', 'fillcolor': 'lightgreen', 'fontname': 'Arial', 'fontsize': '14'}
    output_style = {'shape': 'box', 'style': 'filled', 'fillcolor': 'lightpink', 'fontname': 'Arial', 'fontsize': '14'}
    conv_style = {'shape': 'box', 'style': 'filled', 'fillcolor': 'lightblue', 'fontname': 'Arial', 'fontsize': '14'}
    pool_style = {'shape': 'box', 'style': 'filled', 'fillcolor': 'lightyellow', 'fontname': 'Arial', 'fontsize': '14'}
    
    # 添加输入节点
    dot.node('input', 'Input\n(3, 480, 480)', **input_style)
    
    # 编码器部分
    with dot.subgraph(name='cluster_encoder') as c:
        c.attr(label='Encoder', style='rounded', bgcolor='lightgrey')
        c.node('conv1', 'DoubleConv\n(3→32)', **conv_style)
        c.node('pool1', 'MaxPool2d', **pool_style)
        c.node('conv2', 'DoubleConv\n(32→64)', **conv_style)
        c.node('pool2', 'MaxPool2d', **pool_style)
        c.node('conv3', 'DoubleConv\n(64→128)', **conv_style)
        c.node('pool3', 'MaxPool2d', **pool_style)
        c.node('conv4', 'DoubleConv\n(128→256)', **conv_style)
        c.node('pool4', 'MaxPool2d', **pool_style)
        c.node('conv5', 'DoubleConv\n(256→512)', **conv_style)
    
    # 解码器部分
    with dot.subgraph(name='cluster_decoder') as c:
        c.attr(label='Decoder', style='rounded', bgcolor='lightgrey')
        c.node('up1', 'Up\n(512→256)', **conv_style)
        c.node('up2', 'Up\n(256→128)', **conv_style)
        c.node('up3', 'Up\n(128→64)', **conv_style)
        c.node('up4', 'Up\n(64→32)', **conv_style)
    
    # 输出层
    dot.node('output', 'Output\n(2, 480, 480)', **output_style)
    
    # 添加连接
    # 编码器连接
    dot.edge('input', 'conv1')
    dot.edge('conv1', 'pool1')
    dot.edge('pool1', 'conv2')
    dot.edge('conv2', 'pool2')
    dot.edge('pool2', 'conv3')
    dot.edge('conv3', 'pool3')
    dot.edge('pool3', 'conv4')
    dot.edge('conv4', 'pool4')
    dot.edge('pool4', 'conv5')
    
    # 解码器连接
    dot.edge('conv5', 'up1')
    dot.edge('up1', 'up2')
    dot.edge('up2', 'up3')
    dot.edge('up3', 'up4')
    dot.edge('up4', 'output')
    
    # 跳跃连接
    dot.edge('conv4', 'up1', style='dashed')
    dot.edge('conv3', 'up2', style='dashed')
    dot.edge('conv2', 'up3', style='dashed')
    dot.edge('conv1', 'up4', style='dashed')
    
    # 保存图形
    if not os.path.exists('visualization'):
        os.makedirs('visualization')
    dot.render('visualization/unet_architecture', cleanup=True)
    print("模型架构图已保存到 visualization/unet_architecture.png")

if __name__ == '__main__':
    create_unet_architecture_diagram()
