import torch
import torch.nn as nn
from graphviz import Digraph
import os

def create_unet_architecture_diagram():
    # 创建有向图
    dot = Digraph(comment='UNet Architecture', format='png', graph_attr={
        'rankdir': 'LR',      # 横向布局
        'compound': 'true',   # 允许跨子图连接
        'size': '12,6',       # 宽度稍小
        'dpi': '300'
    })
    dot.attr('node',
             shape='box',
             style='filled',
             fillcolor='lightblue',
             fontname='Arial',
             fontsize='24')
    dot.attr('edge', fontname='Arial', fontsize='12')

    # 节点样式
    input_style = dict(shape='box', style='filled', fillcolor='lightgreen')
    output_style = dict(shape='box', style='filled', fillcolor='lightpink')
    conv_style = dict(shape='box', style='filled', fillcolor='lightblue')
    pool_style = dict(shape='box', style='filled', fillcolor='lightyellow')

    # --- Input 子图 ---
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='')     # 不显示子图框
        c.node('input', 'Input\n(3,480,480)', **input_style)

    # --- Encoder 子图 ---
    with dot.subgraph(name='cluster_encoder') as c:
        c.attr(label='Encoder')
        c.node('conv1', 'DoubleConv\n(3→32)', **conv_style)
        c.node('pool1', 'MaxPool2d', **pool_style)
        c.node('conv2', 'DoubleConv\n(32→64)', **conv_style)
        c.node('pool2', 'MaxPool2d', **pool_style)

        # 用隐形边确保 Encoder 与 Middle、Decoder 的顶点对齐
        c.edge('conv1', 'conv1_mid_anchor', style='invis', lhead='cluster_middle')
        c.edge('conv1', 'conv1_dec_anchor', style='invis', lhead='cluster_decoder')

    # --- Middle 子图 ---
    with dot.subgraph(name='cluster_middle') as c:
        c.attr(label='Middle')
        # 在 Middle 里先加一个“锚点”节点，供隐形边对齐
        c.node('conv1_mid_anchor', '', width='0', height='0', style='invis')
        c.node('conv3', 'DoubleConv\n(64→128)', **conv_style)
        c.node('pool3', 'MaxPool2d', **pool_style)
        c.node('conv4', 'DoubleConv\n(128→256)', **conv_style)
        c.node('pool4', 'MaxPool2d', **pool_style)
        c.node('conv5', 'DoubleConv\n(256→512)', **conv_style)

        c.edge('conv1_mid_anchor', 'conv3', style='invis')
        # 对 decoder 也加同样的隐形对齐
        c.edge('conv1_mid_anchor', 'conv1_dec_anchor', style='invis', lhead='cluster_decoder')

    # --- Decoder 子图 ---
    with dot.subgraph(name='cluster_decoder') as c:
        c.attr(label='Decoder')
        c.node('conv1_dec_anchor', '', width='0', height='0', style='invis')
        c.node('up1', 'Up\n(512→256)', **conv_style)
        c.node('up2', 'Up\n(256→128)', **conv_style)
        c.node('up3', 'Up\n(128→64)', **conv_style)
        c.node('up4', 'Up\n(64→32)', **conv_style)
        c.node('output', 'Output\n(2,480,480)', **output_style)

        c.edge('conv1_dec_anchor', 'up1', style='invis')
        c.edge('up4', 'output')

    # --- 真正的连接线 ---
    # Encoder 流水
    dot.edge('input', 'conv1')
    dot.edge('conv1', 'pool1')
    dot.edge('pool1', 'conv2')
    dot.edge('conv2', 'pool2')
    dot.edge('pool2', 'conv3', lhead='cluster_middle')  # 进入 Middle

    # Middle 流水
    dot.edge('conv3', 'pool3')
    dot.edge('pool3', 'conv4')
    dot.edge('conv4', 'pool4')
    dot.edge('pool4', 'conv5')

    # Decoder 流水
    dot.edge('conv5', 'up1', ltail='cluster_middle', lhead='cluster_decoder')
    dot.edge('up1', 'up2')
    dot.edge('up2', 'up3')
    dot.edge('up3', 'up4')

    # 跳跃连接（虚线）
    dot.edge('conv4', 'up1', style='dashed', ltail='cluster_middle', lhead='cluster_decoder')
    dot.edge('conv3', 'up2', style='dashed', ltail='cluster_middle', lhead='cluster_decoder')
    dot.edge('conv2', 'up3', style='dashed')
    dot.edge('conv1', 'up4', style='dashed')

    # 图标题
    dot.attr(label='UNet Architecture', labelloc='t', fontsize='30', fontname='Arial Bold')

    # 输出
    os.makedirs('visualization', exist_ok=True)
    dot.render('visualization/unet_architecture_packed', cleanup=True)
    print("已生成紧凑并排布局的 UNet 架构图：visualization/unet_architecture_packed.png")

if __name__ == '__main__':
    create_unet_architecture_diagram()
