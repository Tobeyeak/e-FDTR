#!/usr/bin/env python3
"""
YOLO Metrics 可视化对比脚本
支持多个模型指标的对比展示，自动标注最优值

用法:
    python visualize_metrics.py model1_metrics.txt model2_metrics.txt ...
    或
    python visualize_metrics.py --input-dir /path/to/metrics_dir/
"""

import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def parse_yolo_metrics(file_path: str) -> Dict:
    """解析 yolo_metrics.txt 文件"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 从路径提取模型名称：取父目录名（如 deim_4.2_eval）
    path = Path(file_path)
    model_name = path.parent.name
    
    data = {
        'model_name': model_name,
        'classes': {},
        'overall': {},
        'model_info': {},
        'coco_ap': {},
        'coco_ar': {}
    }
    
    # 解析 YOLO Metrics 表格
    yolo_pattern = r'\|\s+(\w+(?:-\w+)*)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|'
    matches = re.findall(yolo_pattern, content)
    
    for match in matches:
        class_name, precision, recall, f1, map50, map75, map5095 = match
        metrics = {
            'Precision': float(precision),
            'Recall': float(recall),
            'F1-Score': float(f1),
            'mAP50': float(map50),
            'mAP75': float(map75),
            'mAP50-95': float(map5095)
        }
        if class_name == 'all':
            data['overall'] = metrics
        else:
            data['classes'][class_name] = metrics
    
    # 解析 Model Info
    params_match = re.search(r'\|\s*([\d,]+)\s*\|', content)
    if params_match:
        params = params_match.group(1).replace(',', '')
        data['model_info']['Parameters'] = int(params)
    
    inference_match = re.search(r'\|\s+([\d.]+)\s+\|', content)
    if inference_match:
        data['model_info']['Inference_Time'] = float(inference_match.group(1))
    
    fps_match = re.search(r'\|\s+([\d.]+)\s+\|$', content, re.MULTILINE)
    if fps_match:
        data['model_info']['FPS'] = float(fps_match.group(1))
    
    # 解析 COCO AP
    ap_patterns = [
        (r'AP.*IoU=0\.50:0\.95.*area=\s+all.*=\s+([\d.]+)', 'AP_all'),
        (r'AP.*IoU=0\.50.*area=\s+all.*=\s+([\d.]+)', 'AP_50'),
        (r'AP.*IoU=0\.75.*area=\s+all.*=\s+([\d.]+)', 'AP_75'),
        (r'AP.*area=\s+small.*=\s+([\d.]+)', 'AP_small'),
        (r'AP.*area=medium.*=\s+([\d.]+)', 'AP_medium'),
        (r'AP.*area=\s+large.*=\s+([\d.]+)', 'AP_large'),
    ]
    
    for pattern, key in ap_patterns:
        match = re.search(pattern, content)
        if match:
            data['coco_ap'][key] = float(match.group(1))
    
    # 解析 COCO AR
    ar_patterns = [
        (r'AR.*maxDets=\s*1.*=\s+([\d.]+)', 'AR_1'),
        (r'AR.*maxDets=\s*10.*=\s+([\d.]+)', 'AR_10'),
        (r'AR.*maxDets=100.*=\s+([\d.]+)', 'AR_100'),
        (r'AR.*area=\s+small.*=\s+([\d.]+)', 'AR_small'),
        (r'AR.*area=medium.*=\s+([\d.]+)', 'AR_medium'),
        (r'AR.*area=\s+large.*=\s+([\d.]+)', 'AR_large'),
    ]
    
    for pattern, key in ar_patterns:
        match = re.search(pattern, content)
        if match:
            data['coco_ar'][key] = float(match.group(1))
    
    return data


def find_best_values(all_data: List[Dict], metric_keys: List[str]) -> Dict[str, Tuple[float, int]]:
    """找出每个指标的最优值及其索引"""
    best_values = {}
    
    for key in metric_keys:
        values = []
        for data in all_data:
            if key in data['overall']:
                values.append(data['overall'][key])
            elif key in data['coco_ap']:
                values.append(data['coco_ap'][key])
            else:
                values.append(0)
        
        if values:
            max_val = max(values)
            max_idx = values.index(max_val)
            best_values[key] = (max_val, max_idx)
    
    return best_values


def plot_comparison(all_data: List[Dict], output_path: str = None):
    """绘制综合对比图"""
    if not all_data:
        print("没有数据可以绘制")
        return
    
    model_names = [d['model_name'] for d in all_data]
    n_models = len(model_names)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建大画布
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('YOLO Metrics Comparison Dashboard', fontsize=18, fontweight='bold', y=0.98)
    
    # 定义颜色方案
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    
    # 创建GridSpec布局
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    # ========== 1. 总体指标对比 (左上) ==========
    ax1 = fig.add_subplot(gs[0, 0])
    overall_metrics = ['Precision', 'Recall', 'F1-Score', 'mAP50', 'mAP75', 'mAP50-95']
    x = np.arange(len(overall_metrics))
    width = 0.8 / n_models
    
    for i, data in enumerate(all_data):
        values = [data['overall'].get(m, 0) for m in overall_metrics]
        offset = width * (i - (n_models - 1) / 2)
        bars = ax1.bar(x + offset, values, width, label=model_names[i], color=colors[i], alpha=0.8)
        
        # 标注数值
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, rotation=45)
    
    
    ax1.set_ylabel('Score', fontsize=10)
    ax1.set_title('Overall Metrics', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(overall_metrics, rotation=45, ha='right')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # ========== 2. COCO AP 对比 (中上) ==========
    ax2 = fig.add_subplot(gs[0, 1])
    ap_metrics = ['AP_all', 'AP_50', 'AP_75', 'AP_small', 'AP_medium', 'AP_large']
    ap_labels = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']
    x = np.arange(len(ap_labels))
    
    for i, data in enumerate(all_data):
        values = [data['coco_ap'].get(m, 0) for m in ap_metrics]
        offset = width * (i - (n_models - 1) / 2)
        bars = ax2.bar(x + offset, values, width, label=model_names[i], color=colors[i], alpha=0.8)
    
    
    ax2.set_ylabel('AP', fontsize=10)
    ax2.set_title('COCO AP by Area/IoU', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(ap_labels)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_ylim(0, max([max([d['coco_ap'].get(m, 0) for m in ap_metrics]) for d in all_data]) * 1.2)
    ax2.grid(axis='y', alpha=0.3)
    
    # ========== 3. COCO AR 对比 (右上) ==========
    ax3 = fig.add_subplot(gs[0, 2])
    ar_metrics = ['AR_1', 'AR_10', 'AR_100', 'AR_small', 'AR_medium', 'AR_large']
    ar_labels = ['AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
    x = np.arange(len(ar_labels))
    
    for i, data in enumerate(all_data):
        values = [data['coco_ar'].get(m, 0) for m in ar_metrics]
        offset = width * (i - (n_models - 1) / 2)
        bars = ax3.bar(x + offset, values, width, label=model_names[i], color=colors[i], alpha=0.8)
    
    
    ax3.set_ylabel('AR', fontsize=10)
    ax3.set_title('COCO AR by Area/MaxDets', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(ar_labels)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(axis='y', alpha=0.3)
    
    # ========== 4. 各类别 mAP50 对比 (左下) ==========
    ax4 = fig.add_subplot(gs[1, 0])
    all_classes = list(all_data[0]['classes'].keys()) if all_data[0]['classes'] else []
    
    if all_classes:
        x = np.arange(len(all_classes))
        width = 0.8 / n_models
        
        for i, data in enumerate(all_data):
            values = [data['classes'].get(c, {}).get('mAP50', 0) for c in all_classes]
            offset = width * (i - (n_models - 1) / 2)
            ax4.bar(x + offset, values, width, label=model_names[i], color=colors[i], alpha=0.8)
        
        ax4.set_ylabel('mAP50', fontsize=10)
        ax4.set_title('Per-Class mAP50', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(all_classes, rotation=45, ha='right', fontsize=8)
        ax4.legend(loc='upper right', fontsize=8)
        ax4.grid(axis='y', alpha=0.3)
    
    # ========== 5. 模型效率对比 (中下) - 横向柱状图 ==========
    # 提取模型信息
    params = [d['model_info'].get('Parameters', 0) / 1e6 for d in all_data]  # 转换为M
    fps = [d['model_info'].get('FPS', 0) for d in all_data]
    
    ax5 = fig.add_subplot(gs[1, 1])
    
    # 准备横向条形图数据
    y_pos = np.arange(n_models)
    height = 0.35
    
    # 绘制横向条形图：Parameters 在左，FPS 在右（使用双X轴）
    bars1 = ax5.barh(y_pos - height/2, params, height, label='Parameters (M)', color='steelblue', alpha=0.8)
    
    # 创建第二个X轴用于FPS
    ax5_fps = ax5.twiny()
    bars2 = ax5_fps.barh(y_pos + height/2, fps, height, label='FPS', color='coral', alpha=0.8)
    
    # 设置Y轴（模型名称）
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(model_names, fontsize=9)
    ax5.set_ylim(-0.5, n_models - 0.5)
    
    # 设置左侧X轴（Parameters）
    ax5.set_xlabel('Parameters (Million)', fontsize=10, color='steelblue')
    ax5.tick_params(axis='x', labelcolor='steelblue', labelsize=8)
    ax5.grid(axis='x', alpha=0.3)
    
    # 设置右侧X轴（FPS）
    ax5_fps.set_xlabel('FPS (Frames Per Second)', fontsize=10, color='coral')
    ax5_fps.tick_params(axis='x', labelcolor='coral', labelsize=8)
    
    # 设置标题
    ax5.set_title('Model Efficiency', fontsize=12, fontweight='bold', pad=10)
    
    # 合并图例
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_fps.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=8)
    
    # ========== 6. 综合评分雷达图 (右下) ==========
    ax6 = fig.add_subplot(gs[1, 2], projection='polar')
    
    # 选择关键指标做雷达图
    radar_metrics = ['mAP50', 'mAP50-95', 'AP_small', 'AP_medium', 'AP_large', 'FPS_norm']
    radar_labels = ['mAP50', 'mAP50-95', 'AP-small', 'AP-med', 'AP-large', 'Speed']
    
    # 归一化FPS到0-1范围
    if fps:
        max_fps = max(fps)
        fps_norm = [f / max_fps if max_fps > 0 else 0 for f in fps]
    else:
        fps_norm = [0] * n_models
    
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    for i, data in enumerate(all_data):
        values = [
            data['overall'].get('mAP50', 0),
            data['overall'].get('mAP50-95', 0),
            data['coco_ap'].get('AP_small', 0),
            data['coco_ap'].get('AP_medium', 0),
            data['coco_ap'].get('AP_large', 0),
            fps_norm[i]
        ]
        values += values[:1]  # 闭合
        
        ax6.plot(angles, values, 'o-', linewidth=2, label=model_names[i], color=colors[i])
        ax6.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(radar_labels, fontsize=9)
    ax6.set_ylim(0, 1)
    ax6.set_title('Comprehensive Score', fontsize=12, fontweight='bold', pad=20)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {output_path}")
    else:
        plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
        print("图表已保存到: metrics_comparison.png")
    
    plt.show()


def print_summary(all_data: List[Dict]):
    """打印文字摘要"""
    print("\n" + "="*80)
    print("模型性能对比摘要".center(80))
    print("="*80)
    
    model_names = [d['model_name'] for d in all_data]
    
    # 找出每个指标的最优模型
    metrics_to_compare = [
        ('mAP50', 'overall'),
        ('mAP50-95', 'overall'),
        ('AP_small', 'coco_ap'),
        ('AP_medium', 'coco_ap'),
        ('AP_large', 'coco_ap'),
        ('FPS', 'model_info')
    ]
    
    print(f"\n{'指标':<20} {'最优模型':<25} {'数值':<15} {'相比最差':<15}")
    print("-"*80)
    
    for metric, category in metrics_to_compare:
        values = []
        for data in all_data:
            if category == 'overall':
                val = data['overall'].get(metric, 0)
            elif category == 'coco_ap':
                val = data['coco_ap'].get(metric, 0)
            else:
                val = data['model_info'].get(metric, 0)
            values.append(val)
        
        if values:
            max_val = max(values)
            min_val = min(values)
            max_idx = values.index(max_val)
            improvement = ((max_val - min_val) / min_val * 100) if min_val > 0 else 0
            
            print(f"{metric:<20} {model_names[max_idx]:<25} {max_val:<15.4f} +{improvement:.1f}%")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='YOLO Metrics 可视化对比工具')
    parser.add_argument('files', nargs='*', help='yolo_metrics.txt 文件路径')
    parser.add_argument('--input-dir', '-d', type=str, help='包含 metrics 文件的目录')
    parser.add_argument('--output', '-o', type=str, help='输出图片路径')
    
    args = parser.parse_args()
    
    # 收集文件
    metric_files = []
    
    if args.files:
        metric_files.extend(args.files)
    
    if args.input_dir:
        dir_path = Path(args.input_dir)
        metric_files.extend(list(dir_path.glob('*_metrics.txt')))
        metric_files.extend(list(dir_path.glob('yolo_metrics.txt')))
    
    if not metric_files:
        # 尝试使用默认路径
        default_dir = Path('/tanjunnan/DETR/DETR/outputs')
        if default_dir.exists():
            metric_files = list(default_dir.glob('*/yolo_metrics.txt'))
    
    if not metric_files:
        print("错误: 未找到 metrics 文件。请提供文件路径或使用 --input-dir")
        sys.exit(1)
    
    # 去重并保持顺序
    metric_files = list(dict.fromkeys([str(f) for f in metric_files]))
    
    print(f"找到 {len(metric_files)} 个 metrics 文件:")
    for f in metric_files:
        print(f"  - {f}")
    
    # 解析所有文件
    all_data = []
    for file_path in metric_files:
        try:
            data = parse_yolo_metrics(file_path)
            all_data.append(data)
        except Exception as e:
            print(f"解析 {file_path} 失败: {e}")
    
    if not all_data:
        print("错误: 无法解析任何文件")
        sys.exit(1)
    
    # 打印摘要
    print_summary(all_data)
    
    # 绘制图表
    plot_comparison(all_data, args.output)


if __name__ == '__main__':
    main()
