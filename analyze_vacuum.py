#!/usr/bin/env python
"""
InBox 単一興味ボックスの限界立証 - 分析実行スクリプト

このスクリプトは学習済みInBoxモデルを使用して、
「平均化の罠（Vacuum Problem）」を立証するための分析を実行します。

使用方法:
    # 基本実行（last-fm データセット）
    python analyze_vacuum.py --dataset last-fm --checkpoint_path ./logs/last-fm/2025-xx-xx --checkpoint checkpoint_train --cuda

    # 詳細オプション付き
    python analyze_vacuum.py --dataset amazon-book --checkpoint_path ./logs/amazon-book/xxx --checkpoint checkpoint_train --cuda --output_dir ./analysis_results/amazon-book --min_history 15 --top_k 100

出力:
    - split_interest_users.csv: 興味分裂ユーザーリスト
    - user_metrics.csv: ユーザーごとのDiversity, Box Volume, NDCG
    - vacuum_analysis_report.md: 総合レポート
    - scatter_plots/: 散布図（matplotlib使用時）

作成日: 2025年12月1日
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import Model
from utils.readdata import load_data
from utils.parser import parse_args
from utils.analysis import (
    scan_split_interest_users,
    scan_non_split_users,
    compute_all_user_metrics,
    compute_zone_hit_rates,
    compare_split_vs_non_split_users,
    verify_vacuum_effect,
    analyze_vacuum_by_tags,
    export_analysis_csv,
    export_hit_rate_comparison_csv,
    compute_correlation,
    run_full_analysis,
    generate_analysis_report
)


def parse_analysis_args():
    """分析用の追加引数をパース"""
    parser = argparse.ArgumentParser(
        description="InBox Vacuum Problem Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # last-fmデータセットで分析
  python analyze_vacuum.py --dataset last-fm --checkpoint_path ./logs/last-fm/2025-01-01 --checkpoint checkpoint_train --cuda
  
  # 詳細分析
  python analyze_vacuum.py --dataset amazon-book --checkpoint_path ./logs/amazon-book/xxx --checkpoint checkpoint_train --cuda --min_history 20 --top_k 100 --output_dir ./analysis_results/amazon-book
        """
    )
    
    # 環境設定
    parser.add_argument("--cuda", action='store_true', help="GPUを使用")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    
    # データセット
    parser.add_argument("--dataset", type=str, default="last-fm",
                        choices=["last-fm", "amazon-book", "alibaba-fashion", "yelp2018"],
                        help="データセット名")
    parser.add_argument("--data_path", type=str, default="data/", help="データパス")
    
    # モデル設定（ロード用）
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="チェックポイントディレクトリ（logs/dataset/timestamp）")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_train",
                        help="チェックポイント名")
    
    # 分析パラメータ
    parser.add_argument("--min_history", type=int, default=10,
                        help="分析対象の最小履歴数")
    parser.add_argument("--top_k", type=int, default=50,
                        help="詳細分析するユーザー数")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="出力ディレクトリ（デフォルト: checkpoint_path/analysis）")
    
    # 分析モード
    parser.add_argument("--quick", action='store_true',
                        help="クイック分析モード（サンプリング）")
    parser.add_argument("--full", action='store_true',
                        help="完全分析モード")
    parser.add_argument("--phase", type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help="実行するフェーズ（0=全て, 1=ユーザー発掘, 2=定量分析, 3=定性分析, 4=正解率比較）")
    
    # 可視化
    parser.add_argument("--plot", action='store_true',
                        help="散布図を生成（matplotlibが必要）")
    
    return parser.parse_args()


def load_model_config(checkpoint_path):
    """チェックポイントディレクトリから設定を読み込み"""
    config_path = os.path.join(checkpoint_path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None


def create_scatter_plots(metrics, output_dir):
    """散布図を生成"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # ヘッドレスモード
    except ImportError:
        print("Warning: matplotlib not found. Skipping plots.")
        return
    
    plot_dir = os.path.join(output_dir, 'scatter_plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # 有効なデータのみ抽出
    valid_metrics = [m for m in metrics if m['ndcg@20'] >= 0 and m['diversity_score'] > 0]
    
    if len(valid_metrics) < 10:
        print("Warning: Not enough valid data for plots.")
        return
    
    diversity = [m['diversity_score'] for m in valid_metrics]
    ndcg = [m['ndcg@20'] for m in valid_metrics]
    box_volume = [m['box_volume'] for m in valid_metrics]
    
    # Figure 1: Diversity vs NDCG
    plt.figure(figsize=(10, 8))
    plt.scatter(diversity, ndcg, alpha=0.5, s=30)
    plt.xlabel('Diversity Score (Average Pairwise Distance)', fontsize=12)
    plt.ylabel('NDCG@20', fontsize=12)
    plt.title('Interest Diversity vs Recommendation Performance', fontsize=14)
    
    # 回帰直線
    z = np.polyfit(diversity, ndcg, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(diversity), max(diversity), 100)
    plt.plot(x_line, p(x_line), "r--", alpha=0.8, label=f'Linear fit (slope={z[0]:.4f})')
    plt.legend()
    
    corr, pval = compute_correlation(valid_metrics, 'diversity_score', 'ndcg@20')
    plt.text(0.05, 0.95, f'Pearson r = {corr:.4f}\np-value = {pval:.4e}',
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'diversity_vs_ndcg.png'), dpi=150)
    plt.close()
    
    # Figure 2: Diversity vs Box Volume
    plt.figure(figsize=(10, 8))
    plt.scatter(diversity, box_volume, alpha=0.5, s=30, c='green')
    plt.xlabel('Diversity Score (Average Pairwise Distance)', fontsize=12)
    plt.ylabel('Box Volume (L1 Norm of Offset)', fontsize=12)
    plt.title('Interest Diversity vs Box Size', fontsize=14)
    
    z2 = np.polyfit(diversity, box_volume, 1)
    p2 = np.poly1d(z2)
    plt.plot(x_line, p2(x_line), "r--", alpha=0.8, label=f'Linear fit (slope={z2[0]:.4f})')
    plt.legend()
    
    corr2, pval2 = compute_correlation(valid_metrics, 'diversity_score', 'box_volume')
    plt.text(0.05, 0.95, f'Pearson r = {corr2:.4f}\np-value = {pval2:.4e}',
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'diversity_vs_volume.png'), dpi=150)
    plt.close()
    
    # Figure 3: Box Volume vs NDCG
    plt.figure(figsize=(10, 8))
    plt.scatter(box_volume, ndcg, alpha=0.5, s=30, c='purple')
    plt.xlabel('Box Volume (L1 Norm of Offset)', fontsize=12)
    plt.ylabel('NDCG@20', fontsize=12)
    plt.title('Box Size vs Recommendation Performance', fontsize=14)
    
    z3 = np.polyfit(box_volume, ndcg, 1)
    p3 = np.poly1d(z3)
    x_line3 = np.linspace(min(box_volume), max(box_volume), 100)
    plt.plot(x_line3, p3(x_line3), "r--", alpha=0.8, label=f'Linear fit (slope={z3[0]:.4f})')
    plt.legend()
    
    corr3, pval3 = compute_correlation(valid_metrics, 'box_volume', 'ndcg@20')
    plt.text(0.05, 0.95, f'Pearson r = {corr3:.4f}\np-value = {pval3:.4e}',
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'volume_vs_ndcg.png'), dpi=150)
    plt.close()
    
    print(f"Scatter plots saved to {plot_dir}")


def main():
    args = parse_analysis_args()
    
    print("=" * 70)
    print("InBox Vacuum Problem Analysis")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.checkpoint_path}/{args.checkpoint}")
    print(f"CUDA: {args.cuda}")
    print("=" * 70)
    
    # 出力ディレクトリ設定
    if args.output_dir is None:
        args.output_dir = os.path.join(args.checkpoint_path, 'analysis')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 元の設定を読み込み
    saved_config = load_model_config(args.checkpoint_path)
    if saved_config:
        print("Loaded config from checkpoint")
    
    # モデル引数を構築
    model_args_list = [
        '--dataset', args.dataset,
        '--data_path', args.data_path,
    ]
    
    if args.cuda:
        model_args_list.append('--cuda')
    
    # 保存された設定から次元などを取得
    if saved_config:
        model_args_list.extend(['--dim', str(saved_config.get('dim', 512))])
        model_args_list.extend(['--gamma', str(saved_config.get('gamma', 12.0))])
        if 'box_mode' in saved_config:
            model_args_list.extend(['--box_mode', saved_config['box_mode']])
    
    model_args = parse_args(model_args_list)
    model_args.save_path = args.checkpoint_path
    
    # CUDA設定
    if args.cuda and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    
    # データロード
    print("\nLoading data...")
    train_cf, test_cf, train_user_set, test_user_set, test_inter_mat, item_tag, \
        triplets_IRT, triplets_TRT, triplets_IRI, n_params = load_data(model_args)
    
    print(f"  Users: {n_params['n_users']}")
    print(f"  Items: {n_params['n_items']}")
    print(f"  Tags: {n_params['n_tags']}")
    print(f"  Train interactions: {len(train_cf)}")
    print(f"  Test interactions: {len(test_cf)}")
    
    # モデル初期化
    print("\nInitializing model...")
    model = Model(model_args, n_params)
    
    if args.cuda and torch.cuda.is_available():
        model = model.cuda()
    
    # チェックポイントロード
    checkpoint_file = os.path.join(args.checkpoint_path, args.checkpoint)
    if os.path.exists(checkpoint_file):
        print(f"Loading checkpoint: {checkpoint_file}")
        # デバイスを明示的に指定してロード
        if args.cuda and torch.cuda.is_available():
            map_location = f'cuda:{args.gpu_id}'
        else:
            map_location = 'cpu'
        checkpoint = torch.load(checkpoint_file, map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
    else:
        print(f"ERROR: Checkpoint not found: {checkpoint_file}")
        sys.exit(1)
    
    model.eval()
    
    # 分析実行
    print("\n" + "=" * 70)
    print("Starting Analysis")
    print("=" * 70)
    
    results = {}
    
    # Phase 1: ユーザー発掘
    if args.phase == 0 or args.phase == 1:
        print("\n[Phase 1] Scanning for split-interest users...")
        split_users = scan_split_interest_users(
            model, train_user_set,
            min_history=args.min_history,
            top_k=args.top_k
        )
        results['split_users'] = split_users
        
        export_analysis_csv(
            split_users,
            os.path.join(args.output_dir, 'split_interest_users.csv'),
            columns=['uid', 'silhouette_score', 'cluster_sizes']
        )
        
        print(f"\n  Found {len(split_users)} users with split interests")
        if split_users:
            high_score_users = [u for u in split_users if u['silhouette_score'] > 0.3]
            print(f"  Users with silhouette > 0.3: {len(high_score_users)}")
    
    # Phase 2: 定量分析
    if args.phase == 0 or args.phase == 2:
        print("\n[Phase 2] Computing user metrics...")
        
        if args.quick:
            # クイックモード: サンプリング
            sampled_users = dict(list(train_user_set.items())[:500])
            metrics = compute_all_user_metrics(
                model, sampled_users, test_user_set,
                item_tag, n_params['n_items'],
                min_history=5
            )
        else:
            metrics = compute_all_user_metrics(
                model, train_user_set, test_user_set,
                item_tag, n_params['n_items'],
                min_history=5
            )
        
        results['metrics'] = metrics
        
        export_analysis_csv(
            metrics,
            os.path.join(args.output_dir, 'user_metrics.csv'),
            columns=['uid', 'diversity_score', 'box_volume', 'ndcg@20', 'history_size']
        )
        
        # 相関分析
        print("\n  Correlation Analysis:")
        div_ndcg_corr, div_ndcg_p = compute_correlation(metrics, 'diversity_score', 'ndcg@20')
        div_vol_corr, div_vol_p = compute_correlation(metrics, 'diversity_score', 'box_volume')
        vol_ndcg_corr, vol_ndcg_p = compute_correlation(metrics, 'box_volume', 'ndcg@20')
        
        print(f"    Diversity vs NDCG@20:     r={div_ndcg_corr:+.4f}, p={div_ndcg_p:.4e}")
        print(f"    Diversity vs Box Volume:  r={div_vol_corr:+.4f}, p={div_vol_p:.4e}")
        print(f"    Box Volume vs NDCG@20:    r={vol_ndcg_corr:+.4f}, p={vol_ndcg_p:.4e}")
        
        results['correlations'] = {
            'diversity_vs_ndcg': (div_ndcg_corr, div_ndcg_p),
            'diversity_vs_volume': (div_vol_corr, div_vol_p),
            'volume_vs_ndcg': (vol_ndcg_corr, vol_ndcg_p)
        }
        
        # 散布図生成
        if args.plot:
            print("\n  Generating scatter plots...")
            create_scatter_plots(metrics, args.output_dir)
    
    # Phase 3: 定性分析
    if args.phase == 0 or args.phase == 3:
        print("\n[Phase 3] Verifying vacuum effect...")
        
        if 'split_users' not in results:
            # Phase 1をスキップした場合は実行
            split_users = scan_split_interest_users(
                model, train_user_set,
                min_history=args.min_history,
                top_k=args.top_k
            )
            results['split_users'] = split_users
        
        vacuum_results = []
        from tqdm import tqdm
        for user_data in tqdm(results['split_users'][:10], desc="Analyzing vacuum"):
            vr = verify_vacuum_effect(
                model, user_data, item_tag, n_params['n_items']
            )
            vacuum_results.append(vr)
        
        results['vacuum_results'] = vacuum_results
        
        # 真空地帯の詳細表示
        print("\n  === Vacuum Zone Examples ===")
        for i, vr in enumerate(vacuum_results[:3]):
            print(f"\n  User {vr['uid']} (silhouette={vr['silhouette_score']:.4f}):")
            print(f"    Cluster representatives:")
            for cid, cinfo in vr['cluster_representatives'].items():
                print(f"      Cluster {cid}: items {cinfo['items'][:3]}")
            print(f"    Vacuum items (nearest to average center):")
            for item in vr['vacuum_items'][:3]:
                print(f"      Item {item['item_id']}: dist={item['distance_to_center']:.4f}")
    
    # Phase 4: 真空地帯 vs 非真空地帯の正解率比較
    if args.phase == 0 or args.phase == 4:
        print("\n[Phase 4] Comparing hit rates: vacuum zone vs non-vacuum zone...")
        
        # 分裂ユーザーの取得（必要に応じて）
        if 'split_users' not in results:
            split_users = scan_split_interest_users(
                model, train_user_set,
                min_history=args.min_history,
                top_k=args.top_k
            )
            results['split_users'] = split_users
        
        # 非分裂ユーザーの取得
        print("\n  Scanning for non-split users (control group)...")
        non_split_users = scan_non_split_users(
            model, train_user_set,
            min_history=args.min_history,
            top_k=args.top_k,
            max_silhouette=0.2
        )
        results['non_split_users'] = non_split_users
        
        export_analysis_csv(
            non_split_users,
            os.path.join(args.output_dir, 'non_split_users.csv'),
            columns=['uid', 'silhouette_score', 'cluster_sizes']
        )
        
        # 正解率比較
        print("\n  Computing hit rates for both groups...")
        hit_rate_comparison = compare_split_vs_non_split_users(
            model, train_user_set, test_user_set, n_params['n_items'],
            results['split_users'][:args.top_k],
            non_split_users[:args.top_k]
        )
        results['hit_rate_comparison'] = hit_rate_comparison
        
        # CSV出力
        export_hit_rate_comparison_csv(
            hit_rate_comparison,
            os.path.join(args.output_dir, 'hit_rate_comparison.csv')
        )
    
    # レポート生成
    print("\n[Final] Generating report...")
    if 'metrics' in results and 'split_users' in results:
        vacuum_res = results.get('vacuum_results', [])
        hit_rate_comp = results.get('hit_rate_comparison', None)
        report_path = generate_analysis_report(
            results['split_users'],
            results['metrics'],
            vacuum_res,
            args.output_dir,
            hit_rate_comparison=hit_rate_comp
        )
        results['report_path'] = report_path
    
    # 結果サマリー
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print(f"Output directory: {args.output_dir}")
    print("\nGenerated files:")
    for f in os.listdir(args.output_dir):
        fpath = os.path.join(args.output_dir, f)
        if os.path.isfile(fpath):
            print(f"  - {f}")
        elif os.path.isdir(fpath):
            print(f"  - {f}/")
    
    # 結論
    print("\n=== Key Findings ===")
    
    if 'correlations' in results:
        div_ndcg_r, div_ndcg_p = results['correlations']['diversity_vs_ndcg']
        print("\n[相関分析]")
        if div_ndcg_r < -0.1 and div_ndcg_p < 0.05:
            print("✓ Diversity vs NDCG: 負の相関確認（仮説支持）")
            print(f"  興味が多様なユーザーほど推薦精度が低い")
        elif div_ndcg_r < 0:
            print("△ Diversity vs NDCG: 弱い負の相関")
        else:
            print("✗ Diversity vs NDCG: 負の相関なし")
    
    if 'hit_rate_comparison' in results and results['hit_rate_comparison']:
        hrc = results['hit_rate_comparison']
        print("\n[真空地帯 vs 非真空地帯 正解率比較]")
        
        split_stats = hrc['split_users']['stats']
        non_split_stats = hrc['non_split_users']['stats']
        
        print(f"\n  分裂ユーザー (n={split_stats['count']}):")
        print(f"    真空地帯正解率: {split_stats['vacuum_hit_rate_mean']:.4f}")
        print(f"    非真空地帯正解率: {split_stats['non_vacuum_hit_rate_mean']:.4f}")
        print(f"    真空地帯が劣る割合: {split_stats['vacuum_worse_ratio']*100:.1f}%")
        
        print(f"\n  非分裂ユーザー (n={non_split_stats['count']}):")
        print(f"    真空地帯正解率: {non_split_stats['vacuum_hit_rate_mean']:.4f}")
        print(f"    非真空地帯正解率: {non_split_stats['non_vacuum_hit_rate_mean']:.4f}")
        print(f"    真空地帯が劣る割合: {non_split_stats['vacuum_worse_ratio']*100:.1f}%")
        
        p_value = hrc['statistical_tests']['mann_whitney_u']['p_value']
        print(f"\n  統計検定 (Mann-Whitney U): p = {p_value:.4e}")
        
        if hrc['summary']['hypothesis_supported']:
            print("\n✓ 真空地帯問題: 分裂ユーザーで顕著に発生（仮説支持）")
            print("  → Multi-InBoxへの移行が正当化される")
        else:
            print("\n△ 真空地帯問題: 統計的有意差は確認されず")
    
    return results


if __name__ == "__main__":
    main()

