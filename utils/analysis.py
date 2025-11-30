"""
InBox 単一興味ボックスの限界立証のための分析モジュール

このモジュールは「平均化の罠（Vacuum Problem）」を定量・定性の両面から実証するための
ツール群を提供します。

Phase 1: ターゲットユーザーの自動発掘（Clustering Scan）
Phase 2: 定量分析（Diversity Score, Box Volume, Performance）
Phase 3: 定性分析（真空地帯のアイテム特定）

作成日: 2025年11月30日
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist
from collections import defaultdict
from tqdm import tqdm
import csv
import os
from typing import Dict, List, Tuple, Optional, Any


# ============================================================
# Phase 1: ターゲットユーザーの自動発掘
# ============================================================

def scan_split_interest_users(
    model,
    train_user_set: Dict[int, List[int]],
    min_history: int = 10,
    top_k: int = 50,
    n_clusters: int = 2,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    幾何学的に興味が分裂しているユーザーを特定する
    
    Args:
        model: 学習済みInBoxモデル
        train_user_set: ユーザーID -> 履歴アイテムIDリストの辞書
        min_history: 分析に必要な最小履歴数
        top_k: 返す上位ユーザー数
        n_clusters: K-Meansのクラスタ数（デフォルト2）
        verbose: プログレスバーの表示
        
    Returns:
        シルエットスコア降順でソートされたユーザー情報リスト
        各要素は {'uid', 'silhouette_score', 'hist_ids', 'cluster_labels', 'cluster_sizes'}
    """
    candidates = []
    
    # アイテム埋め込みをCPUに転送
    item_embeds = model.item_embedding.weight.detach().cpu().numpy()
    
    user_iter = tqdm(train_user_set.items(), desc="Scanning users") if verbose else train_user_set.items()
    
    for uid, item_ids in user_iter:
        if len(item_ids) < min_history:
            continue
        
        # 履歴埋め込みの取得
        item_ids_arr = np.array(item_ids)
        # インデックスがアイテム埋め込みの範囲内か確認
        valid_mask = item_ids_arr < len(item_embeds)
        if not valid_mask.all():
            item_ids_arr = item_ids_arr[valid_mask]
            if len(item_ids_arr) < min_history:
                continue
        
        history_vecs = item_embeds[item_ids_arr]
        
        # K-Meansクラスタリングによる分離度チェック
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(history_vecs)
            
            # クラスタが実際に分離しているか確認（両クラスタに最低2点必要）
            unique, counts = np.unique(labels, return_counts=True)
            if len(unique) < n_clusters or min(counts) < 2:
                continue
                
            score = silhouette_score(history_vecs, labels)
            
            candidates.append({
                'uid': uid,
                'silhouette_score': score,
                'hist_ids': list(item_ids_arr),
                'cluster_labels': labels.tolist(),
                'cluster_sizes': counts.tolist(),
                'cluster_centers': kmeans.cluster_centers_.tolist()
            })
        except Exception as e:
            continue
    
    # スコア降順でソート
    candidates.sort(key=lambda x: x['silhouette_score'], reverse=True)
    
    if verbose:
        print(f"\n分析対象ユーザー数: {len(candidates)} / {len(train_user_set)}")
        if candidates:
            print(f"最高シルエットスコア: {candidates[0]['silhouette_score']:.4f}")
            print(f"最低シルエットスコア: {candidates[-1]['silhouette_score']:.4f}")
    
    return candidates[:top_k]


# ============================================================
# Phase 2: 定量分析
# ============================================================

def compute_diversity_score(
    model,
    item_ids: List[int],
    metric: str = 'euclidean'
) -> float:
    """
    履歴アイテム間の平均ペアワイズ距離（Diversity Score）を計算
    
    Args:
        model: 学習済みInBoxモデル
        item_ids: アイテムIDのリスト
        metric: 距離メトリック（デフォルト'euclidean'）
        
    Returns:
        平均ペアワイズ距離
    """
    if len(item_ids) < 2:
        return 0.0
    
    item_embeds = model.item_embedding.weight.detach().cpu().numpy()
    item_ids_arr = np.array(item_ids)
    valid_mask = item_ids_arr < len(item_embeds)
    item_ids_arr = item_ids_arr[valid_mask]
    
    if len(item_ids_arr) < 2:
        return 0.0
    
    history_vecs = item_embeds[item_ids_arr]
    
    # ペアワイズ距離の計算
    pairwise_dists = pdist(history_vecs, metric=metric)
    
    return float(np.mean(pairwise_dists))


def compute_box_volume(
    model,
    item_ids: List[int],
    item_tag: Dict[int, List],
    use_cuda: bool = True
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    InBoxのロジックでユーザーボックスを生成し、そのボリュームを計算
    
    Args:
        model: 学習済みInBoxモデル
        item_ids: アイテムIDのリスト
        item_tag: アイテムID -> タグ情報の辞書
        use_cuda: CUDAを使用するかどうか
        
    Returns:
        (box_volume, user_center, user_offset)
        - box_volume: オフセットのL1ノルム（ボックスの「体積」の指標）
        - user_center: ユーザーボックスの中心
        - user_offset: ユーザーボックスのオフセット
    """
    model.eval()
    
    with torch.no_grad():
        # 簡易版：アイテム埋め込みの平均で代用
        # （実際のforward_recommenderはタグ情報も使用するが、
        #   主な影響はアイテム埋め込みの平均化）
        item_ids_tensor = torch.tensor(item_ids, dtype=torch.long)
        if use_cuda and torch.cuda.is_available():
            item_ids_tensor = item_ids_tensor.cuda()
        
        # ボックスの中心とオフセットを計算
        item_embeds = model.item_embedding(item_ids_tensor)
        
        # 興味ボックスの取得
        if hasattr(model, 'interest_center_embedding') and hasattr(model, 'interest_offset_embedding'):
            interest_centers = model.interest_center_embedding(item_ids_tensor)
            interest_offsets = model.func(model.interest_offset_embedding(item_ids_tensor))
        else:
            # 代替：アイテム埋め込みを直接使用
            interest_centers = item_embeds
            interest_offsets = torch.ones_like(item_embeds) * 0.1
        
        # 単純平均（InBoxの現行ロジック）
        user_center = interest_centers.mean(dim=0)
        user_offset = interest_offsets.mean(dim=0)
        
        # ボックスボリュームの計算（L1ノルム）
        box_volume = torch.norm(user_offset, p=1).item()
        
        return box_volume, user_center, user_offset


def compute_user_ndcg(
    model,
    uid: int,
    train_user_set: Dict[int, List[int]],
    test_user_set: Dict[int, List[int]],
    item_tag: Dict[int, List],
    n_items: int,
    k: int = 20,
    use_cuda: bool = True
) -> float:
    """
    特定ユーザーのNDCG@kを計算
    
    Args:
        model: 学習済みInBoxモデル
        uid: ユーザーID
        train_user_set: 訓練データ
        test_user_set: テストデータ
        item_tag: アイテムタグ情報
        n_items: アイテム総数
        k: ランキング位置
        use_cuda: CUDAを使用するかどうか
        
    Returns:
        NDCG@k スコア
    """
    if uid not in test_user_set or len(test_user_set[uid]) == 0:
        return -1.0  # テストデータがない場合
    
    model.eval()
    
    with torch.no_grad():
        train_items = train_user_set.get(uid, [])
        test_items = set(test_user_set[uid])
        
        if len(train_items) == 0:
            return -1.0
        
        # ユーザーボックスの計算（簡易版）
        item_ids_tensor = torch.tensor(train_items, dtype=torch.long)
        if use_cuda and torch.cuda.is_available():
            item_ids_tensor = item_ids_tensor.cuda()
        
        interest_centers = model.interest_center_embedding(item_ids_tensor)
        interest_offsets = model.func(model.interest_offset_embedding(item_ids_tensor))
        
        user_center = interest_centers.mean(dim=0, keepdim=True)
        user_offset = interest_offsets.mean(dim=0, keepdim=True)
        
        # 全アイテムに対するスコア計算
        all_items = model.item_embedding.weight
        
        # point_box_logit の簡易版
        # box内部からの距離を計算
        lower = user_center - user_offset
        upper = user_center + user_offset
        
        dist_outside = torch.max(
            torch.zeros_like(all_items),
            torch.max(lower - all_items, all_items - upper)
        )
        dist = torch.norm(dist_outside, p=1, dim=1)
        score = -dist  # 距離が小さいほど高スコア
        
        # 訓練データはマスク（-inf）
        train_mask = torch.zeros(n_items, dtype=torch.bool)
        if use_cuda and torch.cuda.is_available():
            train_mask = train_mask.cuda()
        for item_id in train_items:
            if item_id < n_items:
                train_mask[item_id] = True
        score[train_mask] = float('-inf')
        
        # Top-k取得
        _, topk_indices = torch.topk(score, k)
        topk_indices = topk_indices.cpu().numpy()
        
        # NDCG計算
        relevance = [1 if idx in test_items else 0 for idx in topk_indices]
        
        # DCG
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
        
        # IDCG
        ideal_relevance = [1] * min(len(test_items), k) + [0] * max(0, k - len(test_items))
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg


def compute_all_user_metrics(
    model,
    train_user_set: Dict[int, List[int]],
    test_user_set: Dict[int, List[int]],
    item_tag: Dict[int, List],
    n_items: int,
    min_history: int = 5,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    全ユーザーに対してDiversity Score、Box Volume、NDCGを計算
    
    Args:
        model: 学習済みInBoxモデル
        train_user_set: 訓練データ
        test_user_set: テストデータ
        item_tag: アイテムタグ情報
        n_items: アイテム総数
        min_history: 最小履歴数
        verbose: プログレスバー表示
        
    Returns:
        各ユーザーの指標を含む辞書のリスト
    """
    results = []
    
    user_iter = tqdm(train_user_set.items(), desc="Computing metrics") if verbose else train_user_set.items()
    
    for uid, item_ids in user_iter:
        if len(item_ids) < min_history:
            continue
        
        # Diversity Score
        diversity = compute_diversity_score(model, item_ids)
        
        # Box Volume
        try:
            box_volume, _, _ = compute_box_volume(model, item_ids, item_tag)
        except Exception as e:
            box_volume = -1.0
        
        # NDCG
        try:
            ndcg = compute_user_ndcg(
                model, uid, train_user_set, test_user_set, 
                item_tag, n_items
            )
        except Exception as e:
            ndcg = -1.0
        
        results.append({
            'uid': uid,
            'diversity_score': diversity,
            'box_volume': box_volume,
            'ndcg@20': ndcg,
            'history_size': len(item_ids)
        })
    
    return results


# ============================================================
# Phase 3: 定性分析（真空地帯検証）
# ============================================================

def verify_vacuum_effect(
    model,
    user_data: Dict[str, Any],
    item_tag: Dict[int, List],
    n_items: int,
    exclude_history: bool = True,
    top_k: int = 5,
    use_cuda: bool = True
) -> Dict[str, Any]:
    """
    ユーザーの平均ボックス中心に最も近いアイテムが「無関係」か確認する
    
    Args:
        model: 学習済みInBoxモデル
        user_data: scan_split_interest_usersの出力要素
        item_tag: アイテムタグ情報
        n_items: アイテム総数
        exclude_history: 履歴アイテムを除外するか
        top_k: 返す近傍アイテム数
        use_cuda: CUDAを使用するか
        
    Returns:
        検証結果の辞書
    """
    model.eval()
    
    uid = user_data['uid']
    hist_ids = user_data['hist_ids']
    labels = user_data['cluster_labels']
    
    with torch.no_grad():
        # 履歴アイテムの埋め込み取得
        hist_tensor = torch.tensor(hist_ids, dtype=torch.long)
        if use_cuda and torch.cuda.is_available():
            hist_tensor = hist_tensor.cuda()
        
        item_embeds = model.item_embedding(hist_tensor)
        
        # 全体の中心（真空地帯）
        overall_center = item_embeds.mean(dim=0)
        
        # クラスタ別の中心
        labels_arr = np.array(labels)
        cluster_centers = {}
        for cluster_id in np.unique(labels_arr):
            mask = labels_arr == cluster_id
            # numpy maskをPyTorchテンソルに変換してインデックス
            mask_tensor = torch.from_numpy(mask)
            if use_cuda and torch.cuda.is_available():
                mask_tensor = mask_tensor.cuda()
            cluster_items = hist_tensor[mask_tensor]
            cluster_embeds = model.item_embedding(cluster_items)
            cluster_centers[cluster_id] = cluster_embeds.mean(dim=0)
        
        # 全アイテムとの距離計算
        all_items = model.item_embedding.weight
        dists_to_center = torch.norm(all_items - overall_center, p=2, dim=1)
        
        # 履歴アイテムを除外
        if exclude_history:
            hist_set = set(hist_ids)
            for idx in hist_set:
                if idx < len(dists_to_center):
                    dists_to_center[idx] = float('inf')
        
        # Top-k最近傍アイテム
        _, nearest_indices = torch.topk(dists_to_center, top_k, largest=False)
        nearest_indices = nearest_indices.cpu().numpy()
        
        # 各クラスタへの距離も計算
        cluster_distances = {}
        for cluster_id, center in cluster_centers.items():
            cluster_distances[cluster_id] = [
                torch.norm(all_items[idx] - center, p=2).item()
                for idx in nearest_indices
            ]
        
        # 結果を構築
        vacuum_items = []
        for rank, item_idx in enumerate(nearest_indices):
            item_info = {
                'item_id': int(item_idx),
                'rank': rank + 1,
                'distance_to_center': dists_to_center[item_idx].item(),
            }
            
            # 各クラスタへの距離
            for cluster_id, dists in cluster_distances.items():
                item_info[f'distance_to_cluster_{cluster_id}'] = dists[rank]
            
            # タグ情報（存在すれば）
            if item_idx in item_tag:
                item_info['tags'] = item_tag[int(item_idx)]
            
            vacuum_items.append(item_info)
        
        # クラスタごとの代表アイテム（タグ情報付き）
        cluster_representatives = {}
        for cluster_id in np.unique(labels_arr):
            mask = labels_arr == cluster_id
            cluster_hist = [hist_ids[i] for i in range(len(hist_ids)) if mask[i]]
            cluster_representatives[int(cluster_id)] = {
                'items': cluster_hist[:5],  # 最初の5アイテム
                'tags': [item_tag.get(item_id, []) for item_id in cluster_hist[:5]]
            }
    
    return {
        'uid': uid,
        'overall_center_coords': overall_center.cpu().numpy().tolist(),
        'vacuum_items': vacuum_items,
        'cluster_representatives': cluster_representatives,
        'silhouette_score': user_data.get('silhouette_score', -1)
    }


def analyze_vacuum_by_tags(
    vacuum_result: Dict[str, Any],
    tag_id_to_name: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """
    真空地帯のアイテムとクラスタ代表アイテムのタグを比較分析
    
    Args:
        vacuum_result: verify_vacuum_effectの出力
        tag_id_to_name: タグIDから名前へのマッピング（オプション）
        
    Returns:
        タグ分析結果
    """
    def get_tag_ids(tag_info_list):
        """タグ情報からタグIDのセットを抽出"""
        tags = set()
        for tag_info in tag_info_list:
            if isinstance(tag_info, list):
                for item in tag_info:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        tags.add(item[1])  # [relation, tag]の形式
                    elif isinstance(item, int):
                        tags.add(item)
            elif isinstance(tag_info, int):
                tags.add(tag_info)
        return tags
    
    # クラスタのタグ収集
    cluster_tags = {}
    for cluster_id, info in vacuum_result['cluster_representatives'].items():
        tags_flat = []
        for tag_list in info['tags']:
            tags_flat.extend(tag_list if isinstance(tag_list, list) else [tag_list])
        cluster_tags[cluster_id] = get_tag_ids([tags_flat])
    
    # 真空アイテムのタグ
    vacuum_tags = set()
    for item in vacuum_result['vacuum_items']:
        if 'tags' in item:
            vacuum_tags.update(get_tag_ids([item['tags']]))
    
    # 重複チェック
    all_cluster_tags = set()
    for tags in cluster_tags.values():
        all_cluster_tags.update(tags)
    
    overlap_with_clusters = vacuum_tags.intersection(all_cluster_tags)
    unique_to_vacuum = vacuum_tags - all_cluster_tags
    
    return {
        'uid': vacuum_result['uid'],
        'cluster_tags': {k: list(v) for k, v in cluster_tags.items()},
        'vacuum_item_tags': list(vacuum_tags),
        'overlap_with_clusters': list(overlap_with_clusters),
        'unique_to_vacuum': list(unique_to_vacuum),
        'overlap_ratio': len(overlap_with_clusters) / len(vacuum_tags) if vacuum_tags else 0
    }


# ============================================================
# Phase 4: 真空地帯 vs 非真空地帯の正解率比較
# ============================================================

def compute_zone_hit_rates(
    model,
    uid: int,
    train_user_set: Dict[int, List[int]],
    test_user_set: Dict[int, List[int]],
    n_items: int,
    n_clusters: int = 2,
    top_k: int = 20,
    use_cuda: bool = True
) -> Dict[str, Any]:
    """
    真空地帯と非真空地帯（各クラスタ中心付近）の正解率を比較
    
    Args:
        model: 学習済みInBoxモデル
        uid: ユーザーID
        train_user_set: 訓練データ
        test_user_set: テストデータ
        n_items: アイテム総数
        n_clusters: クラスタ数
        top_k: 各ゾーンから取得するアイテム数
        use_cuda: CUDAを使用するか
        
    Returns:
        各ゾーンの正解率を含む辞書
    """
    if uid not in train_user_set or len(train_user_set[uid]) < 5:
        return None
    
    if uid not in test_user_set or len(test_user_set[uid]) == 0:
        return None
    
    model.eval()
    
    hist_ids = train_user_set[uid]
    test_items = set(test_user_set[uid])
    
    with torch.no_grad():
        # アイテム埋め込み取得
        hist_tensor = torch.tensor(hist_ids, dtype=torch.long)
        if use_cuda and torch.cuda.is_available():
            hist_tensor = hist_tensor.cuda()
        
        item_embeds = model.item_embedding(hist_tensor).cpu().numpy()
        
        # K-Meansクラスタリング
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(item_embeds)
            silhouette = silhouette_score(item_embeds, labels)
        except:
            return None
        
        # 各中心を計算
        overall_center = item_embeds.mean(axis=0)  # 真空地帯中心
        cluster_centers = kmeans.cluster_centers_  # 各クラスタ中心
        
        # 全アイテム埋め込み
        all_items = model.item_embedding.weight.detach()
        if use_cuda and torch.cuda.is_available():
            all_items = all_items.cpu()
        all_items_np = all_items.numpy()
        
        # 訓練データマスク
        train_mask = np.zeros(n_items, dtype=bool)
        for item_id in hist_ids:
            if item_id < n_items:
                train_mask[item_id] = True
        
        # 真空地帯：全体中心に最も近いアイテム
        dists_to_vacuum = np.linalg.norm(all_items_np - overall_center, axis=1)
        dists_to_vacuum[train_mask] = np.inf
        
        # infでないアイテムのみを取得（最大top_k個）
        valid_vacuum_indices = np.where(dists_to_vacuum < np.inf)[0]
        sorted_vacuum_indices = valid_vacuum_indices[np.argsort(dists_to_vacuum[valid_vacuum_indices])]
        vacuum_items = sorted_vacuum_indices[:top_k]
        actual_vacuum_count = len(vacuum_items)
        
        vacuum_hits = sum(1 for item in vacuum_items if item in test_items)
        vacuum_hit_rate = vacuum_hits / actual_vacuum_count if actual_vacuum_count > 0 else 0.0
        
        # 非真空地帯：各クラスタ中心に最も近いアイテム（真空地帯と重複を除く）
        cluster_results = {}
        all_cluster_items = set()
        total_cluster_items = 0
        
        for cluster_id, center in enumerate(cluster_centers):
            dists_to_cluster = np.linalg.norm(all_items_np - center, axis=1)
            dists_to_cluster[train_mask] = np.inf
            
            # 真空地帯アイテムを除外
            for v_item in vacuum_items:
                dists_to_cluster[v_item] = np.inf
            
            # infでないアイテムのみを取得（最大top_k個）
            valid_cluster_indices = np.where(dists_to_cluster < np.inf)[0]
            sorted_cluster_indices = valid_cluster_indices[np.argsort(dists_to_cluster[valid_cluster_indices])]
            cluster_top_items = sorted_cluster_indices[:top_k]
            actual_cluster_count = len(cluster_top_items)
            
            cluster_hits = sum(1 for item in cluster_top_items if item in test_items)
            cluster_hit_rate = cluster_hits / actual_cluster_count if actual_cluster_count > 0 else 0.0
            
            cluster_results[cluster_id] = {
                'items': cluster_top_items.tolist(),
                'hits': cluster_hits,
                'hit_rate': cluster_hit_rate,
                'actual_count': actual_cluster_count
            }
            all_cluster_items.update(cluster_top_items.tolist())
            total_cluster_items += actual_cluster_count
        
        # 非真空地帯の平均正解率
        non_vacuum_total_hits = sum(cr['hits'] for cr in cluster_results.values())
        non_vacuum_hit_rate = non_vacuum_total_hits / total_cluster_items if total_cluster_items > 0 else 0.0
        
        return {
            'uid': uid,
            'silhouette_score': silhouette,
            'history_size': len(hist_ids),
            'test_size': len(test_items),
            # 真空地帯
            'vacuum_items': vacuum_items.tolist(),
            'vacuum_hits': vacuum_hits,
            'vacuum_hit_rate': vacuum_hit_rate,
            'vacuum_item_count': actual_vacuum_count,
            # 非真空地帯（クラスタ別）
            'cluster_results': cluster_results,
            'non_vacuum_hit_rate': non_vacuum_hit_rate,
            'non_vacuum_item_count': total_cluster_items,
            # 比較
            'vacuum_vs_non_vacuum_diff': vacuum_hit_rate - non_vacuum_hit_rate,
            'vacuum_is_worse': vacuum_hit_rate < non_vacuum_hit_rate
        }


def scan_non_split_users(
    model,
    train_user_set: Dict[int, List[int]],
    min_history: int = 10,
    top_k: int = 50,
    n_clusters: int = 2,
    max_silhouette: float = 0.2,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    興味が分裂していない（シルエットスコアが低い）ユーザーを特定
    
    Args:
        model: 学習済みInBoxモデル
        train_user_set: ユーザーID -> 履歴アイテムIDリストの辞書
        min_history: 分析に必要な最小履歴数
        top_k: 返す上位ユーザー数
        n_clusters: K-Meansのクラスタ数
        max_silhouette: このスコア以下を「非分裂」と見なす
        verbose: プログレスバーの表示
        
    Returns:
        シルエットスコア昇順でソートされたユーザー情報リスト
    """
    candidates = []
    
    item_embeds = model.item_embedding.weight.detach().cpu().numpy()
    
    user_iter = tqdm(train_user_set.items(), desc="Scanning non-split users") if verbose else train_user_set.items()
    
    for uid, item_ids in user_iter:
        if len(item_ids) < min_history:
            continue
        
        item_ids_arr = np.array(item_ids)
        valid_mask = item_ids_arr < len(item_embeds)
        if not valid_mask.all():
            item_ids_arr = item_ids_arr[valid_mask]
            if len(item_ids_arr) < min_history:
                continue
        
        history_vecs = item_embeds[item_ids_arr]
        
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(history_vecs)
            
            unique, counts = np.unique(labels, return_counts=True)
            if len(unique) < n_clusters or min(counts) < 2:
                continue
            
            score = silhouette_score(history_vecs, labels)
            
            # 非分裂ユーザー（シルエットスコアが低い）のみ
            if score <= max_silhouette:
                candidates.append({
                    'uid': uid,
                    'silhouette_score': score,
                    'hist_ids': list(item_ids_arr),
                    'cluster_labels': labels.tolist(),
                    'cluster_sizes': counts.tolist()
                })
        except:
            continue
    
    # スコア昇順でソート（最も分裂していないユーザーが先頭）
    candidates.sort(key=lambda x: x['silhouette_score'])
    
    if verbose:
        print(f"\n非分裂ユーザー数（silhouette ≤ {max_silhouette}）: {len(candidates)}")
    
    return candidates[:top_k]


def compare_split_vs_non_split_users(
    model,
    train_user_set: Dict[int, List[int]],
    test_user_set: Dict[int, List[int]],
    n_items: int,
    split_users: List[Dict[str, Any]],
    non_split_users: List[Dict[str, Any]],
    top_k: int = 20,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    興味分裂ユーザーと非分裂ユーザーの真空地帯正解率を比較
    
    Args:
        model: 学習済みInBoxモデル
        train_user_set: 訓練データ
        test_user_set: テストデータ
        n_items: アイテム総数
        split_users: 分裂ユーザーリスト
        non_split_users: 非分裂ユーザーリスト
        top_k: 各ゾーンから取得するアイテム数
        verbose: プログレスバーの表示
        
    Returns:
        比較結果の辞書
    """
    def analyze_group(users, group_name):
        results = []
        iterator = tqdm(users, desc=f"Analyzing {group_name}") if verbose else users
        
        for user_data in iterator:
            uid = user_data['uid']
            hit_result = compute_zone_hit_rates(
                model, uid, train_user_set, test_user_set, n_items, top_k=top_k
            )
            if hit_result:
                results.append(hit_result)
        
        return results
    
    # 分裂ユーザーの分析
    split_results = analyze_group(split_users, "Split Users")
    
    # 非分裂ユーザーの分析
    non_split_results = analyze_group(non_split_users, "Non-Split Users")
    
    # 統計計算
    def compute_stats(results):
        if not results:
            return {
                'count': 0,
                'vacuum_hit_rate_mean': 0,
                'vacuum_hit_rate_std': 0,
                'non_vacuum_hit_rate_mean': 0,
                'non_vacuum_hit_rate_std': 0,
                'vacuum_worse_ratio': 0
            }
        
        vacuum_rates = [r['vacuum_hit_rate'] for r in results]
        non_vacuum_rates = [r['non_vacuum_hit_rate'] for r in results]
        vacuum_worse = [r['vacuum_is_worse'] for r in results]
        
        return {
            'count': len(results),
            'vacuum_hit_rate_mean': np.mean(vacuum_rates),
            'vacuum_hit_rate_std': np.std(vacuum_rates),
            'non_vacuum_hit_rate_mean': np.mean(non_vacuum_rates),
            'non_vacuum_hit_rate_std': np.std(non_vacuum_rates),
            'vacuum_worse_ratio': sum(vacuum_worse) / len(vacuum_worse),
            'avg_silhouette': np.mean([r['silhouette_score'] for r in results])
        }
    
    split_stats = compute_stats(split_results)
    non_split_stats = compute_stats(non_split_results)
    
    # 統計的有意差検定
    from scipy.stats import mannwhitneyu, ttest_ind
    
    if split_results and non_split_results:
        split_vacuum_rates = [r['vacuum_hit_rate'] for r in split_results]
        non_split_vacuum_rates = [r['vacuum_hit_rate'] for r in non_split_results]
        
        try:
            # Mann-Whitney U検定（ノンパラメトリック）
            stat_u, p_value_u = mannwhitneyu(split_vacuum_rates, non_split_vacuum_rates, alternative='two-sided')
            # t検定
            stat_t, p_value_t = ttest_ind(split_vacuum_rates, non_split_vacuum_rates)
        except:
            stat_u, p_value_u = 0, 1.0
            stat_t, p_value_t = 0, 1.0
    else:
        stat_u, p_value_u = 0, 1.0
        stat_t, p_value_t = 0, 1.0
    
    comparison = {
        'split_users': {
            'stats': split_stats,
            'results': split_results
        },
        'non_split_users': {
            'stats': non_split_stats,
            'results': non_split_results
        },
        'statistical_tests': {
            'mann_whitney_u': {'statistic': stat_u, 'p_value': p_value_u},
            't_test': {'statistic': stat_t, 'p_value': p_value_t}
        },
        'summary': {
            'split_vacuum_hit_rate': split_stats['vacuum_hit_rate_mean'],
            'non_split_vacuum_hit_rate': non_split_stats['vacuum_hit_rate_mean'],
            'split_vacuum_worse_ratio': split_stats['vacuum_worse_ratio'],
            'non_split_vacuum_worse_ratio': non_split_stats['vacuum_worse_ratio'],
            'hypothesis_supported': (
                split_stats['vacuum_worse_ratio'] > non_split_stats['vacuum_worse_ratio'] and
                p_value_u < 0.05
            )
        }
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("Comparison: Split vs Non-Split Users")
        print("=" * 60)
        print(f"\n【分裂ユーザー (n={split_stats['count']}, avg silhouette={split_stats.get('avg_silhouette', 0):.3f})】")
        print(f"  真空地帯 正解率: {split_stats['vacuum_hit_rate_mean']:.4f} ± {split_stats['vacuum_hit_rate_std']:.4f}")
        print(f"  非真空地帯 正解率: {split_stats['non_vacuum_hit_rate_mean']:.4f} ± {split_stats['non_vacuum_hit_rate_std']:.4f}")
        print(f"  真空地帯の方が悪い割合: {split_stats['vacuum_worse_ratio']*100:.1f}%")
        
        print(f"\n【非分裂ユーザー (n={non_split_stats['count']}, avg silhouette={non_split_stats.get('avg_silhouette', 0):.3f})】")
        print(f"  真空地帯 正解率: {non_split_stats['vacuum_hit_rate_mean']:.4f} ± {non_split_stats['vacuum_hit_rate_std']:.4f}")
        print(f"  非真空地帯 正解率: {non_split_stats['non_vacuum_hit_rate_mean']:.4f} ± {non_split_stats['non_vacuum_hit_rate_std']:.4f}")
        print(f"  真空地帯の方が悪い割合: {non_split_stats['vacuum_worse_ratio']*100:.1f}%")
        
        print(f"\n【統計検定】")
        print(f"  Mann-Whitney U: p={p_value_u:.4e}")
        print(f"  t-test: p={p_value_t:.4e}")
        
        if comparison['summary']['hypothesis_supported']:
            print("\n✓ 仮説支持: 分裂ユーザーの方が真空地帯問題の影響を強く受けている")
        else:
            print("\n△ 仮説は明確に支持されず。追加分析が必要")
    
    return comparison


def export_hit_rate_comparison_csv(
    comparison: Dict[str, Any],
    output_path: str
) -> None:
    """
    真空地帯 vs 非真空地帯の正解率比較結果をCSV出力
    """
    rows = []
    
    # 分裂ユーザー
    for r in comparison['split_users']['results']:
        rows.append({
            'uid': r['uid'],
            'user_type': 'split',
            'silhouette_score': r['silhouette_score'],
            'history_size': r['history_size'],
            'vacuum_hit_rate': r['vacuum_hit_rate'],
            'non_vacuum_hit_rate': r['non_vacuum_hit_rate'],
            'vacuum_vs_non_vacuum_diff': r['vacuum_vs_non_vacuum_diff'],
            'vacuum_is_worse': r['vacuum_is_worse']
        })
    
    # 非分裂ユーザー
    for r in comparison['non_split_users']['results']:
        rows.append({
            'uid': r['uid'],
            'user_type': 'non_split',
            'silhouette_score': r['silhouette_score'],
            'history_size': r['history_size'],
            'vacuum_hit_rate': r['vacuum_hit_rate'],
            'non_vacuum_hit_rate': r['non_vacuum_hit_rate'],
            'vacuum_vs_non_vacuum_diff': r['vacuum_vs_non_vacuum_diff'],
            'vacuum_is_worse': r['vacuum_is_worse']
        })
    
    export_analysis_csv(rows, output_path)


# ============================================================
# ユーティリティ関数
# ============================================================

def export_analysis_csv(
    results: List[Dict[str, Any]],
    output_path: str,
    columns: Optional[List[str]] = None
) -> None:
    """
    分析結果をCSVファイルに出力
    
    Args:
        results: 結果のリスト
        output_path: 出力ファイルパス
        columns: 出力する列名リスト（Noneの場合は全列）
    """
    if not results:
        print("No results to export")
        return
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    if columns is None:
        columns = list(results[0].keys())
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Exported {len(results)} records to {output_path}")


def compute_correlation(
    results: List[Dict[str, Any]],
    x_key: str,
    y_key: str
) -> Tuple[float, float]:
    """
    2つの指標間の相関係数を計算
    
    Args:
        results: 分析結果リスト
        x_key: X軸の指標キー
        y_key: Y軸の指標キー
        
    Returns:
        (Pearson相関係数, p値)
    """
    from scipy.stats import pearsonr
    
    x_vals = []
    y_vals = []
    
    for r in results:
        x = r.get(x_key)
        y = r.get(y_key)
        if x is not None and y is not None and x != -1.0 and y != -1.0:
            x_vals.append(x)
            y_vals.append(y)
    
    if len(x_vals) < 3:
        return (0.0, 1.0)
    
    corr, p_value = pearsonr(x_vals, y_vals)
    return (corr, p_value)


def generate_analysis_report(
    split_users: List[Dict[str, Any]],
    metrics: List[Dict[str, Any]],
    vacuum_results: List[Dict[str, Any]],
    output_dir: str = './analysis_results',
    hit_rate_comparison: Optional[Dict[str, Any]] = None
) -> str:
    """
    総合分析レポートを生成
    
    Args:
        split_users: scan_split_interest_usersの結果
        metrics: compute_all_user_metricsの結果
        vacuum_results: verify_vacuum_effectの結果リスト
        output_dir: 出力ディレクトリ
        hit_rate_comparison: compare_split_vs_non_split_usersの結果（オプション）
        
    Returns:
        レポートのパス
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'vacuum_analysis_report.md')
    
    # 相関分析
    diversity_ndcg_corr, diversity_ndcg_p = compute_correlation(
        metrics, 'diversity_score', 'ndcg@20'
    )
    diversity_volume_corr, diversity_volume_p = compute_correlation(
        metrics, 'diversity_score', 'box_volume'
    )
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# InBox 単一興味ボックス限界立証レポート\n\n")
        f.write(f"**生成日時:** {np.datetime64('now')}\n\n")
        
        f.write("## 1. 概要統計\n\n")
        f.write(f"- 分析対象ユーザー数: {len(metrics)}\n")
        f.write(f"- 興味分裂ユーザー数（シルエットスコア > 0.3）: {len([u for u in split_users if u['silhouette_score'] > 0.3])}\n\n")
        
        f.write("## 2. 相関分析結果\n\n")
        f.write("| 指標ペア | Pearson相関係数 | p値 | 解釈 |\n")
        f.write("|---------|---------------|-----|------|\n")
        f.write(f"| Diversity vs NDCG@20 | {diversity_ndcg_corr:.4f} | {diversity_ndcg_p:.4e} | {'**負の相関（仮説支持）**' if diversity_ndcg_corr < -0.1 else '弱い/正の相関'} |\n")
        f.write(f"| Diversity vs Box Volume | {diversity_volume_corr:.4f} | {diversity_volume_p:.4e} | {'**正の相関（仮説支持）**' if diversity_volume_corr > 0.1 else '弱い/負の相関'} |\n\n")
        
        # 真空地帯 vs 非真空地帯の正解率比較
        if hit_rate_comparison:
            f.write("## 3. 真空地帯 vs 非真空地帯 正解率比較\n\n")
            f.write("### 3.1 分裂ユーザー vs 非分裂ユーザー\n\n")
            
            split_stats = hit_rate_comparison['split_users']['stats']
            non_split_stats = hit_rate_comparison['non_split_users']['stats']
            
            f.write("| ユーザータイプ | n | 平均シルエット | 真空地帯正解率 | 非真空地帯正解率 | 真空が劣る割合 |\n")
            f.write("|--------------|---|---------------|--------------|----------------|---------------|\n")
            f.write(f"| 分裂ユーザー | {split_stats['count']} | {split_stats.get('avg_silhouette', 0):.3f} | {split_stats['vacuum_hit_rate_mean']:.4f} ± {split_stats['vacuum_hit_rate_std']:.4f} | {split_stats['non_vacuum_hit_rate_mean']:.4f} ± {split_stats['non_vacuum_hit_rate_std']:.4f} | {split_stats['vacuum_worse_ratio']*100:.1f}% |\n")
            f.write(f"| 非分裂ユーザー | {non_split_stats['count']} | {non_split_stats.get('avg_silhouette', 0):.3f} | {non_split_stats['vacuum_hit_rate_mean']:.4f} ± {non_split_stats['vacuum_hit_rate_std']:.4f} | {non_split_stats['non_vacuum_hit_rate_mean']:.4f} ± {non_split_stats['non_vacuum_hit_rate_std']:.4f} | {non_split_stats['vacuum_worse_ratio']*100:.1f}% |\n\n")
            
            f.write("### 3.2 統計的検定\n\n")
            tests = hit_rate_comparison['statistical_tests']
            f.write(f"- **Mann-Whitney U検定**: p = {tests['mann_whitney_u']['p_value']:.4e}\n")
            f.write(f"- **t検定**: p = {tests['t_test']['p_value']:.4e}\n\n")
            
            f.write("### 3.3 解釈\n\n")
            if hit_rate_comparison['summary']['hypothesis_supported']:
                f.write("✓ **仮説支持**: 興味分裂ユーザーは真空地帯問題の影響をより強く受けている\n\n")
                f.write("- 分裂ユーザーでは真空地帯の正解率が非真空地帯より低い傾向が強い\n")
                f.write("- 非分裂ユーザーではこの傾向が弱いか存在しない\n")
                f.write("- **これは単一ボックスによる「平均化の罠」の直接的な証拠**\n\n")
            else:
                f.write("△ 仮説は明確に支持されず。以下の要因を検討：\n\n")
                f.write("- サンプルサイズの不足\n")
                f.write("- シルエットスコア閾値の調整が必要\n")
                f.write("- データセット固有の特性\n\n")
        
        f.write("## 4. 真空地帯検証（Top 5ユーザー）\n\n")
        for i, vr in enumerate(vacuum_results[:5]):
            silhouette = vr.get('silhouette_score', None)
            silhouette_str = f"{silhouette:.4f}" if silhouette is not None else "N/A"
            f.write(f"### ユーザー {vr['uid']} (シルエットスコア: {silhouette_str})\n\n")
            f.write("**推薦された「真空地帯」アイテム:**\n")
            for item in vr['vacuum_items'][:3]:
                f.write(f"- Item {item['item_id']}: 中心からの距離 = {item['distance_to_center']:.4f}\n")
            f.write("\n")
        
        f.write("## 5. 総合結論\n\n")
        
        # 仮説支持の判定
        hypothesis_points = 0
        total_points = 0
        
        # 相関分析
        total_points += 1
        if diversity_ndcg_corr < -0.1:
            hypothesis_points += 1
            f.write("✓ Diversity vs NDCG: 負の相関確認\n")
        else:
            f.write("✗ Diversity vs NDCG: 負の相関なし\n")
        
        total_points += 1
        if diversity_volume_corr > 0.1:
            hypothesis_points += 1
            f.write("✓ Diversity vs Box Volume: 正の相関確認\n")
        else:
            f.write("✗ Diversity vs Box Volume: 正の相関なし\n")
        
        # 正解率比較
        if hit_rate_comparison:
            total_points += 1
            if hit_rate_comparison['summary']['hypothesis_supported']:
                hypothesis_points += 1
                f.write("✓ 真空地帯問題: 分裂ユーザーで顕著\n")
            else:
                f.write("✗ 真空地帯問題: 統計的有意差なし\n")
        
        f.write(f"\n**総合判定: {hypothesis_points}/{total_points} の指標が仮説を支持**\n\n")
        
        if hypothesis_points >= total_points * 0.6:
            f.write("### 🎯 結論: 仮説は支持されました\n\n")
            f.write("単一ボックスによる「平均化の罠（Vacuum Problem）」は実在し、\n")
            f.write("Multi-InBox（多重興味モデル）への移行が正当化されます。\n")
        elif hypothesis_points >= total_points * 0.3:
            f.write("### ⚠️ 結論: 部分的に支持\n\n")
            f.write("一部の指標で仮説が支持されましたが、追加分析が推奨されます。\n")
        else:
            f.write("### ❌ 結論: 仮説は支持されず\n\n")
            f.write("現時点のデータでは「平均化の罠」の明確な証拠は得られませんでした。\n")
    
    print(f"Report generated: {report_path}")
    return report_path


# ============================================================
# メイン実行用関数
# ============================================================

def run_full_analysis(
    model,
    train_user_set: Dict[int, List[int]],
    test_user_set: Dict[int, List[int]],
    item_tag: Dict[int, List],
    n_items: int,
    output_dir: str = './analysis_results',
    min_history: int = 10,
    top_split_users: int = 50,
    include_hit_rate_comparison: bool = True
) -> Dict[str, Any]:
    """
    完全な分析パイプラインを実行
    
    Args:
        model: 学習済みInBoxモデル
        train_user_set: 訓練データ
        test_user_set: テストデータ
        item_tag: アイテムタグ情報
        n_items: アイテム総数
        output_dir: 出力ディレクトリ
        min_history: 最小履歴数
        top_split_users: 分析する分裂ユーザー数
        include_hit_rate_comparison: 真空地帯正解率比較を含めるか
        
    Returns:
        全分析結果を含む辞書
    """
    print("=" * 60)
    print("InBox Vacuum Problem Analysis")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Phase 1: ターゲットユーザー発掘（分裂ユーザー）
    print("\n[Phase 1a] Scanning for split-interest users...")
    split_users = scan_split_interest_users(
        model, train_user_set, 
        min_history=min_history, 
        top_k=top_split_users
    )
    export_analysis_csv(
        split_users, 
        os.path.join(output_dir, 'split_interest_users.csv'),
        columns=['uid', 'silhouette_score', 'cluster_sizes']
    )
    
    # Phase 1b: 非分裂ユーザーの発掘
    print("\n[Phase 1b] Scanning for non-split users (control group)...")
    non_split_users = scan_non_split_users(
        model, train_user_set,
        min_history=min_history,
        top_k=top_split_users,
        max_silhouette=0.2
    )
    export_analysis_csv(
        non_split_users,
        os.path.join(output_dir, 'non_split_users.csv'),
        columns=['uid', 'silhouette_score', 'cluster_sizes']
    )
    
    # Phase 2: 定量分析
    print("\n[Phase 2] Computing user metrics...")
    metrics = compute_all_user_metrics(
        model, train_user_set, test_user_set, 
        item_tag, n_items, min_history=5
    )
    export_analysis_csv(
        metrics,
        os.path.join(output_dir, 'user_metrics.csv'),
        columns=['uid', 'diversity_score', 'box_volume', 'ndcg@20', 'history_size']
    )
    
    # 相関分析
    print("\n[Phase 2b] Correlation analysis...")
    div_ndcg_corr, div_ndcg_p = compute_correlation(metrics, 'diversity_score', 'ndcg@20')
    div_vol_corr, div_vol_p = compute_correlation(metrics, 'diversity_score', 'box_volume')
    print(f"  Diversity vs NDCG@20: r={div_ndcg_corr:.4f}, p={div_ndcg_p:.4e}")
    print(f"  Diversity vs Box Volume: r={div_vol_corr:.4f}, p={div_vol_p:.4e}")
    
    # Phase 3: 定性分析
    print("\n[Phase 3] Verifying vacuum effect...")
    vacuum_results = []
    for user_data in tqdm(split_users[:10], desc="Analyzing vacuum"):
        vr = verify_vacuum_effect(model, user_data, item_tag, n_items)
        vacuum_results.append(vr)
    
    # Phase 4: 真空地帯 vs 非真空地帯の正解率比較
    hit_rate_comparison = None
    if include_hit_rate_comparison:
        print("\n[Phase 4] Comparing hit rates: vacuum zone vs non-vacuum zone...")
        hit_rate_comparison = compare_split_vs_non_split_users(
            model, train_user_set, test_user_set, n_items,
            split_users[:top_split_users],
            non_split_users[:top_split_users]
        )
        export_hit_rate_comparison_csv(
            hit_rate_comparison,
            os.path.join(output_dir, 'hit_rate_comparison.csv')
        )
    
    # レポート生成
    print("\n[Final] Generating report...")
    report_path = generate_analysis_report(
        split_users, metrics, vacuum_results, output_dir,
        hit_rate_comparison=hit_rate_comparison
    )
    
    return {
        'split_users': split_users,
        'non_split_users': non_split_users,
        'metrics': metrics,
        'vacuum_results': vacuum_results,
        'hit_rate_comparison': hit_rate_comparison,
        'correlations': {
            'diversity_vs_ndcg': (div_ndcg_corr, div_ndcg_p),
            'diversity_vs_volume': (div_vol_corr, div_vol_p)
        },
        'report_path': report_path
    }

