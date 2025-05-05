# utils/model_storage.py

import os
import pickle
import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

# モデル保存用のディレクトリパス
MODELS_DIR = "saved_models"

def ensure_model_directory():
    """モデル保存用ディレクトリが存在することを確認、なければ作成"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

def save_model(model: Any, 
              model_name: str, 
              model_type: str,
              target_variable: str, 
              selected_features: List[str],
              car_class: str,
              comparison_results: pd.DataFrame,
              metrics: Dict[str, float]) -> str:
    """
    学習済みモデルと関連情報を保存
    
    Parameters:
    -----------
    model: PyCaret model
        保存するモデルオブジェクト
    model_name: str
        モデルに付ける名前（ユーザー指定）
    model_type: str
        モデルの種類（例：'xgboost', 'lightgbm'）
    target_variable: str
        予測対象の変数名
    selected_features: List[str]
        モデルの学習に使用した特徴量リスト
    car_class: str
        対象の車両クラス（'全クラス'の場合もある）
    comparison_results: pd.DataFrame
        PyCaretのモデル比較結果
    metrics: Dict[str, float]
        モデルの評価指標
        
    Returns:
    --------
    str: 保存されたモデルのファイルパス
    """
    ensure_model_directory()
    
    # タイムスタンプ生成（ファイル名用）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ファイル名を生成（重複防止のためタイムスタンプを追加）
    safe_model_name = model_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    model_filename = f"{safe_model_name}_{timestamp}.pkl"
    model_path = os.path.join(MODELS_DIR, model_filename)
    
    # メタデータを作成
    metadata = {
        "model_name": model_name,
        "model_type": model_type,
        "target_variable": target_variable,
        "selected_features": selected_features,
        "car_class": car_class,
        "creation_date": timestamp,
        "metrics": metrics,
    }
    
    # 比較結果をCSVとして保存
    comparison_csv_filename = f"{safe_model_name}_{timestamp}_comparison.csv"
    comparison_csv_path = os.path.join(MODELS_DIR, comparison_csv_filename)
    comparison_results.to_csv(comparison_csv_path, index=False)
    
    # メタデータにパスを追加
    metadata["comparison_results_path"] = comparison_csv_path
    
    # メタデータをJSONとして保存
    metadata_filename = f"{safe_model_name}_{timestamp}_metadata.json"
    metadata_path = os.path.join(MODELS_DIR, metadata_filename)
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # モデルをpickleで保存
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model_path

def load_model(model_path: str) -> Optional[Any]:
    """保存されたモデルを読み込む"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"モデルの読み込み中にエラーが発生しました: {e}")
        return None

def get_model_metadata(model_filename: str) -> Optional[Dict[str, Any]]:
    """モデルのメタデータを取得"""
    # モデルファイル名からメタデータファイル名を生成
    base_name = model_filename.split('.')[0]  # 拡張子を除去
    metadata_filename = f"{base_name}_metadata.json"
    metadata_path = os.path.join(MODELS_DIR, metadata_filename)
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        st.error(f"メタデータの読み込み中にエラーが発生しました: {e}")
        return None

def list_saved_models() -> List[Dict[str, Any]]:
    """保存されたモデルの一覧とメタデータを取得"""
    ensure_model_directory()
    
    models_info = []
    
    # モデルディレクトリ内のファイルを検索
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith('.pkl'):
            # モデルのメタデータを取得
            metadata = get_model_metadata(filename)
            if metadata:
                model_info = {
                    "filename": filename,
                    "path": os.path.join(MODELS_DIR, filename),
                    **metadata
                }
                models_info.append(model_info)
    
    # 作成日時で降順ソート（最新のモデルが先頭に）
    models_info.sort(key=lambda x: x.get('creation_date', ''), reverse=True)
    
    return models_info

def load_comparison_results(comparison_path: str) -> Optional[pd.DataFrame]:
    """保存されたモデル比較結果を読み込む"""
    try:
        df = pd.read_csv(comparison_path)
        return df
    except Exception as e:
        st.error(f"比較結果の読み込み中にエラーが発生しました: {e}")
        return None

def delete_model(model_path: str) -> bool:
    """
    モデルとその関連ファイル（メタデータと比較結果）を削除
    
    Parameters:
    -----------
    model_path: str
        削除するモデルファイルのパス
        
    Returns:
    --------
    bool: 削除に成功したかどうか
    """
    try:
        # モデルのベース名を取得
        model_dir = os.path.dirname(model_path)
        filename = os.path.basename(model_path)
        base_name = filename.split('.')[0]  # 拡張子を除去
        
        # メタデータファイルのパスを生成
        metadata_filename = f"{base_name}_metadata.json"
        metadata_path = os.path.join(model_dir, metadata_filename)
        
        # モデルのメタデータを読み込んで比較結果ファイルのパスを取得
        metadata = get_model_metadata(filename)
        comparison_path = metadata.get("comparison_results_path") if metadata else None
        
        # ファイルを削除
        if os.path.exists(model_path):
            os.remove(model_path)
        
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        if comparison_path and os.path.exists(comparison_path):
            os.remove(comparison_path)
        
        return True
    except Exception as e:
        st.error(f"モデルの削除中にエラーが発生しました: {e}")
        return False 