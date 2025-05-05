# utils/model_storage.py

import os
import pickle
import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np

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
              metrics: Dict[str, float],
              model_columns: Optional[List[str]] = None,  # 追加: 学習時の特徴量列名
              categorical_features: Optional[List[str]] = None  # 追加: カテゴリカル特徴量
              ) -> str:
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
    model_columns: List[str], optional
        学習時に使用された特徴量の列名（変換後のもの）
    categorical_features: List[str], optional
        カテゴリカル特徴量のリスト
        
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
    
    # 追加: 学習時の特徴量情報をメタデータに保存
    if model_columns is not None:
        metadata["model_columns"] = model_columns
    
    if categorical_features is not None:
        metadata["categorical_features"] = categorical_features
    
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

def prepare_features_for_prediction(data: pd.DataFrame, model_metadata: Dict[str, Any]) -> pd.DataFrame:
    """
    予測用データを学習時の特徴量形式に変換する
    
    Parameters:
    -----------
    data: pd.DataFrame
        変換する元データ
    model_metadata: Dict[str, Any]
        モデルのメタデータ（特徴量情報を含む）
        
    Returns:
    --------
    pd.DataFrame: 変換後のデータ
    """
    if not model_metadata:
        st.error("モデルのメタデータがありません。特徴量変換をスキップします。")
        return data
    
    # 学習時の特徴量列名を取得
    model_columns = model_metadata.get("model_columns")
    if not model_columns:
        st.warning("モデルメタデータに特徴量列名がありません。特徴量変換をスキップします。")
        return data
    
    # 各モデル列がどの元の特徴量に由来するかを分析するためのマッピング
    column_mapping = {}
    
    try:
        # データのコピーを作成
        input_data = data.copy()
        
        # 元の列を表示（デバッグ用）
        st.info(f"元データの列: {input_data.columns.tolist()}")
        st.info(f"モデルが期待する列: {model_columns}")
        
        # 結果データフレームを新しく構築する
        data_transformed = pd.DataFrame(index=input_data.index)
        
        # --- 1. 特徴量の分類 ---
        # 元の列名パターンを分析
        date_columns = []
        categorical_columns = []
        numeric_columns = []
        
        # 日付型列を検出
        for col in input_data.columns:
            if pd.api.types.is_datetime64_dtype(input_data[col]):
                date_columns.append(col)
            elif pd.api.types.is_numeric_dtype(input_data[col]):
                numeric_columns.append(col)
            else:
                categorical_columns.append(col)
        
        st.info(f"日付型列: {date_columns}")
        st.info(f"カテゴリカル列: {categorical_columns}")
        st.info(f"数値型列: {numeric_columns}")
        
        # --- 2. モデル列の分類 ---
        # モデル列をパターンで分類
        date_derived_columns = {}
        categorical_derived_columns = {}
        numeric_model_columns = []
        
        for col in model_columns:
            # 日付派生列の検出 (例: 利用日_year)
            if '_year' in col or '_month' in col or '_day' in col:
                base_col = col.split('_')[0]
                if base_col not in date_derived_columns:
                    date_derived_columns[base_col] = []
                date_derived_columns[base_col].append(col)
            
            # カテゴリ派生列の検出 (例: ja_name_平日)
            elif '_' in col and not any(x in col for x in ['year', 'month', 'day', 'lag']):
                prefix = col.split('_')[0]
                if prefix not in categorical_derived_columns:
                    categorical_derived_columns[prefix] = []
                categorical_derived_columns[prefix].append(col)
            
            # それ以外は数値列と仮定
            else:
                numeric_model_columns.append(col)
        
        st.info(f"日付派生列: {date_derived_columns}")
        st.info(f"カテゴリ派生列: {categorical_derived_columns}")
        st.info(f"数値型モデル列: {numeric_model_columns}")
        
        # --- 3. 特徴量変換の実行 ---
        # 日付列の処理
        for base_col, derived_cols in date_derived_columns.items():
            if base_col in input_data.columns:
                if pd.api.types.is_datetime64_dtype(input_data[base_col]):
                    st.info(f"日付列 '{base_col}' を分解します")
                    for derived_col in derived_cols:
                        suffix = derived_col.replace(f"{base_col}_", "")
                        if suffix == 'year':
                            data_transformed[derived_col] = input_data[base_col].dt.year
                        elif suffix == 'month':
                            data_transformed[derived_col] = input_data[base_col].dt.month
                        elif suffix == 'day':
                            data_transformed[derived_col] = input_data[base_col].dt.day
                else:
                    st.warning(f"列 '{base_col}' は日付型ではありません。日付派生列の生成をスキップします。")
        
        # カテゴリ列の処理
        for base_col, derived_cols in categorical_derived_columns.items():
            if base_col in input_data.columns:
                st.info(f"カテゴリ列 '{base_col}' をワンホットエンコーディングします")
                
                # カテゴリ値を取得して派生列名から接頭辞を取り除く
                category_values = []
                for derived_col in derived_cols:
                    if '_' in derived_col:
                        category_value = derived_col.replace(f"{base_col}_", "", 1)
                        category_values.append(category_value)
                
                # 現在のデータから生成したダミー変数
                dummies = pd.get_dummies(input_data[base_col], prefix=base_col)
                
                # 学習時にあった全カテゴリ値を確保
                for value in category_values:
                    dummy_col = f"{base_col}_{value}"
                    if dummy_col not in dummies.columns:
                        dummies[dummy_col] = 0
                        st.info(f"カテゴリ値 '{value}' が現在のデータにないため、列 '{dummy_col}' を追加しました")
                
                # 必要な列だけを選択
                for derived_col in derived_cols:
                    if derived_col in dummies.columns:
                        data_transformed[derived_col] = dummies[derived_col]
                    else:
                        st.warning(f"カテゴリ派生列 '{derived_col}' を作成できませんでした")
                        data_transformed[derived_col] = 0
        
        # 数値列の処理
        for col in numeric_model_columns:
            if col in input_data.columns:
                data_transformed[col] = input_data[col]
            else:
                st.warning(f"数値列 '{col}' がデータにありません。0で初期化します。")
                data_transformed[col] = 0
        
        # --- 4. 最終確認と調整 ---
        # 不足している列を確認
        missing_cols = set(model_columns) - set(data_transformed.columns)
        if missing_cols:
            st.warning(f"変換後も不足している列があります: {missing_cols}")
            for col in missing_cols:
                data_transformed[col] = 0
        
        # 列の順序をモデルと同じにする
        data_transformed = data_transformed[model_columns]
        
        # 非数値列を確認
        non_numeric_cols = data_transformed.select_dtypes(exclude=['number', 'bool', 'category']).columns.tolist()
        if non_numeric_cols:
            st.warning(f"非数値型の列が残っています: {non_numeric_cols}")
            for col in non_numeric_cols:
                # object型の列を強制的に数値型に変換
                data_transformed[col] = pd.to_numeric(data_transformed[col], errors='coerce').fillna(0)
        
        # NaN値をチェック
        nan_cols = data_transformed.columns[data_transformed.isna().any()].tolist()
        if nan_cols:
            st.warning(f"NaN値を含む列: {nan_cols}")
            data_transformed = data_transformed.fillna(0)
        
        # 最終確認 - すべての列がfloat、int、boolのいずれかであることを確認
        final_dtypes = data_transformed.dtypes
        object_cols = final_dtypes[final_dtypes == 'object'].index.tolist()
        if object_cols:
            st.warning(f"最終確認: まだobject型の列が残っています。XGBoostでエラーが発生する可能性があります: {object_cols}")
            # 最終手段: すべての列を強制的にfloat64に変換（object型の列を削除する代わりに）
            for col in object_cols:
                try:
                    data_transformed[col] = data_transformed[col].astype('float64')
                except:
                    # カテゴリとして扱えるか試みる
                    try:
                        data_transformed[col] = data_transformed[col].astype('category').cat.codes.astype('float64')
                    except:
                        # どうしても変換できない場合は削除
                        st.warning(f"列 '{col}' を変換できないため削除します")
                        data_transformed = data_transformed.drop(columns=[col])
        
        st.success(f"特徴量変換完了: {len(data_transformed.columns)}個の特徴量を学習時と同じ形式に変換しました")
        
        # 変換結果のサンプルを表示（デバッグ用）- ネストしたエクスパンダーを避ける
        # エクスパンダー内なのでここでexpanderを使わない
        st.subheader("変換後の特徴量サンプル")
        st.dataframe(data_transformed.head(2))
        
        return data_transformed
        
    except Exception as e:
        st.error(f"特徴量変換中にエラーが発生しました: {e}")
        import traceback
        st.error(traceback.format_exc())
        return data 