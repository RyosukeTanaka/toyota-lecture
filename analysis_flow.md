# 利用台数予測と分析の実行フロー

このドキュメントは、Streamlitアプリケーションで「分析・予測実行」ボタンがクリックされた後の、データ処理、モデル比較、予測、結果表示までの詳細なフローを説明します。

## 概要

ユーザーが分析対象のデータ（CSVファイル、車両クラス、利用日）を選択し、「分析・予測実行」ボタンを押すと、以下の主要な処理が実行されます。

1.  **モデル比較用データの準備:** 選択された車両クラスに基づいて、モデル比較に使用するデータを準備します。
2.  **モデル比較と最良モデルの選択:** PyCaretライブラリを使用し、ユーザーが選択した複数の機械学習モデルを訓練・評価し、指定された評価指標（RMSE）に基づいて最も性能の良いモデル（最良モデル）を選択します。
3.  **価格最終変更点の検出:** 選択された利用日の実績データから、価格 (`価格_トヨタ`, `価格_オリックス`) が最後に変更されたリードタイムを自動的に検出します。
4.  **予測用シナリオデータの作成:** 上記で検出された最終変更リードタイム `X` とその時点での価格に基づき、「もしリードタイム `X` 以降の価格が、リードタイム `X` 時点の価格で一定だったら」という仮定のシナリオデータを作成します。（価格変更がない場合は、このシナリオでの予測は行いません。）
5.  **予測の実行:** 選択された最良モデルとシナリオデータを用いて、利用台数の予測値を計算します。
6.  **結果の表示:** モデル比較の結果（各モデルの評価指標）と、実績データと予測結果（最終価格固定シナリオ）を比較するグラフとテーブルを表示します。

## 処理フロー図 (Mermaid)

```mermaid
graph TD
    A[分析・予測実行 ボタンクリック] --> B{車両クラス選択チェック};
    B -- "全クラス" --> C["全データ使用"];
    B -- "特定クラス" --> D["選択クラスでフィルタ"];
    C --> E["setup_and_compare_models"];
    D --> E;
    E -- 成功 --> F["モデル比較結果と\n最良モデル取得"];
    E -- 失敗 --> G["エラー表示\n処理中断"];
    F --> H_find["find_last_price_change_lead_time\n(価格最終変更点検出)"];
    H_find --> H_check{最終変更点あり?};
    H_check -- Yes --> H["create_scenario_data\n(scenario='last_change_fixed')"];
    H_check -- No --> H_warn["警告表示\n(価格変動なし)"];
    H_warn --> M; # 変動なければ評価結果のみ表示など
    H --> I["最終価格固定\nシナリオデータ作成"];
    I --> J["predict_with_model"];
    J -- 成功 --> K["予測結果取得"];
    J -- 失敗 --> L["エラー表示\n処理中断"];
    K --> M["モデル評価結果表示"];
    K --> N["実績vs予測グラフ表示\n(最終価格固定シナリオ)"];

    subgraph "ステップ詳細"
        direction LR
        E_sub["setup_and_compare_models:\npycaret.setup()\npycaret.compare_models()\npycaret.pull()"]
        H_find_sub["find_last_price_change_lead_time:\n- 価格列の差分計算\n- 最後の変化点のLT特定"]
        H_sub["create_scenario_data (last_change_fixed):\n- データコピー\n- 基準LT時点の価格取得\n- 基準LT以降の価格を固定\n- 価格差/比を再計算"]
        J_sub["predict_with_model:\npycaret.predict_model()"]
    end

    E --> E_sub;
    H_find --> H_find_sub;
    H --> H_sub;
    J --> J_sub;
```

## 各ステップの詳細

1.  **モデル比較と最良モデル選択 (`setup_and_compare_models` in `utils/modeling.py`)**
    *   **入力:** 選択された車両クラスでフィルタリングされたデータ (`data_for_modeling`)、目的変数名 (`TARGET_VARIABLE`)、数値・カテゴリ特徴量のリスト (`selected_numeric`, `selected_categorical`)、比較対象モデルのリスト (`models_to_compare`)。
    *   **処理:**
        *   `pycaret.regression.setup()`: PyCaretの環境を初期化します。データ型推論、欠損値補完、カテゴリ特徴量のエンコーディングなどの前処理パイプラインを定義し、データを訓練用とテスト用に分割します。(`verbose=False`, `html=False` で実行され、Streamlit上での冗長な出力を抑制します。)
        *   `pycaret.regression.compare_models()`: `setup`で定義された前処理パイプラインを適用した後、指定されたモデル (`include_models`) を訓練し、クロスバリデーションによって評価します。指定された評価指標 (`sort_metric='RMSE'`) に基づいてモデルをランク付けし、最も性能の良いモデルを返します。(`verbose=False` で実行されます。)
        *   `pycaret.regression.pull()`: `compare_models` の実行結果（各モデルの評価指標がまとめられた表）をPandas DataFrameとして取得します。
    *   **出力:** 最良と判断されたモデルオブジェクト (`best_model`)、モデル比較結果のDataFrame (`comparison_results`)。処理に失敗した場合は `None` と空のDataFrameを返します。

2.  **価格最終変更点検出 (`find_last_price_change_lead_time` in `utils/data_processing.py`)**
    *   **入力:** 選択された特定日のデータ (`data_filtered_sorted`)、価格関連の列名リスト (`PRICE_COLUMNS`)、リードタイム列名 (`LEAD_TIME_COLUMN`)。
    *   **処理:**
        *   `価格_トヨタ` と `価格_オリックス` の各列について、リードタイムを降順にソートし、価格が前回から変化した箇所を探します。
        *   各列で見つかった最後の変化（=最もリードタイムが小さい変化）のリードタイムを取得します。
        *   両方の価格列の最終変更リードタイムのうち、より小さい方（利用日に近い方）を返します。
        *   どちらの価格も一度も変化していない場合は `None` を返します。
    *   **出力:** 最後に価格が変更されたリードタイム (整数)、または `None`。

3.  **予測用シナリオデータ作成 (`create_scenario_data` in `utils/data_processing.py`)**
    *   **入力:** 選択された特定日のデータ (`data_filtered_sorted`)、価格関連の列名リスト (`PRICE_COLUMNS`)、リードタイム列名 (`LEAD_TIME_COLUMN`)、シナリオタイプ (`scenario_type='last_change_fixed'`)、検出された最終変更リードタイム (`change_lead_time`)。
    *   **処理 (`last_change_fixed` シナリオの場合):**
        *   入力データのコピーを作成します。
        *   指定された `change_lead_time` 時点での実績価格（トヨタ、オリックス）を取得します。
        *   コピーしたデータフレームで、リードタイムが `change_lead_time` **以下**の全ての行について、価格列の値を上記で取得した固定価格で上書きします。
        *   価格が変更されたため、価格差 (`価格差`) と価格比 (`価格比`) の特徴量を再計算します。
    *   **出力:** 価格情報が「最終価格固定シナリオ」に基づいて変更され、価格差・価格比が再計算されたDataFrame (`data_scenario`)。価格変動がない場合や処理に失敗した場合は空のDataFrameを返すことがあります。

4.  **予測実行 (`predict_with_model` in `utils/modeling.py`)**
    *   **入力:** 選択された最良モデルとシナリオデータ (`best_model`, `data_scenario`)。
    *   **処理:**
        *   `pycaret.regression.predict_model()`: 選択されたモデルとシナリオデータを用いて、利用台数の予測値を計算します。
    *   **出力:** 予測結果のDataFrame (`predictions`)。処理に失敗した場合は `None` を返します。

5.  **結果の表示 (`app.py`)**
    *   **モデル評価比較結果:** ステップ1で取得した `comparison_results` DataFrame を `st.dataframe()` で表示します。
    *   **実績 vs 予測比較グラフ・テーブル:**
        *   価格変動が検出され、予測が成功した場合に表示されます。
        *   ステップ4で取得した `predictions` DataFrame と、元の特定日のデータ (`data_filtered_sorted`) を使用します。
        *   `plot_comparison_curve` 関数 (in `utils/visualization.py`) を呼び出してグラフを生成し、`st.plotly_chart()` で表示します。
        *   比較用のデータテーブルも作成し、`st.dataframe()` で表示します。
        *   グラフとテーブルのタイトルには、「最終価格固定シナリオ (LT=X)」のように、どのシナリオに基づいているかが明記されます。
        *   価格変動が検出されなかった場合は、その旨の警告メッセージが表示されます。