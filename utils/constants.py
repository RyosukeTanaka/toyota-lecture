# utils/constants.py

# --- 定数 ---
TARGET_VARIABLE = '利用台数累積'
DATE_COLUMN = '利用日'
PRICE_COLUMNS = ['価格_トヨタ']
LEAD_TIME_COLUMN = 'リードタイム_計算済'
CAR_CLASS_COLUMN = '車両クラス'
BOOKING_DATE_COLUMN = '予約日'
USAGE_COUNT_COLUMN = '利用台数'
LAG_TARGET_COLUMN = '利用台数累積' # ラグ計算の対象列
LAG_DAYS = 30 # ラグ日数
LAG_GROUP_COLS = [DATE_COLUMN, CAR_CLASS_COLUMN] # ラグ計算時のグループ化列 