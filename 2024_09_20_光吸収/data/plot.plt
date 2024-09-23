gyou=122 #行番号を指定
retsu=1 #列番号を指定
filename="test.txt" # ファイル名を指定
elem=system("cat " . filename . " | awk \'NR==" . gyou ."{print $" . retsu . "}\'") # 特定の行列から値を拾う

# ここから通常の処理
set title elem
plot filename u 1:2 with linespoint