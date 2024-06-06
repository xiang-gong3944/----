# 環境設定についてのメモ
参考にしたページや更新履歴を残していきたい。

## 2024-06-06 更新
LaTeX の pdf 出力先が out ではなく .tex ファイルと同じところに出力されるのを解決するため、
setting.json と .latexmkrc を更新した。
LaTeX workshop によるレシピから latexmkrc を直接呼ぶレシピにした。
### 変更点
.latexmkrc を以下のようにした。
一次ファイルと出力フォルダについて書き換えた。
これはこの
[ただしい高速LaTeX論](https://event.phys.s.u-tokyo.ac.jp/physlab2024/advent-calendar/24-2/)
という記事を参考にした。以降も参考にしていきたい。

~~~
#!/usr/bin/env perl

# カレントディレクトリ変更
$do_cd = 1;

# LaTeX
$latex = 'platex -synctex=1 -halt-on-error -file-line-error %O %S';
$max_repeat = 5;

# BibTeX
$bibtex = 'pbibtex %O %S';
$biber = 'biber --bblencoding=utf8 -u -U --output_safechars %O %S';

# index
$makeindex = 'mendex %O -o %D %S';

# DVI / PDF
$dvipdf = 'dvipdfmx %O -o %D %S';
$pdf_mode = 3;

# preview
$pvc_view_file_via_temporary = 0;
if ($^O eq 'linux') {
    $dvi_previewer = "xdg-open %S";
    $pdf_previewer = "xdg-open %S";
} elsif ($^O eq 'darwin') {
    $dvi_previewer = "open %S";
    $pdf_previewer = "open %S";
} else {
    $dvi_previewer = "start %S";
    $pdf_previewer = "start %S";
}
# 出力フォルダ指定
$out_dir = "./out";
# 中間ファイルを別フォルダに隠しておける
$emulate_aux = 1;
$aux_dir = ".tmp";

# # clean up
# $clean_full_ext = "%R.synctex.gz"
~~~
またこれに伴い setting.json で不要になった部分を削除した。