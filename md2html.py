import mistletoe

# ファイルへ書き出し
def write_file(file_path:str, html_str:str) -> None:
  with open(file_path, "w", encoding="utf-8") as output_file:
    output_file.write(html_str)

def get_body(md_str:str) -> dict:
  markup = mistletoe.markdown(md_str)
  return "\n".join([
    '<body>',
    markup,
    '</body>'
  ])

def md2html(md_str: str):
  body = get_body(md_str)
  head = "\n".join([
    r'<meta charset="utf-8">',
    r'<base target="_blank">', # リンクは新規タブで開く
    r'<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, minimum-scale=1.0">', # viewport を設定しておく
  ])

  markup = "\n".join([
    '<!DOCTYPE html>',
    '<html lang="ja">',
    '<link rel="stylesheet" href="style.css">',
    head,
    body,
    '</html>'
  ])

  return markup