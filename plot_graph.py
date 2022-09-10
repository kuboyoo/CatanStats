from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import japanize_matplotlib
import numpy as np
import sys

from md2html import md2html

#dataframe to markdown table(str)
def df2md(df: pd.DataFrame):
  return df.to_markdown(tablefmt="pipe")

def loadData(path, select_year):
  with open(path) as f:
    lines = f.readlines()
  
  results = []
  gnums = {}

  for line in lines:
    line = line.strip("\n")
    if line.endswith(";") == True:
      date = line.strip(";")
      date_s = date.split("-")
      year, month, day = date_s[:3]
      if len(date_s) > 3:
        attr = date_s[3] #属性(o->オンライン, y->吉田さん参加)
        if attr == 'oy':
          is_on, is_yoshi = True, True
        elif attr == 'o':
          is_on, is_yoshi = True, False
        elif attr == 'y':
          is_on, is_yoshi = False, True
      else:
        is_on, is_yoshi = False, False

      date = "-".join([year, month, day])
    else:
      g_id, w_name = line.split()
      g_id = int(g_id)
      if select_year == "all" or year == select_year:
        results.append([date, year, month, day, g_id, is_on, is_yoshi, w_name])
        if date in gnums.keys():
          gnums[date] += 1
        else:
          gnums[date] = 1
  
  ticks = np.array([0] + [n for n in gnums.values()])
  ticks = np.cumsum(ticks)

  return np.array(results), ticks

#pandas dataframe型に変換 & 勝利数を累積
def calcNumWins(results, names):
  print(results)
  col = ["date", "year", "month", "day", "game_id", "online", "yoshi", "winner"]
  df = pd.DataFrame(results, columns=col)
  df[["year", "month", "day", "game_id"]] = \
  df[["year", "month", "day", "game_id"]].astype(int)
  d = {'True': True, 'False': False}
  df[["online", "yoshi"]] = \
  df[["online", "yoshi"]].replace(d)
  print(df)

  wnums = {} #プレイヤー毎の勝利数推移
  const_wnums, const_lnums = {}, {} #プレイヤー毎の最大連勝数, 連敗数
  for name in names:
    wnums[name] = np.array([0 for _ in range(len(df))])
  
  for i, w_name in enumerate(df["winner"]):
    wnums[w_name][i] = 1
  
  #オンライン試合区間抽出
  o_ids = df["online"].values.astype(int).nonzero()[0]
  uni   = o_ids - np.arange(len(o_ids))
  groups= [np.where(uni == u)[0]  for u in np.unique(uni)]
  o_terms = [[o_ids[min(g)], o_ids[max(g)]+1] for g in groups]
  print("onlline terms: ", o_terms)
  
  #連勝/連敗記録計算
  for key in wnums.keys():
    wflgs = wnums[key]  #0/1系列
    const_wnums[key] = max(np.diff(np.nonzero(np.append(wflgs, [0]) == 0)[0]) - 1)
    const_lnums[key] = max(np.diff(np.nonzero(1-(np.append(wflgs, [1])) == 0)[0]) - 1)
    wnums[key] = [0] + np.cumsum(wnums[key]).tolist()

    #print(key, np.diff(np.nonzero(1-wflgs == 0)[0]) - 1)

  return df, (df["date"], wnums, const_wnums, const_lnums, o_terms)#, const_ldnums)

#最大連続数のカウント 
# x: 0/1系列
# flg: True -> ==1, False -> ==0 の最大連続数
def calcConstMax(x: np.ndarray, flg: bool, df: pd.DataFrame):
  x_ = np.array([0] + x.tolist() + [0]) if flg == True else np.array([0] + (1-x).tolist() + [0])
  nz = np.nonzero(x_ == 0)[0]
  d = np.diff(nz) - 1
  n = max(d)
  s = nz[np.argmax(d)]

  s_id = "%s_%s" % (df["date"].values[s], df["game_id"].values[s])
  e_id = "%s_%s" % (df["date"].values[s+n-1], df["game_id"].values[s+n-1])
  
  return n, s_id, e_id

def plotGraph(X, colors, path, xticks, year):
  dates, wnums, const_wnums, const_lnums, o_terms=X#, const_ldnums = X
  N = len(dates) #通算試合数
  #xticks = [0, 3, 6, 10, 14]
  max_num = np.max([v for v in wnums.values()])
  yticks = np.arange(0, max_num, 3)
  print(xticks)
  idx = np.arange(N)
  res = np.setdiff1d(idx, xticks)
  flg = np.zeros(N, int)
  flg[res] = 1
  dates = [dates[i] if flg[i] == 0 else dates[i] + "-%d"%i for i in range(N)] + [dates[N-1] + "-%d"%(N+1)]
  dates[-1] = ""

  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  ax.set_title("Catan Num of Wins (%s)" % year)
  ax.set_xlabel('Date')
  ax.set_ylabel('Num of Wins')
  ax.grid(which = "major", color = "gray", alpha = 0.8,
        linestyle = "--", linewidth = 1)

  #折れ線
  for i, (name, nums) in enumerate(wnums.items()):
    n = nums[-1]
    plt.plot(dates, nums, c=colors[i], label=name + " (%2.1f%% %d / %d)" % (n/N*100., n, N))
  
  #オンライン区間塗り潰し
  label = 'Online'
  for [start, end] in o_terms:
    ax.axvspan(start, end, color="skyblue", label=label, alpha=0.4)
    label = None
  plt.xticks(xticks, rotation=45, )
  plt.yticks(yticks)
  fig.subplots_adjust(bottom=0.2)
  
  plt.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=1, fontsize=11)

  """
  #記録プロット
  record = [
    ["最多連勝(試合)", *max(const_wnums.items(), key=lambda x: x[1])],
    ["最多連敗(試合)", *max(const_lnums.items(), key=lambda x: x[1])]
    #["最多連敗(日)"  , *max(const_ldnums.items(), key=lambda x: x[1])]
  ]
  print(record)
  col_labels = ["", "プレイヤー名", "記録"]

  tbl = plt.table(cellText=record,
    colLabels=col_labels,
    colColours=["#EEEEEE","#EEEEEE","#EEEEEE"],
    colWidths=[0.2, 0.2, 0.2],
    loc='lower right'
  )
  #tbl[(1,0)].set_facecolor("w")
  tbl.auto_set_font_size(False)
  tbl.set_fontsize(8)
  """

  plt.savefig(path)

#順位表計算
def calcLB(DF: pd.DataFrame):
  cols = ["プレイヤー", "勝", "敗", "試合", "勝率", "差", "首位差", "対面", "オンライン", "直近10"]
  LB = pd.DataFrame(columns=cols)
  LB["プレイヤー"] = DF["winner"].unique()
  LB = LB.set_index("プレイヤー")
  LB["勝"] = DF.groupby(["winner"]).count()["game_id"]
  LB["試合"] = LB["勝"].sum()
  LB["敗"] = LB["試合"] - LB["勝"]
  LB["勝率"] = LB["勝"] / LB["試合"]
  LB["順"] = LB.rank(ascending=False, method='min')["勝率"].astype(int)
  LB = LB.sort_values("勝率", ascending=False) #勝率で降順ソート
  LB["差"] = np.abs(np.array([0] + np.diff(LB["勝"].values).tolist()))
  LB["首位差"] = np.cumsum(LB["差"].values)
  LB["オンライン"] = DF.groupby(["winner"]).sum()["online"]
  LB["対面"] = LB["勝"] - LB["オンライン"]
  LB["直近10"] = DF[-10:].groupby(["winner"]).count()["game_id"]
  LB = LB.fillna(0)
  LB["直近10"] = LB["直近10"].astype(int)
  LB = LB.reset_index(inplace=False)
  LB = LB.set_index("順")
  return LB

#全体, 個人のデータを計算 fmt: 対戦形式 ('all' / 'online' / 'inperson')
def calcRecord(DF: pd.DataFrame, fmt: str):
  if fmt == "オンライン": #オンライン
    DF = DF.loc[DF["online"] == True]
  elif fmt == "対面": #対面
    DF = DF.loc[DF["online"] == False]

  players = DF["winner"].unique()
  RW = pd.DataFrame([[p, *calcConstMax((DF["winner"] == p), True, DF[["date", "game_id"]])] \
    for p in players], columns=["連勝", "試合数", "開始", "終了"])
  RL = pd.DataFrame([[p, *calcConstMax((DF["winner"] == p), False, DF[["date", "game_id"]])] \
    for p in players], columns=["連敗", "試合数", "開始", "終了"])

  RW["順"] = RW.rank(ascending=False, method='min')["試合数"].astype(int)
  RL["順"] = RL.rank(ascending=False, method='min')["試合数"].astype(int)
  RW = RW.set_index("順").sort_values("順")
  RL = RL.set_index("順").sort_values("順")

  return RW, RL
  
  
#txtを出力
def saveTxt(T: str, path: Path):
  with open(path, "w", encoding="utf-8") as f:
    f.write(T)

def printMD(LB: pd.DataFrame, rec: dict, year: str, md_path: Path, graph_path: str):

  #DataFrame => Markdown
  LB["勝率"] = ['{:.3f}'.format(r)[1:] for r in LB["勝率"]] #体裁を整える
  LB["差"] = ["-"] + [d for d in LB["差"][1:]]
  LB["首位差"] = ["-"] + [d for d in LB["首位差"][1:]]
  num_on, num_inp = LB["オンライン"].sum(), LB["対面"].sum()
  LB["対面"]      = ["%d-%d" % (n, num_inp) for n in LB["対面"]]
  LB["オンライン"] = ["%d-%d" % (n, num_on) for n in LB["オンライン"]]
  LB = LB.astype(str)
  LB_md = df2md(LB)
  LB_md = LB_md.replace("0.", ".")
  print(LB_md)
  
  rec_md = {}
  for fmt, (dfw, dfl) in rec.items():
    mdw = df2md(dfw) #連勝
    mdl = df2md(dfl) #連敗
    rec_md[fmt] = "%s  \n\n%s" % (mdw, mdl)
  
  graph_url = "![graph](%s)" % str(graph_path)
  year = "通算" if year == "all" else year
  body = "# カタン会リーグ成績（%s）\n" % year
  body+= "## 順位表\n%s\n" % LB_md
  body+= "## 勝ち数推移\n%s\n" % graph_url
  
  for name, md in rec_md.items():
    body+= "### 記録（%s）\n%s\n" % (name, md)
  
  saveTxt(body, md_path)

  #Markdown => HTML
  html = md2html(body)
  html = html.replace("<h3>", "<div class=\"record\">\n<h3>")
  html = html.replace("<div class=\"record\">\n<h3>", "</div>\n<div class=\"record\">\n<h3>")
  html = html.replace("h3", "div")
  saveTxt(html, md_path.with_suffix(".html"))

#順位表, 全体記録, 個人記録, その他指標 の計算
def summarize(DF: pd.DataFrame, fmts: list):
  LB = calcLB(DF) #順位表
  rec = {} #{name: DataFrame}
  for fmt in fmts:
    rec[fmt] = calcRecord(DF, fmt)

  return LB, rec

def main(da_path, graph_path, year):
  names = ['Ishida','Kubo', 'Nisaka', 'Ohnishi']
  colors = ['red', 'navajowhite', 'darkblue', 'orange']
  results, xticks = loadData(da_path, year)
  DF, X = calcNumWins(results, names)
  DF.to_csv("./catan_record_%s.csv" % year, index=None)
  plotGraph(X, colors, graph_path, xticks, year)

  md_path = Path("./record_%s.md" % year)
  fmts = ["対面+オンライン", "対面", "オンライン"]
  LB, rec = summarize(DF, fmts)
  printMD(LB, rec, year, md_path, graph_path)

if __name__ == '__main__':
  _, da_path, year = sys.argv
  graph_path = "./wnums_%s.png" % year
  print("Selected Year: ", year)
  main(da_path, graph_path, year)
