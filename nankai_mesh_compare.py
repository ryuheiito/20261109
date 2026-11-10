#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
南海トラフ巨大地震 津波浸水深メッシュ（内閣府・10mメッシュCSVのZIP）を
市区町村ごとに集計し、2つの想定（A/B）を比較する。

使い方:
  pip install numpy pandas shapely matplotlib      # 図が不要なら matplotlib は不要
  python nankai_mesh_compare.py config.json --out 出力フォルダ [--maps]

config.json の例（同梱の config_example.json 参照）:
{
  "jobs": [
    {"pref": "三重県", "city": "津市",
     "a": {"label": "平成24年 ケース9", "zip": "H24/09.zip"},
     "b": {"label": "令和7年 ケース1",  "zip": "R7/01_.zip"}}
  ]
}
- pref/city は国土数値情報(N03)の都道府県名・市区町村名（政令市は市名で全区を合算）。
- zip は内閣府のメッシュZIP。中のCSV(系ごと)を全部走査し、市域内のメッシュだけ抽出する。
- 浸水深の列は「参考値：浸水深（m）」があればそれ、なければ「浸水深（m）」を自動選択。
- 単独のZIPだけ集計したい場合は "b" を省略（A単独の集計になる）。
出力: summary_cumulative.csv / summary_bands.csv / summary_overlap.csv / stats.json
      (--maps で 比較図 PNG)
"""
import argparse, json, os, sys, zipfile, urllib.request
import numpy as np, pandas as pd
import shapely
from shapely.geometry import shape
from shapely.ops import unary_union

BOUNDARY_URL = ("https://raw.githubusercontent.com/smartnews-smri/japan-topography/main/"
                "data/municipality/geojson/s0010/N03-21_210101.json")
THRESH = [0.01, 0.3, 1, 2, 5, 10]
BANDS = [(0.01, 0.3), (0.3, 1), (1, 2), (2, 5), (5, 10), (10, 1e9)]
BAND_LABELS = ["0.01-0.3m", "0.3-1m", "1-2m", "2-5m", "5-10m", "10m以上"]
CELL_M2 = 100.0          # 1メッシュ=10m×10m
COAST_BUFFER_DEG = 0.003 # 海岸線簡略化の補正（約300m）。0で無効

def load_boundaries(path):
    if not os.path.exists(path):
        print("市域境界をダウンロード:", BOUNDARY_URL, file=sys.stderr)
        urllib.request.urlretrieve(BOUNDARY_URL, path)
    return json.load(open(path, encoding="utf-8"))["features"]

def city_geoms(features, pref, city):
    sel = [f for f in features if f["properties"].get("N03_001") == pref and
           (f["properties"].get("N03_004") == city or f["properties"].get("N03_003") == city)]
    if not sel:
        raise SystemExit(f"市域が見つかりません: {pref} {city}")
    g = unary_union([shape(f["geometry"]) for f in sel])
    ids = {id(f) for f in sel}
    x0, y0, x1, y1 = g.bounds
    near = [shape(f["geometry"]) for f in features if id(f) not in ids]
    near = [n for n in near if n.bounds[0] < x1+.1 and n.bounds[2] > x0-.1
            and n.bounds[1] < y1+.1 and n.bounds[3] > y0-.1]
    return g, (unary_union(near) if near else None)

def pick_depth_col(cols):
    for key in ("参考値：浸水深", "浸水深（m）"):
        for i, c in enumerate(cols):
            if key in c and "公表値" not in c:
                return i
    raise SystemExit("浸水深の列が見つかりません: " + str(cols))

def extract_points(zpath, geom, others, cache_dir, tag):
    cache = os.path.join(cache_dir, tag + ".npy")
    if os.path.exists(cache):
        return np.load(cache)
    z = zipfile.ZipFile(zpath)
    x0, y0, x1, y1 = geom.bounds; m = .01
    buf = geom.buffer(COAST_BUFFER_DEG) if COAST_BUFFER_DEG else geom
    out = []
    for info in z.infolist():
        if info.is_dir() or not info.filename.lower().endswith(".csv"):
            continue
        with z.open(info) as f:
            head = f.readline()
        try: cols = head.decode("cp932").replace('"', "").strip().split(",")
        except UnicodeDecodeError: cols = head.decode("utf-8", "replace").replace('"', "").strip().split(",")
        if "経度" not in cols[0]:
            continue
        dc = pick_depth_col(cols)
        with z.open(info) as f:
            for ch in pd.read_csv(f, encoding="cp932", usecols=[0, 1, dc], chunksize=2_000_000, dtype=np.float64):
                lon, lat, d = (ch.iloc[:, 0].values, ch.iloc[:, 1].values, ch.iloc[:, 2].values)
                k = (lon >= x0-m) & (lon <= x1+m) & (lat >= y0-m) & (lat <= y1+m) & (d >= 0.005)
                if not k.any(): continue
                lo, la, dd = lon[k], lat[k], d[k]
                inside = shapely.contains_xy(geom, lo, la)
                if COAST_BUFFER_DEG:
                    r = (~inside) & shapely.contains_xy(buf, lo, la)
                    if r.any():
                        idx = np.where(r)[0]
                        free = ~shapely.contains_xy(others, lo[idx], la[idx]) if others is not None else np.ones(len(idx), bool)
                        inside[idx[free]] = True
                if inside.any(): out.append(np.c_[lo[inside], la[inside], dd[inside]])
    a = np.vstack(out) if out else np.zeros((0, 3))
    np.save(cache, a)
    return a

def to_grid(a, dx, dy):
    ix = np.round(a[:, 0] / dx).astype(np.int64); iy = np.round(a[:, 1] / dy).astype(np.int64)
    key = ix * 10_000_000 + iy
    o = np.argsort(key); key = key[o]; d = a[o, 2]
    u, st = np.unique(key, return_index=True)
    return u, np.maximum.reduceat(d, st)

def summarize(D):
    f = D[D >= 0.01]
    return {"cumulative_ha": [round(float((D >= t - 1e-9).sum()) * CELL_M2 / 1e4, 1) for t in THRESH],
            "band_ha": [round(float(((D >= a) & (D < b)).sum()) * CELL_M2 / 1e4, 1) for a, b in BANDS],
            "max_m": round(float(D.max()), 2) if len(D) else 0,
            "mean_m": round(float(f.mean()), 2) if len(f) else 0,
            "p95_m": round(float(np.percentile(f, 95)), 2) if len(f) else 0}

def make_map(path, name, la, lb, lon, lat, A, B, dx, dy, geom):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib import font_manager as fm
    for n in ("Noto Sans CJK JP", "Yu Gothic", "Meiryo", "MS Gothic", "IPAexGothic"):
        if n in {f.name for f in fm.fontManager.ttflist}: plt.rcParams["font.family"] = n; break
    cm = ListedColormap(["#00ff00", "#ffff00", "#ffa500", "#f08080", "#ff2000", "#a00060", "#8000ff"])
    nm = BoundaryNorm([0.01, 0.3, 1, 2, 5, 10, 20, 100], 7)
    dcm = ListedColormap(["#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#b2182b"]); dn = BoundaryNorm([-30, -1, -.5, .5, 1, 30], 5)
    ix = np.round(lon/dx).astype(int); iy = np.round(lat/dy).astype(int)
    x0, y0 = ix.min(), iy.min(); W = ix.max()-x0+1; H = iy.max()-y0+1
    def mk(v): m = np.full((H, W), np.nan); m[iy-y0, ix-x0] = v; return m
    ext = ((x0-.5)*dx, (x0+W-.5)*dx, (y0-.5)*dy, (y0+H-.5)*dy)
    asp = 1/np.cos(np.radians((ext[2]+ext[3])/2))
    diff = np.where((A >= .01) | (B >= .01), B-A, np.nan)
    fig, ax = plt.subplots(1, 3, figsize=(21, 7.6))
    for a, (arr, t, c, n) in zip(ax, [(np.where(A >= .01, A, np.nan), la, cm, nm), (np.where(B >= .01, B, np.nan), lb, cm, nm), (diff, "差分（B − A）", dcm, dn)]):
        im = a.imshow(mk(arr), extent=ext, origin="lower", cmap=c, norm=n, interpolation="nearest", aspect=asp)
        for p in (geom.geoms if geom.geom_type == "MultiPolygon" else [geom]):
            xs, ys = p.exterior.xy; a.plot(xs, ys, "k", lw=.5)
        a.set_title(t, fontsize=15); a.set_xticks([]); a.set_yticks([]); a.set_xlim(ext[0], ext[1]); a.set_ylim(ext[2], ext[3]); a.set_facecolor("#e8e8e8")
        fig.colorbar(im, ax=a, fraction=.04, pad=.02)
    fig.suptitle(f"{name}　浸水深の比較", fontsize=19, fontweight="bold"); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config"); ap.add_argument("--out", default="out"); ap.add_argument("--maps", action="store_true")
    ap.add_argument("--boundary", default=None, help="N03 GeoJSON のパス（無ければ自動ダウンロード）")
    ap.add_argument("--base", default=".", help="config内の相対パスの基準フォルダ")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True); cache = os.path.join(args.out, "cache"); os.makedirs(cache, exist_ok=True)
    feats = load_boundaries(args.boundary or os.path.join(args.out, "N03_boundaries.json"))
    cfg = json.load(open(args.config, encoding="utf-8"))
    cum, bnd, ovl, stats = [], [], [], {}
    for job in cfg["jobs"]:
        pref, city = job["pref"], job["city"]
        geom, others = city_geoms(feats, pref, city)
        sc = {}
        for k in ("a", "b"):
            if k not in job: continue
            s = job[k]; zp = s["zip"] if os.path.isabs(s["zip"]) else os.path.join(args.base, s["zip"])
            tag = f"{city}_{os.path.basename(zp)}"
            sc[k] = (s["label"], extract_points(zp, geom, others, cache, tag))
            print(city, s["label"], len(sc[k][1]), "mesh", flush=True)
        lat0 = np.mean([v[1][:, 1].mean() for v in sc.values() if len(v[1])])
        dy = 10/110574.0; dx = 10/(111320*np.cos(np.radians(lat0)))
        grids = {k: to_grid(v[1], dx, dy) for k, v in sc.items()}
        allk = np.unique(np.concatenate([g[0] for g in grids.values()]))
        arr = {}
        for k, (u, d) in grids.items():
            D = np.zeros(len(allk)); D[np.searchsorted(allk, u)] = d; arr[k] = D
        stats[f"{pref}{city}"] = {}
        for k, (label, _) in sc.items():
            s = summarize(arr[k]); stats[f"{pref}{city}"][label] = s
            for t, v in zip(THRESH, s["cumulative_ha"]): cum.append([pref, city, label, f"{t}m以上", v])
            for bl, v in zip(BAND_LABELS, s["band_ha"]): bnd.append([pref, city, label, bl, v])
            cum.append([pref, city, label, "最大浸水深(m)", s["max_m"]]); cum.append([pref, city, label, "平均浸水深(m,浸水メッシュ)", s["mean_m"]])
        if "a" in arr and "b" in arr:
            A, B = arr["a"], arr["b"]; fa, fb = A >= .01, B >= .01; both = fa & fb; df = B-A; c2 = CELL_M2/1e4
            row = {"両方で浸水(ha)": both.sum()*c2, "Aのみ浸水(ha)": (fa & ~fb).sum()*c2, "Bのみ浸水(ha)": (fb & ~fa).sum()*c2,
                   "両方で浸水かつBが0.5m以上深い(ha)": (both & (df >= .5)).sum()*c2, "両方で浸水かつBが0.5m以上浅い(ha)": (both & (df <= -.5)).sum()*c2,
                   "両方で浸水かつ差0.5m未満(ha)": (both & (abs(df) < .5)).sum()*c2, "両方で浸水の平均差B−A(m)": float(df[both].mean()) if both.any() else 0}
            for kk, v in row.items(): ovl.append([pref, city, sc["a"][0], sc["b"][0], kk, round(float(v), 2)])
            if args.maps:
                make_map(os.path.join(args.out, f"map_{city}.png"), city, sc["a"][0], sc["b"][0], (allk//10_000_000)*dx, (allk % 10_000_000)*dy, A, B, dx, dy, geom)
    enc = "utf-8-sig"  # Excelで文字化けしない
    pd.DataFrame(cum, columns=["都道府県", "市", "想定", "指標", "値(ha または m)"]).to_csv(os.path.join(args.out, "summary_cumulative.csv"), index=False, encoding=enc)
    pd.DataFrame(bnd, columns=["都道府県", "市", "想定", "浸水深区分", "面積(ha)"]).to_csv(os.path.join(args.out, "summary_bands.csv"), index=False, encoding=enc)
    if ovl: pd.DataFrame(ovl, columns=["都道府県", "市", "A", "B", "指標", "値"]).to_csv(os.path.join(args.out, "summary_overlap.csv"), index=False, encoding=enc)
    json.dump(stats, open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("完了:", os.path.abspath(args.out))

if __name__ == "__main__":
    main()
