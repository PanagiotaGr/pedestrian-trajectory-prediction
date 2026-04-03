import json, csv, math
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime

BASE    = Path.home() / "imptc_project"
OUT_DIR = BASE / "preprocessed"
STEP    = 10
MAX_FRAMES = 50

GROUND_NAMES = {0:"road",1:"sidewalk",2:"ground",3:"curb",
                4:"road_line",5:"crosswalk",6:"bikelane",7:"unknown"}
LIGHT_NAMES  = {2:"yellow-blinking",4:"green",10:"red",
                11:"disabled",20:"yellow",30:"red-yellow",-1:"no_signal"}

def heading_angle(vx, vy):
    if abs(vx)<1e-8 and abs(vy)<1e-8: return 0.0
    return math.atan2(vy, vx)

def rotation_matrix(theta):
    c,s = math.cos(theta), math.sin(theta)
    return np.array([[c,-s],[s,c]])

def ts_features(ts_us):
    try:
        dt  = datetime.fromtimestamp(ts_us/1e6)
        h,m = dt.hour, dt.month
        season = ("winter" if m in[12,1,2] else "spring" if m in[3,4,5]
                  else "summer" if m in[6,7,8] else "autumn")
        tod    = ("morning" if 6<=h<12 else "afternoon" if 12<=h<17
                  else "evening" if 17<=h<21 else "night")
        return h, m, season, tod
    except:
        return -1,-1,"unknown","unknown"

def load_track(path):
    data   = json.loads(Path(path).read_text())
    frames = sorted(data["track_data"].keys(), key=lambda x: int(x))
    pos, ts, gts, spds = [], [], [], []
    for k in frames:
        f = data["track_data"][k]
        c = f["coordinates"]
        pos.append([c[0], c[1]])
        ts.append(int(f["ts"]))
        gts.append(f.get("ground_type") or -1)
        spds.append(f.get("velocity") or 0.0)
    pos  = np.array(pos,  dtype=float)
    ts   = np.array(ts,   dtype=float)
    spds = np.array(spds, dtype=float)
    dt_  = np.diff(ts/1e6)
    dt_  = np.where(dt_<1e-8, 1e-8, dt_)
    dvx  = np.diff(pos[:,0])/dt_
    dvy  = np.diff(pos[:,1])/dt_
    vx   = np.concatenate([[dvx[0]], dvx])
    vy   = np.concatenate([[dvy[0]], dvy])
    vel  = np.stack([vx,vy], axis=1)
    return pos, vel, ts, gts, spds, data["overview"].get("class_name","unknown")

def load_tl(seq_folder):
    f = Path(seq_folder)/"context"/"traffic_light_signals.json"
    if not f.exists(): return {}
    d = json.loads(f.read_text())
    return {int(k):v for k,v in d.get("status_data",{}).items()}

def load_weather(seq_folder):
    f = Path(seq_folder)/"context"/"weather_data.json"
    if not f.exists(): return {}
    d = json.loads(f.read_text())
    return {int(k):v for k,v in d.get("weather_data",{}).items()}

def get_at(d, ts):
    if not d: return None
    keys = np.array(sorted(d.keys()))
    idx  = min(np.searchsorted(keys, ts), len(keys)-1)
    return d[keys[idx]]

# Crosswalk zones
cw_centers = None
try:
    zones = []
    with open(BASE/"data/crosswalk_zones.csv") as f:
        for row in csv.DictReader(f):
            zones.append([float(row["center_x"]), float(row["center_y"])])
    cw_centers = np.array(zones)
    print(f"✓ Crosswalk zones: {cw_centers}")
except: print("[!] crosswalk_zones.csv δεν βρέθηκε")

def nearest_signal(x, y):
    if cw_centers is None: return -1
    dists = np.sqrt(np.sum((cw_centers-[x,y])**2, axis=1))
    idx   = np.argmin(dists)
    return (idx+1) if dists[idx]<8.0 else -1

# Φόρτωσε matches
matches = {}
with open(OUT_DIR/"matches_all.csv") as f:
    for row in csv.DictReader(f):
        matches[row["traj_id"]] = row

# ════════════════════════════════════════
# HEADERS
# ════════════════════════════════════════
headers = (
    # Ταυτότητα
    ["traj_id","split","class_name","seq_datetime","track_id"] +
    # Χρόνος
    ["hour","month","season","time_of_day"] +
    # Weather
    ["weather_class","temperature","precipitation","wind_force","visibility"] +
    # Θέση + κίνηση
    ["start_x","start_y","theta","total_distance_m"] +
    # Στατιστικά
    ["n_frames","duration_sec","mean_speed_ms","max_speed_ms","mean_accel"] +
    # Ground type
    ["dominant_gt","dominant_gt_name","on_crosswalk"] +
    # Φανάρια
    ["nearest_signal","f1_start","f2_start","f3_start",
     "pedestrian_light_state","pedestrian_light_name"] +
    # ΟΛΗ η τροχιά
    [f"x_{i}"     for i in range(MAX_FRAMES)] +
    [f"y_{i}"     for i in range(MAX_FRAMES)] +
    [f"vx_{i}"    for i in range(MAX_FRAMES)] +
    [f"vy_{i}"    for i in range(MAX_FRAMES)] +
    [f"speed_{i}" for i in range(MAX_FRAMES)] +
    [f"gt_{i}"    for i in range(MAX_FRAMES)]
)

print(f"\nΣύνολο columns: {len(headers)}")

# ════════════════════════════════════════
# ΓΡΑΨΕ ΟΛΑ ΣΕ 1 CSV
# ════════════════════════════════════════
out_csv = OUT_DIR / "ALL_VRUs.csv"
written = 0
stats   = Counter()

with open(out_csv, "w", newline="") as fout:
    writer = csv.DictWriter(fout, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()

    for split in ["train","eval"]:
        split_dir = BASE / split
        traj_dirs = sorted(split_dir.iterdir())
        print(f"\n[{split.upper()}] {len(traj_dirs)} trajectories...")

        for traj_dir in traj_dirs:
            traj_id    = traj_dir.name
            track_full = traj_dir / "track_full.json"
            if not track_full.exists(): continue

            try:
                pos, vel, ts, gts, spds, cls = load_track(track_full)
            except: continue

            match      = matches.get(traj_id, {})
            seq_folder = match.get("seq_folder","")
            tl_data    = load_tl(seq_folder)     if seq_folder else {}
            w_data     = load_weather(seq_folder) if seq_folder else {}

            # Downsample
            pos  = pos[::STEP]
            vel  = vel[::STEP]
            ts   = ts[::STEP]
            gts  = gts[::STEP]
            spds = spds[::STEP]
            N    = len(pos)
            if N < 2: continue

            # Rotation (anchor = πρώτο frame)
            r_A   = pos[0].copy()
            v_A   = vel[0].copy()
            theta = heading_angle(v_A[0], v_A[1])
            R_A   = rotation_matrix(theta)
            pos_l = (R_A @ (pos - r_A).T).T
            vel_l = (R_A @ vel.T).T

            # Datetime
            hour, month, season, tod = ts_features(int(ts[0]))

            # Weather
            w    = get_at(w_data, int(ts[0]))
            wc   = int(w.get("weather_class",-1))       if w else -1
            temp = round(float(w.get("temperature",0)),2) if w else 0.0
            prec = round(float(w.get("precipitation_amount",0)),2) if w else 0.0
            wf   = round(float(w.get("wind_force",0)),2) if w else 0.0
            vis  = round(float(w.get("visibility",0)),2) if w else 0.0

            # Traffic lights
            tl  = get_at(tl_data, int(ts[0]))
            f1s = tl["f1"] if tl else -1
            f2s = tl["f2"] if tl else -1
            f3s = tl["f3"] if tl else -1
            sig = nearest_signal(r_A[0], r_A[1])
            lt  = {1:f1s,2:f2s,3:f3s}.get(sig,-1)

            # Ground type
            cnt = Counter(gts)
            dom = cnt.most_common(1)[0][0]

            # Στατιστικά
            diffs  = np.diff(pos, axis=0)
            dists  = np.sqrt(np.sum(diffs**2, axis=1))
            total_d= round(float(np.sum(dists)),4)
            dur    = round(float((ts[-1]-ts[0])/1e6),2)
            speeds = np.sqrt(vel[:,0]**2+vel[:,1]**2)
            accels = np.abs(np.gradient(speeds))

            # Padding
            def pad(arr, fill=""):
                out = [round(float(v),4) if fill=="" else int(v)
                       for v in arr[:MAX_FRAMES]]
                out += [fill]*(MAX_FRAMES-len(out))
                return out

            row = {
                "traj_id":       traj_id,
                "split":         split,
                "class_name":    cls,
                "seq_datetime":  match.get("seq_datetime",""),
                "track_id":      match.get("track_id",""),
                "hour":          hour,
                "month":         month,
                "season":        season,
                "time_of_day":   tod,
                "weather_class": wc,
                "temperature":   temp,
                "precipitation": prec,
                "wind_force":    wf,
                "visibility":    vis,
                "start_x":       round(float(r_A[0]),4),
                "start_y":       round(float(r_A[1]),4),
                "theta":         round(theta,6),
                "total_distance_m": total_d,
                "n_frames":      N,
                "duration_sec":  dur,
                "mean_speed_ms": round(float(np.mean(speeds)),4),
                "max_speed_ms":  round(float(np.max(speeds)),4),
                "mean_accel":    round(float(np.mean(accels)),4),
                "dominant_gt":   dom,
                "dominant_gt_name": GROUND_NAMES.get(dom,"unknown"),
                "on_crosswalk":  1 if dom==5 else 0,
                "nearest_signal": sig,
                "f1_start":      f1s,
                "f2_start":      f2s,
                "f3_start":      f3s,
                "pedestrian_light_state": lt,
                "pedestrian_light_name":  LIGHT_NAMES.get(lt,"unknown"),
            }

            xl  = pad(pos_l[:,0])
            yl  = pad(pos_l[:,1])
            vxl = pad(vel_l[:,0])
            vyl = pad(vel_l[:,1])
            sp  = pad(speeds)
            gt  = [int(g) for g in gts[:MAX_FRAMES]] + [""]*(MAX_FRAMES-min(N,MAX_FRAMES))

            for i in range(MAX_FRAMES):
                row[f"x_{i}"]     = xl[i]
                row[f"y_{i}"]     = yl[i]
                row[f"vx_{i}"]    = vxl[i]
                row[f"vy_{i}"]    = vyl[i]
                row[f"speed_{i}"] = sp[i]
                row[f"gt_{i}"]    = gt[i]

            writer.writerow(row)
            written += 1
            stats[cls] += 1

size_mb = out_csv.stat().st_size/1024/1024
print(f"\n{'='*50}")
print(f"✓ ALL_VRUs.csv")
print(f"  Γραμμές  : {written} VRUs")
print(f"  Columns  : {len(headers)}")
print(f"  Μέγεθος  : {size_mb:.1f} MB")
print(f"\nΑνά κλάση:")
for cls, n in stats.most_common():
    print(f"  {cls:12s}: {n} VRUs")
