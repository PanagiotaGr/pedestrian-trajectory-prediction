"""
build_full_context_dataset_fixed.py
===================================

Διορθώνει το πρόβλημα συγχώνευσης track_id μεταξύ dataset_relative_geometry.csv
και pedestrian_context.csv.

Λειτουργία:
1. Φορτώνει dataset_relative_geometry.csv (ζεύγη αλληλεπιδρασμάτων με sample_id).
2. Χρησιμοποιεί το sample_id για να βρει το member_path από matched_codes.csv.
3. Εξάγει το real_track_id_A από το member_path:
   Π.χ.: "0000_20230322_081506/vrus/000/track.json" -> "0000_20230322_081506/000"
4. Προσαθροίζει (aggregate) τα ζεύγη ανά (real_track_id_A, ts) υπολογίζοντας κοινωνικά
   χαρακτηριστικά (social context features).
5. Συγχωνεύει με pedestrian_context.csv βάσει real_track_id_A == track_id και ts.
6. Αποθηκεύει το τελικό dataset με όλα τα χαρακτηριστικά.

Έξοδος: results/dataset_A_full_context_fixed.csv

Χρήση:
    python build_full_context_dataset_fixed.py

Απαιτήσεις:
    python 3.x, numpy
"""

import csv
import math
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

# ==============================
# Ρυθμίσεις
# ==============================
BASE_DIR = Path(".")  # τρέχον φάκελο (βρίσκεται στο vrus-trajectory-prediction)
RESULTS_DIR = BASE_DIR / "results"

DATASET_RELATIVE_GEOMETRY = RESULTS_DIR / "dataset_relative_geometry.csv"
MATCHED_CODES = RESULTS_DIR / "matched_codes.csv"
PEDESTRIAN_CONTEXT = RESULTS_DIR / "pedestrian_context.csv"  # πρÑ‰ει να υπάρχει
OUTPUT_CSV = RESULTS_DIR / "dataset_A_full_context_fixed.csv"

# -------------------------------------------------------------------------
def extract_real_track_id_from_member_path(member_path: str) -> str:
    """
    Εξάγει το real track ID από το member_path.
    Μορφή: "{scene_folder}/vrus/{track_id}/track.json"
    Επιστρέφει: "{scene_folder}/{track_id}"
    Π.χ.: "0000_20230322_081506/vrus/000/track.json" -> "0000_20230322_081506/000"
    """
    if not member_path:
        return None
    parts = member_path.strip().split("/")
    if len(parts) < 4:
        return None
    # Περίμενουμε: [scene_folder, "vrus", track_id, "track.json"]
    scene_folder = parts[0]
    vrus_dir = parts[1]
    track_id = parts[2]
    # Ελέγχουμε ότι το vrus_dir είναι "vrus" (γοιPeus ασφαλείας)
    if vrus_dir != "vrus":
        # Μπορεί να είναι "vehicles" κλπ., αλλά για τους περίπτωσης pedestrian A πρÑ‰ει vrus.
        pass
    return f"{scene_folder}/{track_id}"


# ==============================
# 1. Φόρτωσηmatched_codes για mapping sample_id -> real track ID
# ==============================
print("Φόρτωση matched_codes.csv...")
sample_to_realtrack = {}
with open(MATCHED_CODES, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        sid = row["sample_id"]
        member_path = row.get("member_path", "")
        real_track_id = extract_real_track_id_from_member_path(member_path)
        if real_track_id:
            sample_to_realtrack[sid] = real_track_id
print(f"  Δημιουργήθηκε mapping για {len(sample_to_realtrack)} sample_id")


# ==============================
# 2. Φόρτωση dataset_relative_geometry.csv και προσθήκη real_track_id_A
# ==============================
print("Φόρτωση dataset_relative_geometry.csv...")
dataset_rows = []
missing_mapping = set()
with open(DATASET_RELATIVE_GEOMETRY, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames.copy()
    for row in reader:
        sample_id = row["sample_id"]
        # Προσθήκη real_track_id_A
        real_track_id = sample_to_realtrack.get(sample_id)
        if not real_track_id:
            missing_mapping.add(sample_id)
            # Θα παραλείψουμε αυτά τα rows στη συγκοπή, ή μπορούμε να προσθέσουμε None
            row["real_track_id_A"] = None
        else:
            row["real_track_id_A"] = real_track_id
        dataset_rows.append(row)

print(f"  Φορτώθηκαν {len(dataset_rows)} γραμμές")
if missing_mapping:
    print(f"  Προσοχή: {len(missing_mapping)} sample_id δεν έχουν mapping (θα παραλειφθούν στη συγκοπή)")


# ==============================
# 3. Συλλογή ομάδων ανά (real_track_id_A, ts)
# ==============================
print("Συσσώρευση δεδομένων ανά (real_track_id_A, ts)...")
groups = defaultdict(list)  # key -> list of rows (each row is a dict)

for row in dataset_rows:
    track_id = row.get("real_track_id_A")
    ts = row["ts"]
    if track_id is None:
        continue  # παραλειφθείσα χωρίς mapping
    key = (track_id, ts)
    groups[key].append(row)

print(f"  Δημιουργήθηκαν {len(groups)} ομάδες (track_id_A, ts)")


# ==============================
# 4. Συναρτήσεις υπολογισμού χαρακτηριστικών
# ==============================
def compute_directional_counts(theta_locals):
    """Υπολογίζει ταξινομημένα κατηγορίες κατεύθυνσης βάσει theta_local ( Radiants )."""
    # Θέτουμε κανόνες: local x-axis είναι κατεύθυνση κίνησης του A.
    # Theta_local: lign to B από A.
    front = 0
    right = 0
    back = 0
    left = 0
    eps = 1e-6
    for th in theta_locals:
        # Κανονικοποίηση στο [-π, π]
        # Κάνουμε εργασία με arctan2 τιμές.
        # Ορίζουμε διαστήματα:
        #   front: -π/4 < th <= π/4  (περίπου μπροστά)
        #   right: π/4 < th <= 3π/4
        #   back: th > 3π/4 ή th <= -3π/4
        #   left: -3π/4 < th <= -π/4
        # Αλλά πρέπει να είμαι προσεκτικός με τα bounds.
        if -math.pi/4 < th <= math.pi/4:
            front += 1
        elif math.pi/4 < th <= 3*math.pi/4:
            right += 1
        elif th > 3*math.pi/4 or th <= -3*math.pi/4:
            back += 1
        elif -3*math.pi/4 < th <= -math.pi/4:
            left += 1
        else:
            # Περίπτωση ακριβώς στα όρια; ας βάλουμε σε ένα cluster
            front += 1
    return {"dir_front": front, "dir_right": right, "dir_back": back, "dir_left": left}


def compute_heading_relation_counts(heading_diffs):
    """Υπολογίζει πλήθος για σχέσεις κεφαλής: similar, opposite, perpendicular."""
    similar = 0
    opposite = 0
    perpendicular = 0
    for hd in heading_diffs:
        # Κανονικοποίηση πολύ μικρή; το heading_diff είναι ήδη σε [-π, π] με wrap_angle
        ah = abs(hd)
        if ah <= math.pi/4:
            similar += 1
        elif ah >= 3*math.pi/4:
            opposite += 1
        else:
            perpendicular += 1
    return {"head_similar": similar, "head_opposite": opposite, "head_perpendicular": perpendicular}


def aggregate_group(rows):
    """
    Υπολογίζει όλα τα συγκεντρωτικά χαρακτηριστικά για μια ομάδα (ίδιο track_id_A, ίδιο ts).
    Επιστρέφει λεξικό με τα χαρακτηριστικά.
    """
    # Εξαγωγή λίστων από τα rows
    track_id_Bs = [r["track_id_B"] for r in rows]
    class_Bs = [r["class_B"] for r in rows]
    dists = [float(r["dist_xy"]) for r in rows]
    closing_speeds = [float(r["closing_speed"]) for r in rows]
    heading_diffs = [float(r["heading_diff"]) for r in rows]
    theta_locals = [float(r["theta_local"]) for r in rows]

    # Βασικά μεγέθη
    total_neighbors = len(rows)
    unique_neighbors = len(set(track_id_Bs))

    # Κατηγορίες κλάσεων (class_B)
    class_counts = Counter(class_Bs)
    # Προσθήκη μετρητών για κάθε πιθανή κλάση. Οι πιθανές κλάσεις όπως在山amento patch.
    # Τυπικές: pedestrian, cyclist, motorcycle, scooter, stroller, wheelchair, unknown.
    # Θα δημιουργήσουμε στήλες:_n_pedestrian, n_cyclist, κλπ.
    class_count_dict = {}
    possible_classes = ["pedestrian", "cyclist", "motorcycle", "scooter", "stroller", "wheelchair", "unknown"]
    for cls in possible_classes:
        class_count_dict[f"n_{cls}"] = class_counts.get(cls, 0)
    # Επίσης μια στήλη με το string των μοναδικών κλάσεων (για debugging)
    class_composition_str = ",".join(sorted(set(class_Bs))) if class_Bs else ""

    # Directional counts (βασισμένα στο theta_local)
    dir_counts = compute_directional_counts(theta_locals)

    # Approaching / receding counts (βάσει closing_speed)
    approaching_count = sum(1 for cs in closing_speeds if cs > 0)
    receding_count = sum(1 for cs in closing_speeds if cs < 0)
    # neutral: cs ~0 ignored

    # Heading relation counts
    head_rel_counts = compute_heading_relation_counts(heading_diffs)

    # Distance statistics
    if dists:
        min_dist = min(dists)
        max_dist = max(dists)
        mean_dist = sum(dists) / len(dists)
        # std
        if len(dists) > 1:
            var = sum((d - mean_dist)**2 for d in dists) / (len(dists)-1)
            std_dist = math.sqrt(var)
        else:
            std_dist = 0.0
    else:
        min_dist = max_dist = mean_dist = std_dist = 0.0

    # Collision risk: επικίνδυνο αν κάποιος neighbor approach με μικρή απόσταση.
    # Ορίζουμε ένα score: max(0, closing_speed / (dist + ε)) μόνο για approaching.
    # Αν κανείς δεν approaching, score = 0.
    risk_scores = []
    for d, cs in zip(dists, closing_speeds):
        if cs > 0:
            risk_scores.append(cs / (d + 1e-6))
    collision_risk = max(risk_scores) if risk_scores else 0.0

    # Interaction intensity: συνολική "ένταση" απόστασης. Χρησιμοποιούμε sum(1/(d+ε)).
    intensity = sum(1.0 / (d + 1e-6) for d in dists)

    # Επιπλέον: πλήθος διαφορετικών track_id_B (unique neighbors) ήδη δίνεται ως unique_neighbors.
    # Θα μπορούσε να χρησιμοποιηθεί ως total_neighbors (ή και distinct). Εδώ θα κρατήσουμε total_neighbors (πλήθος εμφανίσεων στους διαφορετικούs timestamp? Όχι, αυτή η ομάδα είναι για ένα συγκεκριμένο timestamp. Άρα total_neighbors = πλήθος γειτόνων που υπάρχουν ακριβώς σε αυτό το timestamp. Δistinct neighbors στο timestamp? Πιθανώς duplicate δεν υπάρχει για ίδιο track_id_B στο ίδιο timestamp? Μπορεί να υπάρχουν πολλές γραμμές για το ίδιο ζευγάρι (track_id_A, track_id_B) στο ίδιο timestamp; Μάλλον όχι, το dataset_relative_geometry είναι per timestamp per pair, οπότε κάθε (A,B,ts) θα υπάρχει μία φορά. Άρα total_neighbors = πλήθος διαφορετικών B που συνυπάρχουν με A στο ts.

    # Άρα unique_neighbors = total_neighbors (για αυτή τη ομάδα). Αν όμως υπάρχουν duplicates (πιθανό λόγω κάποιας θεωρίας), κρατάμε το unique.

    # Δημιουργία γραμμής εξόδου
    # Παίρνουμε τα metadata από το πρώτο row
    first = rows[0]
    out_row = {
        "track_id_A": first["real_track_id_A"],
        "ts": first["ts"],
        "split": first["split"],
        "class_A": first["class_A"],
        # Social context features
        "total_neighbors": unique_neighbors,
        "class_composition": class_composition_str,
        "dir_front": dir_counts["dir_front"],
        "dir_right": dir_counts["dir_right"],
        "dir_back": dir_counts["dir_back"],
        "dir_left": dir_counts["dir_left"],
        "approaching_count": approaching_count,
        "receding_count": receding_count,
        "head_similar": head_rel_counts["head_similar"],
        "head_opposite": head_rel_counts["head_opposite"],
        "head_perpendicular": head_rel_counts["head_perpendicular"],
        "min_dist": round(min_dist, 4),
        "mean_dist": round(mean_dist, 4),
        "max_dist": round(max_dist, 4),
        "std_dist": round(std_dist, 4),
        "collision_risk": round(collision_risk, 4),
        "interaction_intensity": round(intensity, 4),
    }
    # Προσθήκη counts για κάθε κλάση (για ευκολία)
    out_row.update({k: int(v) for k, v in class_count_dict.items()})

    return out_row


# ==============================
# 5. Συγκέντρωση όλων των ομάδων
# ==============================
print("Συναθροισμός χαρακτηριστικών...")
aggregated_rows = []
for key, rows in groups.items():
    agg = aggregate_group(rows)
    aggregated_rows.append(agg)

print(f"  Δημιουργήθηκαν {len(aggregated_rows):,} συγκεντρωμένες γραμμές")


# ==============================
# 6. Φόρτωση pedestrian_context.csv
# ==============================
print(f"Φόρτωση {PEDESTRIAN_CONTEXT.name}...")
if not PEDESTRIAN_CONTEXT.exists():
    print(f"  ΣΦΑΛΜΑ: Δεν βρέθηκε το αρχείο {PEDESTRIAN_CONTEXT}")
    print("  Βεβαιώσου ότι το αρχείο υπάρχει στο results/")
    exit(1)

with open(PEDESTRIAN_CONTEXT, newline="", encoding="utf-8") as f:
    context_reader = csv.DictReader(f)
    context_fieldnames = context_reader.fieldnames
    context_data = list(context_reader)

print(f"  Φορτώθηκαν {len(context_data)} γραμμές pedestrian context")

# Δημιουργία πίνακα για γρήγορη ανάκτηση: (track_id, ts) -> row dict
context_lookup = {}
for row in context_data:
    tid = row["track_id"]
    ts = row["ts"]
    context_lookup[(tid, ts)] = row

print(f"  Δημιουργήθηκε index για {len(context_lookup)} entries")


# ==============================
# 7. Συγχώνευση (merge) των aggregated rows με pedestrian_context
# ==============================
print("Συγχώνευση aggregated data με pedestrian context...")
merged_rows = []
matched_count = 0
unmatched_count = 0

for agg in aggregated_rows:
    track_id = agg["track_id_A"]
    ts = agg["ts"]
    key = (track_id, ts)
    ctx = context_lookup.get(key)
    if ctx:
        # Συγχώνευση: προσθήκη όλων των πεδίων από το context (εκτός από track_id και ts που υπάρχουν ήδη)
        merged = agg.copy()
        for k, v in ctx.items():
            if k not in ("track_id", "ts"):  # αποφυγή διπλοτύπου
                merged[k] = v
        matched_count += 1
    else:
        # Χωρίς context: προσθέτει κενά για τα πεδία του context (érés None)
        merged = agg.copy()
        # Προσθήκη κενών πεδίων για όσα πεδία υπάρχουν στο context εκτός track_id, ts
        # Δεν τα ξέρουμε όλα αυτάματα, αλλά μπορούμε να βρούμε τα πεδία από το context_fieldnames
        for col in context_fieldnames:
            if col not in ("track_id", "ts"):
                merged[col] = None  # ή ""
        unmatched_count += 1
    merged_rows.append(merged)

print(f"  Συγχωνεύθηκαν (matched): {matched_count}")
print(f"  Δεν βρέθηκε context: {unmatched_count}")


# ==============================
# 8. Αποθήκευση τελικού CSV
# ==============================
print(f"Αποθήκευση σε {OUTPUT_CSV.name}...")
if not merged_rows:
    print("  Δεν υπάρχουν γραμμές για αποθήκευση!")
    exit(1)

# Συλλογή όλων των fieldnames: όλα τα κλειδιά από το πρώτο merged row
fieldnames = list(merged_rows[0].keys())

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fout:
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(merged_rows)

print(f"  Αποθηκεύτηκαν {len(merged_rows):,} γραμμές")


# ==============================
# 9. Στατιστικά και επικύρωση
# ==============================
print("\n" + "="*55)
print("ΕΠΙΚΥΡΩΣΗ ΚΑΙ ΣΤΑΤΙΣΤΙΚΑ")
print("="*55)

# Φόρτωση ξανα του τελικού CSV για στατιστικά (ή χρήση merged_rows)
total_saved = len(merged_rows)
rows_loaded = len(dataset_rows)
matched_pct = (matched_count / total_saved * 100) if total_saved else 0
unmatched_pct = (unmatched_count / total_saved * 100) if total_saved else 0

print(f"Γραμμές φορτωμένες (dataset_relative_geometry): {rows_loaded:,}")
print(f"Γραμμές σωζούμενες (τελικό dataset): {total_saved:,}")
print(f"ΣυνοLOCK matched context rows: {matched_count:,} ({matched_pct:.1f}%)")
print(f"Μη-αναγνωρίσιμες (unmatched): {unmatched_count:,} ({unmatched_pct:.1f}%)")

# Πόσοι on_crosswalk (από pedestrian_context)
on_crosswalk_count = 0
red_light_count = 0
green_light_count = 0
total_with_context = 0

# Αρίθμηση κλάσεων VRU από το class_A
class_totals = Counter()
avg_neighbors_sum = 0
avg_neighbors_cnt = 0

for row in merged_rows:
    cls = row.get("class_A")
    if cls:
        class_totals[cls] += 1
    # neighbors count (social)
    try:
        neigh = float(row.get("total_neighbors", 0))
        avg_neighbors_sum += neigh
        avg_neighbors_cnt += 1
    except:
        pass
    # on_crosswalk: maybe from pedestrian_context.on_crosswalk (0/1) or from on_crosswalk column
    if "on_crosswalk" in row and row["on_crosswalk"] is not None:
        try:
            if int(row["on_crosswalk"]) == 1:
                on_crosswalk_count += 1
        except:
            pass
        total_with_context += 1
    # Traffic lights: f1_state ή f1; assume f1_state contains state ID (e.g., 10=red, 4=green)
    # Έλεγχος και για f1_state και f1 (αν υπάρχει)
    for fld in ("f1_state", "f1"):
        if fld in row and row[fld] is not None:
            try:
                val = int(row[fld])
                if val == 4:  # green
                    green_light_count += 1
                elif val == 10:  # red
                    red_light_count += 1
            except:
                pass
        # Μόνο μια φορά ανά γραμμή; θα μπορούσε και να υπάρχουν πολλαπλά φανάρια αλλά συνήθως ένα.
        # Αν τα μετράμε ξεχωριστά, το πράγμα είναι να δώσουμε ποσοστά.
        # Αν τα μετράμε σε όλα τα rows που έχουν f1_state, μπορούμε να τυπώσουμε ποσοστά από το σύνολο ή από τα rows με φως πληροφορία.
        # Χρήση: % με κόκκινο ή πράσινο φως ανά τα σετ που έχουν φωταgressive info.
        break  # έλεγχος μόνο του first available

# Υπολογισμός ποσοστών
on_cw_pct = (on_crosswalk_count / total_saved * 100) if total_saved else 0

# Για φανάρια, μπορούμε να υπολογίσουμε ποσοστό βάσει των rows που έχουν.status info.
# Θα χρησιμοποιήσουμε ως μ ολο τα rows που έχουν f1_state/f1 διαθέσιμο (δηλαδή row.get("f1_state") not None).
total_with_light = 0
for row in merged_rows:
    if row.get("f1_state") is not None or row.get("f1") is not None:
        total_with_light += 1
red_pct = (red_light_count / total_with_light * 100) if total_with_light else 0
green_pct = (green_light_count / total_with_light * 100) if total_with_light else 0

print("\nΠοσοστά:")
print(f"  On crosswalk: {on_crosswalk_count}/{total_saved} ({on_cw_pct:.1f}%)")
if total_with_light:
    print(f"  Red lights:   {red_light_count}/{total_with_light} ({red_pct:.1f}%)")
    print(f"  Green lights: {green_light_count}/{total_with_light} ({green_pct:.1f}%)")
else:
    print("  Δεν βρέθηκαν πληροφορίες φαναριών.")

print("\nΚλάσεις VRU (από class_A):")
for cls, cnt in class_totals.most_common():
    pct = cnt / total_saved * 100 if total_saved else 0
    print(f"  {cls:<15}: {cnt:>6} ({pct:.1f}%)")

avg_neighbors = avg_neighbors_sum / avg_neighbors_cnt if avg_neighbors_cnt else 0
print(f"\nΜέσος αριθμός γειτόνων (social context total_neighbors): {avg_neighbors:.2f}")

print("\n" + "="*55)
print("ΤΕΛΟΣ ΕΠΙΚΥΡΩΣΗΣ")
print("="*55)
