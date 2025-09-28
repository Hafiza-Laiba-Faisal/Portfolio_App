# prediction.py
import math
import numpy as np
from datetime import datetime

# -----------------------------
# Helpers & Defaults
# -----------------------------
def safe_get(d, *keys, default=None):
    """Nested dict safe getter: safe_get(d, 'a','b') => d['a']['b'] or default"""
    try:
        cur = d
        for k in keys:
            cur = cur[k]
        return cur
    except Exception:
        return default

def bbox_area_km2(bbox):
    """Rough area estimate (km^2) from bbox in degrees (approx). bbox = [minlat, minlon, maxlat, maxlon]"""
    minlat, minlon, maxlat, maxlon = bbox
    mean_lat = (minlat + maxlat) / 2.0
    lat_km = (maxlat - minlat) * 111.32
    lon_km = (maxlon - minlon) * 111.32 * math.cos(math.radians(mean_lat))
    area = abs(lat_km * lon_km)
    return max(area, 0.001)

# -----------------------------
# Antecedent Moisture (AMC) estimator
# -----------------------------
def estimate_amc(flood_data):
    """Estimate Antecedent Moisture Condition (AMC) on a 0..1 scale."""
    hist = safe_get(flood_data, "data_sources", "weather", "historical")
    sm = safe_get(flood_data, "data_sources", "soil", "soil_moisture") or safe_get(flood_data, "data_sources", "soil", "soil_type")
    recent_total_7 = 0.0
    recent_total_30 = 0.0
    try:
        if hist and "daily" in hist and "precipitation_sum" in hist["daily"]:
            pr = hist["daily"]["precipitation_sum"]
            # Filter out invalid values (None, non-numeric)
            pr = [x for x in pr if isinstance(x, (int, float)) and not np.isnan(x)]
            arr = np.array(pr, dtype=float)
            recent_total_7 = float(arr[-7:].sum()) if arr.size >= 1 else 0.0
            recent_total_30 = float(arr[-30:].sum()) if arr.size >= 1 else recent_total_7
    except Exception as e:
        print(f"Error in estimate_amc precipitation parsing: {e}")
        recent_total_7 = 0.0
        recent_total_30 = 0.0
    sm_val = None
    try:
        if sm and isinstance(sm, dict):
            if "hourly" in sm and "soil_moisture_0_to_7cm" in sm["hourly"]:
                arr = np.array([x for x in sm["hourly"]["soil_moisture_0_to_7cm"] if isinstance(x, (int, float)) and not np.isnan(x)], dtype=float)
                sm_val = float(np.nanmean(arr)) if arr.size > 0 else None
            elif "soil_moisture" in sm and isinstance(sm["soil_moisture"], dict):
                sm_val = float(sm["soil_moisture"].get("soil_moisture", np.nan))
                if np.isnan(sm_val): sm_val = None
            else:
                def find_number(x):
                    if isinstance(x, (float, int)) and not np.isnan(x): return float(x)
                    if isinstance(x, dict):
                        for v in x.values():
                            r = find_number(v)
                            if r is not None: return r
                    if isinstance(x, list):
                        valid = [v for v in x if isinstance(v, (float, int)) and not np.isnan(v)]
                        return float(np.mean(valid)) if valid else None
                    return None
                sm_val = find_number(sm)
        # Normalize soil moisture to 0–0.6 range (Open-Meteo typical max)
        if sm_val is not None and sm_val > 0.6:
            sm_val = sm_val / 100.0 if sm_val > 1.0 else sm_val  # Handle possible percentage or large values
    except Exception as e:
        print(f"Error in estimate_amc soil moisture parsing: {e}")
        sm_val = None
    rain_score = 0.0
    if recent_total_7 >= 100: rain_score = 1.0
    elif recent_total_7 >= 50: rain_score = 0.75
    elif recent_total_7 >= 20: rain_score = 0.45
    elif recent_total_7 >= 5: rain_score = 0.2
    else: rain_score = 0.05
    soil_score = float(min(max(sm_val / 0.6, 0.0), 1.0)) if sm_val is not None else 0.2
    amc = 0.6 * rain_score + 0.4 * soil_score
    amc = min(max(amc, 0.0), 1.0)
    return {
        "amc": amc,
        "recent_total_7": recent_total_7,
        "recent_total_30": recent_total_30,
        "soil_moisture_proxy": sm_val if sm_val is not None else "N/A"
    }

# -----------------------------
# SCS-CN Runoff
# -----------------------------
def scs_cn_runoff(precip_mm, cn):
    """SCS-CN method to estimate direct runoff depth Q (mm)"""
    if precip_mm <= 0: return 0.0
    if cn <= 0 or cn >= 100: cn = 75.0
    S = (25400.0 / cn) - 254.0
    Ia = 0.2 * S
    if precip_mm <= Ia: return 0.0
    Q = ((precip_mm - Ia) ** 2) / (precip_mm + 0.8 * S)
    return max(Q, 0.0)

def select_curve_number(flood_data):
    """Choose a representative Curve Number (CN) 30..98"""
    landuse = safe_get(flood_data, "data_sources", "landuse")
    soil = safe_get(flood_data, "data_sources", "soil", "soil_type")
    cn = 75.0
    try:
        if landuse:
            s = str(landuse).lower()
            if "urban" in s or "built" in s or "impervious" in s: cn = 90.0
            elif "cropland" in s or "cropping" in s or "agri" in s: cn = 75.0
            elif "forest" in s or "grass" in s or "vegetation" in s: cn = 60.0
        if soil and isinstance(soil, dict):
            sand = soil.get("sand")
            clay = soil.get("clay")
            if sand is not None and clay is not None:
                if sand > clay + 10: cn = max(50.0, cn - 10.0)
                elif clay > sand + 10: cn = min(92.0, cn + 8.0)
    except Exception:
        cn = 75.0
    cn = float(min(max(cn, 30.0), 98.0))
    return cn

# -----------------------------
# Rational method peak discharge
# -----------------------------
def rational_peak_discharge(runoff_mm, area_km2, tc_hours=1.0, c_runoff_coeff=None):
    """Estimate peak discharge using simplified Rational method"""
    A_m2 = area_km2 * 1e6
    runoff_m = runoff_mm / 1000.0
    if tc_hours <= 0: tc_hours = 1.0
    i_m_per_hr = runoff_m / tc_hours
    C = c_runoff_coeff if c_runoff_coeff is not None else 0.6
    Q_m3_per_hr = C * i_m_per_hr * A_m2
    Q_m3_per_s = Q_m3_per_hr / 3600.0
    return max(Q_m3_per_s, 0.0), {"C": C, "i_m_per_hr": i_m_per_hr, "A_m2": A_m2, "tc_hours": tc_hours}

# -----------------------------
# Time-to-peak estimate
# -----------------------------
def estimate_time_to_peak(area_km2, slope=0.01):
    """Rough time-to-peak (hours) estimate"""
    L_km = math.sqrt(max(area_km2, 0.0001))
    S = max(slope, 0.0001)
    tc = 0.0195 * (L_km ** 0.77) * (S ** -0.385)
    return max(tc, 0.1)

# -----------------------------
# Historical anomaly score
# -----------------------------
def precipitation_anomaly_score(flood_data):
    """Compare forecast precipitation vs historical distribution"""
    forecast = safe_get(flood_data, "data_sources", "weather", "open_meteo", "daily", "precipitation_sum")
    hist = safe_get(flood_data, "data_sources", "weather", "historical", "daily", "precipitation_sum")
    try:
        f = np.array([x for x in (forecast or []) if isinstance(x, (int, float)) and not np.isnan(x)], dtype=float)
        h = np.array([x for x in (hist or []) if isinstance(x, (int, float)) and not np.isnan(x)], dtype=float)
        f_total = float(f.sum()) if f.size > 0 else 0.0
        h_mean = float(np.mean(h)) if h.size > 0 else 0.0
        h_std = float(np.std(h)) if h.size > 0 else 1.0
        if h_std <= 0:
            return 0.0, {"f_total": f_total, "h_mean": h_mean, "h_std": h_std}
        z = (f_total - h_mean) / h_std
        score = 1.0 / (1.0 + math.exp(-0.6 * (z - 1.0)))
        return float(min(max(score, 0.0), 1.0)), {"f_total": f_total, "h_mean": h_mean, "h_std": h_std, "z": z}
    except Exception as e:
        print(f"Error in precipitation_anomaly_score: {e}")
        return 0.0, {"f_total": 0.0, "h_mean": 0.0, "h_std": 1.0, "z": 0.0}

# -----------------------------
# Final predict_flood_risk wrapper
# -----------------------------
def predict_flood_risk(flood_data, area_km2_override=None):
    """Predict flood risk with structured output"""
    meta = flood_data.get("meta", {})
    bbox = meta.get("bbox") or [0,0,0,0]
    area_km2 = area_km2_override if area_km2_override is not None else bbox_area_km2(bbox)
    amc_info = estimate_amc(flood_data)
    amc = amc_info["amc"]
    forecast_precip = safe_get(flood_data, "data_sources", "weather", "open_meteo", "daily", "precipitation_sum") or []
    try:
        next1 = float(forecast_precip[0]) if len(forecast_precip) >= 1 and isinstance(forecast_precip[0], (int, float)) else 0.0
    except Exception:
        next1 = 0.0
    try:
        next3 = float(np.sum([x for x in forecast_precip[:3] if isinstance(x, (int, float)) and not np.isnan(x)])) if len(forecast_precip) >= 3 else next1
    except Exception:
        next3 = next1
    try:
        next7 = float(np.sum([x for x in forecast_precip[:7] if isinstance(x, (int, float)) and not np.isnan(x)])) if len(forecast_precip) >= 7 else next3
    except Exception:
        next7 = next3
    cn = select_curve_number(flood_data)
    runoff_1d = scs_cn_runoff(next1, cn)
    runoff_3d = scs_cn_runoff(next3, cn)
    runoff_7d = scs_cn_runoff(next7, cn)
    runoff_mm = max(runoff_1d, runoff_3d, runoff_7d)
    slope = 0.01
    elev_profile = safe_get(flood_data, "data_sources", "elevation", "profile")
    try:
        if elev_profile and isinstance(elev_profile, dict) and "results" in elev_profile:
            elevs = [r.get("elevation") for r in elev_profile.get("results", []) if isinstance(r, dict) and "elevation" in r]
            if elevs:
                elev_range = float(max(elevs) - min(elevs))
                L_km = math.sqrt(max(area_km2, 0.0001))
                slope = max(min(elev_range / (L_km * 1000.0), 0.2), 0.001)
    except Exception:
        slope = 0.01
    tc = estimate_time_to_peak(area_km2, slope=slope)
    event_precip = max(next1, next3, next7, 1e-6)
    runoff_fraction = runoff_mm / event_precip if event_precip > 0 else 0.6
    c_guess = min(max(runoff_fraction, 0.1), 0.95)
    q_peak_m3s, rational_meta = rational_peak_discharge(runoff_mm, area_km2, tc_hours=tc, c_runoff_coeff=c_guess)
    precip_score, precip_meta = precipitation_anomaly_score(flood_data)
    water_elements = safe_get(flood_data, "data_sources", "water_bodies", "elements") or []
    water_count = len(water_elements) if isinstance(water_elements, list) else 0
    proximity_factor = min(water_count / 10.0, 1.0)
    precip_norm = min(next7 / 100.0, 1.0)
    score = 0.45 * precip_norm + 0.25 * amc + 0.15 * precip_score + 0.15 * proximity_factor
    score = float(min(max(score, 0.0), 1.0))
    if score >= 0.65 or q_peak_m3s > max(50.0, 10.0 * math.sqrt(area_km2)):
        risk_level = "HIGH"
    elif score >= 0.35 or q_peak_m3s > max(10.0, 2.0 * math.sqrt(area_km2)):
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    recs = []
    if risk_level == "HIGH":
        recs = [
            "Immediate alert: heavy rainfall + wet antecedent conditions detected. Prepare evacuations for low-lying/river-adjacent areas.",
            "Monitor river gauges and warn communities near critical water bodies.",
            "Mobilize pumps and temporary flood barriers; clear drains now."
        ]
    elif risk_level == "MEDIUM":
        recs = [
            "Monitor forecasts closely and inspect drainage. Be ready to deploy localized sandbags near low-lying areas.",
            "Check vulnerable infrastructure (bridges, low roads) and move valuables to higher ground."
        ]
    else:
        recs = [
            "Risk low — continue regular monitoring and keep drainage channels clear.",
            "Share public advisory: low risk but watch for sudden changes in forecast."
        ]
    urdu = ""
    if risk_level == "HIGH":
        urdu = "اہم: مستقبل قریب میں سیلاب کا سنگین خطرہ ہے۔ کم بلندی اور ندی کے قریب رہنے والوں کو فوراً محفوظ مقام پر منتقل کریں۔"
    elif risk_level == "MEDIUM":
        urdu = "احتیاطی انتباہ: سیلاب کا درمیانہ خطرہ موجود ہے۔ ندی نالوں اور کم بلدی علاقوں کی نگرانی کریں اور نکاسیِ آب صاف رکھیں۔"
    else:
        urdu = "خلاصہ: موجودہ پیشن گوئی کے مطابق سیلاب کا خطرہ کم ہے۔ موسم کی نگرانی جاری رکھیں۔"
    out = {
        "meta": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "area_name": meta.get("area_name"),
            "bbox": bbox,
            "area_km2": area_km2
        },
        "risk_level": risk_level,
        "probability_score": score,
        "numbers": {
            "forecast_precip_next1_mm": next1,
            "forecast_precip_next3_mm": next3,
            "forecast_precip_next7_mm": next7,
            "curve_number": cn,
            "estimated_runoff_mm": runoff_mm,
            "estimated_peak_discharge_m3s": q_peak_m3s,
            "estimated_time_to_peak_hours": tc
        },
        "key_factors": [
            f"Antecedent moisture (0-1): {amc:.2f}",
            f"Next-7-day precipitation (mm): {next7:.1f}",
            f"Curve Number (CN): {cn:.1f}",
            f"Waterbody proximity factor (0-1): {proximity_factor:.2f}",
            f"Precipitation anomaly score (0-1): {precip_score:.2f}"
        ],
        "recommendations": recs,
        "urdu_summary": urdu,
        "diagnostics": {
            "amc_info": amc_info,
            "precip_meta": precip_meta,
            "rational_meta": rational_meta,
            "runoff_components": {
                "runoff_1d_mm": runoff_1d,
                "runoff_3d_mm": runoff_3d,
                "runoff_7d_mm": runoff_7d
            },
            "water_count": water_count
        }
    }
    return out