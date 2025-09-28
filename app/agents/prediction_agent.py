"""
prediction_agent.py

Prediction Agent for flood forecasting — deterministic + heuristic approach.
Functions:
- predict_flood_risk(flood_data): main entry, returns structured JSON forecast.
"""

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
    """Rough area estimate (km^2) from bbox in degrees (approx). bbox = [minlon,minlat,maxlon,maxlat]
    Uses mean latitude to scale longitude degree length."""
    minlon, minlat, maxlon, maxlat = bbox
    mean_lat = (minlat + maxlat) / 2.0
    # 1 deg lat ~ 111.32 km, 1 deg lon ~ 111.32 * cos(lat)
    lat_km = (maxlat - minlat) * 111.32
    lon_km = (maxlon - minlon) * 111.32 * math.cos(math.radians(mean_lat))
    area = abs(lat_km * lon_km)
    return max(area, 0.001)  # avoid zero

# -----------------------------
# 1) Antecedent Moisture (AMC) estimator
# -----------------------------
def estimate_amc(flood_data):
    """
    Estimate Antecedent Moisture Condition (AMC) on a 0..1 scale.
    Uses:
      - recent historical precipitation (last 7/30 days)
      - soil_moisture observation (if available)
      - landuse/soil (fallback)
    Returns float 0..1 (0=dry, 1=very wet)
    """
    # from weather historical precipitation (last 7 days / 30 days)
    hist = safe_get(flood_data, "data_sources", "weather", "historical")
    sm = safe_get(flood_data, "data_sources", "soil", "smap") or safe_get(flood_data, "data_sources", "soil", "soilgrids")
    # compute simple rainfall totals if present
    recent_total_7 = 0.0
    recent_total_30 = 0.0
    try:
        if hist and "daily" in hist and "precipitation_sum" in hist["daily"]:
            pr = hist["daily"]["precipitation_sum"]
            # assume pr is list of up to 30 days with most recent last
            arr = np.array(pr, dtype=float)
            recent_total_7 = float(arr[-7:].sum()) if arr.size >= 1 else 0.0
            recent_total_30 = float(arr[-30:].sum()) if arr.size >= 1 else recent_total_7
    except Exception:
        recent_total_7 = 0.0
        recent_total_30 = 0.0

    # soil moisture proxy (0..1) if available - Open-Meteo values may be 0..0.5 etc.
    sm_val = None
    try:
        # try common keys
        if sm:
            if isinstance(sm, dict):
                # try expected structure from open-meteo: hourly soil_moisture_0_7cm
                if "hourly" in sm and "soil_moisture_0_7cm" in sm["hourly"]:
                    arr = np.array(sm["hourly"]["soil_moisture_0_7cm"], dtype=float)
                    sm_val = float(np.nanmean(arr))
                elif "smap" in sm and isinstance(sm["smap"], dict):
                    sm_val = float(sm["smap"].get("soil_moisture", np.nan))
                # else attempt to find a numeric field
                else:
                    # flatten numeric leaves
                    def find_number(x):
                        if isinstance(x, (float, int)):
                            return float(x)
                        if isinstance(x, dict):
                            for v in x.values():
                                r = find_number(v)
                                if r is not None:
                                    return r
                        return None
                    sm_val = find_number(sm)
    except Exception:
        sm_val = None

    # Heuristics to map rainfall/soil moisture to AMC
    # Base from rainfall:
    #   recent_total_7 > 100 mm => very wet
    #   50-100 => wet
    #   20-50 => moderate
    #   <20 => dry
    rain_score = 0.0
    if recent_total_7 >= 100:
        rain_score = 1.0
    elif recent_total_7 >= 50:
        rain_score = 0.75
    elif recent_total_7 >= 20:
        rain_score = 0.45
    elif recent_total_7 >= 5:
        rain_score = 0.2
    else:
        rain_score = 0.05

    soil_score = None
    if sm_val is not None:
        # normalize sm_val with plausible bounds (0..0.6)
        soil_score = float(min(max(sm_val / 0.6, 0.0), 1.0))
    else:
        # fallback to default based on landuse
        landuse = safe_get(flood_data, "data_sources", "landuse")
        # if we have no landuse, assume moderately dry
        soil_score = 0.2

    # Weighted average: give rainfall slightly more weight
    amc = 0.6 * rain_score + 0.4 * soil_score
    amc = min(max(amc, 0.0), 1.0)
    return {
        "amc": amc,
        "recent_total_7": recent_total_7,
        "recent_total_30": recent_total_30,
        "soil_moisture_proxy": sm_val
    }

# -----------------------------
# 2) SCS-CN Runoff (approximate)
# -----------------------------
def scs_cn_runoff(precip_mm, cn):
    """
    SCS-CN method to estimate direct runoff depth Q (mm)
      S = (25400 / CN) - 254  # potential retention (mm)
      Q = (P - 0.2*S)^2 / (P + 0.8*S)  if P > 0.2*S else 0
    precip_mm: rainfall depth (mm) for event
    cn: curve number (0..100)
    returns Q_mm
    """
    if precip_mm <= 0:
        return 0.0
    if cn <= 0 or cn >= 100:
        # invalid CN -> fallback moderate
        cn = 75.0
    S = (25400.0 / cn) - 254.0
    Ia = 0.2 * S
    if precip_mm <= Ia:
        return 0.0
    Q = ((precip_mm - Ia) ** 2) / (precip_mm + 0.8 * S)
    return max(Q, 0.0)

def select_curve_number(flood_data):
    """
    Choose a representative Curve Number (CN) 30..98 based on:
      - landuse (urban -> high CN)
      - soil texture (sand -> low CN)
      - impervious surfaces if present
    Fallback default: 75
    """
    # attempt to use landuse/soil
    landuse = safe_get(flood_data, "data_sources", "landuse")
    soil = safe_get(flood_data, "data_sources", "soil", "soilgrids") or safe_get(flood_data, "data_sources", "soil", "soilgrids")
    cn = 75.0  # default
    try:
        # If landuse hints urban or built-up
        if landuse:
            # simple substring checks if landuse string exists
            s = str(landuse).lower()
            if "urban" in s or "built" in s or "impervious" in s:
                cn = 90.0
            elif "cropland" in s or "cropping" in s or "agri" in s:
                cn = 75.0
            elif "forest" in s or "grass" in s or "vegetation" in s:
                cn = 60.0

        # Soil textural fallback
        if soil and isinstance(soil, dict):
            sand = None
            clay = None
            # attempt to read keys
            for k in ("sand", "silt", "clay"):
                if k in soil:
                    pass
            # best effort numeric extraction
            def find_num(d):
                if d is None: return None
                if isinstance(d, (int, float)): return float(d)
                if isinstance(d, dict):
                    for v in d.values():
                        r = find_num(v)
                        if r is not None:
                            return r
                return None
            sand = find_num(soil.get("sand"))
            clay = find_num(soil.get("clay"))
            if sand is not None and clay is not None:
                # sandy soils -> lower CN, clayey -> higher CN
                if sand > clay + 10:
                    cn = max(50.0, cn - 10.0)
                elif clay > sand + 10:
                    cn = min(92.0, cn + 8.0)
    except Exception:
        cn = 75.0
    # clamp
    cn = float(min(max(cn, 30.0), 98.0))
    return cn

# -----------------------------
# 3) Rational method peak discharge (quick)
# -----------------------------
def rational_peak_discharge(runoff_mm, area_km2, tc_hours=1.0, c_runoff_coeff=None):
    """
    Estimate peak discharge using simplified Rational method:
      Q_peak (m3/s) = C * i * A
      where i = intensity (m/hr) over storm duration approximated from runoff depth/runoff_ratio
    Inputs:
      - runoff_mm: depth of direct runoff (mm) (for design storm)
      - area_km2: area in km^2
      - tc_hours: time of concentration (hrs), default 1 hr (caller may tune)
      - c_runoff_coeff: rational runoff coefficient (0..1). If None, approximate as runoff_mm / event_precip_mm (caller may supply)
    Returns Q_peak_m3s
    Note: This is a very rough estimate for quick ranking, not for hydraulic design.
    """
    # Convert area to m^2
    A_m2 = area_km2 * 1e6

    # approximate intensity i (m/hr) — assume storm duration tc_hours and that runoff_mm occurred over that duration
    # convert mm -> m
    runoff_m = runoff_mm / 1000.0
    if tc_hours <= 0:
        tc_hours = 1.0
    i_m_per_hr = runoff_m / tc_hours  # m/hr
    # rational C: estimate from runoff fraction to precipitation if provided (else use 0.5)
    C = c_runoff_coeff if c_runoff_coeff is not None else 0.6
    # Q = C * i * A -> units m/hr * m^2 = m^3/hr -> convert to m^3/s
    Q_m3_per_hr = C * i_m_per_hr * A_m2
    Q_m3_per_s = Q_m3_per_hr / 3600.0
    return max(Q_m3_per_s, 0.0), {"C": C, "i_m_per_hr": i_m_per_hr, "A_m2": A_m2, "tc_hours": tc_hours}

# -----------------------------
# 4) Time-to-peak rough estimate
# -----------------------------
def estimate_time_to_peak(area_km2, slope=0.01):
    """
    Rough time-to-peak (hours) estimate using empirical relationship:
      tc (hours) ≈ 0.0195 * (L^0.77) * (S^-0.385)
    Where L is characteristic basin length in km. For simplicity compute L ~ sqrt(area)
    S is slope (m/m) - if unknown, use 0.01 (1%).
    This is approximate.
    """
    L_km = math.sqrt(max(area_km2, 0.0001))
    S = max(slope, 0.0001)
    tc = 0.0195 * (L_km ** 0.77) * (S ** -0.385)
    # ensure minimum
    return max(tc, 0.1)

# -----------------------------
# 5) Historical anomaly score
# -----------------------------
def precipitation_anomaly_score(flood_data):
    """
    Compare forecast precipitation (next 7 days) vs historical distribution.
    Returns anomaly_score (0..1) where 1 = extremely anomalous (much wetter than usual)
    """
    forecast = safe_get(flood_data, "data_sources", "weather", "open_meteo", "daily", "precipitation_sum")
    hist = safe_get(flood_data, "data_sources", "weather", "historical", "daily", "precipitation_sum")
    try:
        f = np.array(forecast, dtype=float) if forecast is not None else np.zeros(7)
        h = np.array(hist, dtype=float) if hist is not None else np.zeros(30)
        f_total = float(f.sum())
        h_mean = float(np.mean(h)) if h.size > 0 else 0.0
        h_std = float(np.std(h)) if h.size > 0 else 1.0
        # z-score
        if h_std <= 0:
            return 0.0, {"f_total": f_total, "h_mean": h_mean, "h_std": h_std}
        z = (f_total - h_mean) / h_std
        # map z to 0..1 via logistic
        score = 1.0 / (1.0 + math.exp(-0.6 * (z - 1.0)))  # shift to be conservative
        return float(min(max(score, 0.0), 1.0)), {"f_total": f_total, "h_mean": h_mean, "h_std": h_std, "z": z}
    except Exception:
        return 0.0, {}

# -----------------------------
# 6) Final predict_flood_risk wrapper
# -----------------------------
def predict_flood_risk(flood_data, area_km2_override=None):
    """
    Input: flood_data = output of collect_flood_data (dict)
    Output: structured dict with:
      - risk_level (LOW/MEDIUM/HIGH)
      - probability_score (0..1)
      - numbers: runoff_mm, est_peak_m3s, time_to_peak_hours
      - key_factors, recommendations, urdu_summary
      - diagnostics (internal values)
    """
    meta = flood_data.get("meta", {})
    bbox = meta.get("bbox") or safe_get(flood_data, "meta", "bbox") or [0,0,0,0]
    area_km2 = area_km2_override if area_km2_override is not None else bbox_area_km2(bbox)

    # 1) AMC
    amc_info = estimate_amc(flood_data)
    amc = amc_info["amc"]

    # 2) Forecast precipitation (next day / next 24h proxy)
    forecast_precip = safe_get(flood_data, "data_sources", "weather", "open_meteo", "daily", "precipitation_sum") or []
    # total next 1 day and next 3/7 days
    try:
        next1 = float(forecast_precip[0]) if len(forecast_precip) >= 1 else 0.0
    except Exception:
        next1 = 0.0
    try:
        next3 = float(np.sum(forecast_precip[:3])) if len(forecast_precip) >= 3 else float(np.sum(forecast_precip))
    except Exception:
        next3 = next1
    try:
        next7 = float(np.sum(forecast_precip[:7])) if len(forecast_precip) >= 7 else float(np.sum(forecast_precip))
    except Exception:
        next7 = next3

    # 3) Curve Number selection
    cn = select_curve_number(flood_data)

    # 4) Runoff estimate (use worst-case of 24h and 3-day aggregated storms)
    runoff_1d = scs_cn_runoff(next1, cn)
    runoff_3d = scs_cn_runoff(next3, cn)
    runoff_7d = scs_cn_runoff(next7, cn)
    runoff_mm = max(runoff_1d, runoff_3d, runoff_7d)

    # 5) Peak discharge estimate (approx)
    # estimate a reasonable Tc (time of concentration) from area and assume slope from DEM if available
    slope = 0.01  # default 1%
    # try get slope from elevation profile: approximate delta/effective length
    elev_profile = safe_get(flood_data, "data_sources", "elevation", "profile")
    try:
        if elev_profile and isinstance(elev_profile, dict) and "results" in elev_profile:
            elevs = [r.get("elevation") for r in elev_profile.get("results", []) if isinstance(r, dict) and "elevation" in r]
            if elevs:
                elev_range = float(max(elevs) - min(elevs))
                # rough slope = elevation difference / sqrt(area_km2) in meters per km converted to m/m
                L_km = math.sqrt(max(area_km2, 0.0001))
                slope = max(min(elev_range / (L_km * 1000.0), 0.2), 0.001)
    except Exception:
        slope = 0.01

    tc = estimate_time_to_peak(area_km2, slope=slope)
    # rational coefficient estimate: approximate runoff ratio
    event_precip = max(next1, next3, next7, 1e-6)
    runoff_fraction = runoff_mm / event_precip if event_precip > 0 else 0.6
    c_guess = min(max(runoff_fraction, 0.1), 0.95)
    q_peak_m3s, rational_meta = rational_peak_discharge(runoff_mm, area_km2, tc_hours=tc, c_runoff_coeff=c_guess)

    # 6) Precip anomaly
    precip_score, precip_meta = precipitation_anomaly_score(flood_data)

    # 7) Water proximity & infrastructure multiplier
    water_elements = safe_get(flood_data, "data_sources", "water_bodies", "elements") or safe_get(flood_data, "data_sources", "water_bodies")
    water_count = 0
    try:
        if isinstance(water_elements, list):
            water_count = len(water_elements)
    except Exception:
        water_count = 0
    proximity_factor = min(water_count / 10.0, 1.0)  # 0..1

    # 8) Aggregate probability score (weighted)
    # components: precipitation magnitude (normalized), amc, precip anomaly, proximity
    # Normalize precip (use next7, map >100->1)
    precip_norm = min(next7 / 100.0, 1.0)
    score = 0.45 * precip_norm + 0.25 * amc + 0.15 * precip_score + 0.15 * proximity_factor
    score = float(min(max(score, 0.0), 1.0))

    # 9) Risk level thresholds
    if score >= 0.65 or q_peak_m3s > max(50.0, 10.0 * math.sqrt(area_km2)):
        risk_level = "HIGH"
    elif score >= 0.35 or q_peak_m3s > max(10.0, 2.0 * math.sqrt(area_km2)):
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # 10) Recommendations heuristics
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

    # 11) Urdu summary (2 lines)
    urdu = ""
    if risk_level == "HIGH":
        urdu = "اہم: مستقبل قریب میں سیلاب کا سنگین خطرہ ہے۔ کم بلندی اور ندی کے قریب رہنے والوں کو فوراً محفوظ مقام پر منتقل کریں۔"
    elif risk_level == "MEDIUM":
        urdu = "احتیاطی انتباہ: سیلاب کا درمیانہ خطرہ موجود ہے۔ ندی نالوں اور کم بلدی علاقوں کی نگرانی کریں اور نکاسیِ آب صاف رکھیں۔"
    else:
        urdu = "خلاصہ: موجودہ پیشن گوئی کے مطابق سیلاب کا خطرہ کم ہے۔ موسم کی نگرانی جاری رکھیں۔"

    # 12) Compose output
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
            f"Precipitation anomaly score (0-1): {precip_meta.get('f_total') if precip_meta else 'N/A'}"
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

# -----------------------------
# If run as script for quick test
# -----------------------------
if __name__ == "__main__":
    # demo: create a fake flood_data minimal structure
    demo = {
        "meta": {"bbox": [67.0,23.0,69.0,25.0], "area_name": "Karachi"},
        "data_sources": {
            "weather": {
                "open_meteo": {"daily": {"precipitation_sum": [0, 10, 5, 0, 0, 2, 0]}},
                "historical": {"daily": {"precipitation_sum": [0,1,0,2,0,3,5,0,0,1]}}
            },
            "soil": {"smap": {"hourly": {"soil_moisture_0_7cm":[0.12,0.10,0.11]}}},
            "elevation": {},
            "water_bodies": {"elements": [{}, {}, {}]},
            "landuse": "urban"
        }
    }
    res = predict_flood_risk(demo)
    import json
    print(json.dumps(res, indent=2, ensure_ascii=False))
