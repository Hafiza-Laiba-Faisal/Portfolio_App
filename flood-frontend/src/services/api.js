// src/services/api.js
const API_BASE = "http://127.0.0.1:8000"; // Backend host

/**
 * Orchestrate call to Flask agents
 * @param {string} agent_type - "hydro_met" or "evacuation"
 * @param {object} inputData - Input JSON for the agent
 */
export async function orchestrateAction(agent_type, inputData) {
  try {
    const res = await fetch(`${API_BASE}/api/agent_run`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ agent_type, input_data: inputData }),
    });

    if (!res.ok) {
      throw new Error("Failed to call orchestrator");
    }

    return await res.json();
  } catch (err) {
    console.error("API error:", err);
    return { error: err.message };
  }
}

import karachiData from "../new_agents/hydro_met/flood_report_Karachi_Metropolitan_1759022221.json";
import lahoreData from "../new_agents/hydro_met/flood_report_Lahore_District_XXXX.json"; // replace with actual filename
import evacData from "../agents/demo_evacuation.json"; // demo evacuation
import reconData from "../agents/demo_reconstruction.json"; // demo reconstruction

export async function orchestrateAction(userMessage) {
  try {
    const msg = userMessage.trim().toLowerCase();

    if (msg.includes("evacuation")) {
      return evacData || {
        camps: [
          { name: "PDMA Karachi Relief Camp", location: "Karachi", lat: 24.8607, lon: 67.0011 },
          { name: "NDMA Lahore Shelter", location: "Lahore", lat: 31.5204, lon: 74.3587 }
        ]
      };
    }

    if (msg.includes("reconstruction")) {
      return reconData || {
        safe_return_timeline: "Return possible in 1-2 months",
        recommended_actions: [
          "Provide clean water supply",
          "Restore electricity",
          "Increase housing kits allocation",
          "Expand financial aid coverage"
        ],
        updated_aid: { housing_kits: 150, financial_aid: 50000, volunteers: 30 },
        rehab_progress: {
          water_restored: false,
          electricity_restored: false,
          housing_support_pct: 62.5,
          financial_coverage_pct: 41.67
        },
        coordination_notes: [
          "Request satellite verification",
          "Request field survey verification",
          "Coordinate with local councils & housing authorities",
          "Sync with NDMA/PDMA rehabilitation plans"
        ]
      };
    }

    if (msg.includes("karachi")) return karachiData;
    if (msg.includes("lahore")) return lahoreData;

    return { error: "Type 'evacuation', 'reconstruction', 'Lahore', or 'Karachi'" };

  } catch (err) {
    console.error(" API error:", err);
    return { error: err.message };
  }
}
