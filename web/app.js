/*
 * Amsterdam Airbnb segmentation — frontend logic.
 *
 * Loads:
 *   data.json              — clusters + neighbourhood breakdowns + centroids
 *   neighbourhoods.geojson — 22 polygons of Amsterdam
 *
 * Renders:
 *   • a Leaflet map with each neighbourhood coloured by its dominant cluster
 *   • a side panel with the cluster legend + per-neighbourhood detail
 *   • a "find your segment" form that does a nearest-centroid lookup in JS
 */

const COLOR_VARS = ["--c0", "--c1", "--c2", "--c3", "--c4", "--c5", "--c6", "--c7"];

const state = {
  data: null,       // data.json contents
  geo: null,        // geojson FeatureCollection
  map: null,
  geoLayer: null,
  selectedCluster: null,  // for legend filtering / highlight
  selectedNeigh: null,
  pickedLatLng: null,     // location chosen on the map for "find your segment"
  pickMarker: null,
};

// amenities is no longer asked in the form (not meaningful to a user) — use a
// sensible typical value so the centroid match still works.
const DEFAULT_AMENITIES = 28;
const CITY_CENTER = { lat: 52.370, lng: 4.895 };

const $ = (id) => document.getElementById(id);
const fmtPrice = (v) => `€${Math.round(v)}`;
const fmtPct = (v) => `${Math.round(v * 100)}%`;

function colorForCluster(cid) {
  const v = COLOR_VARS[cid % COLOR_VARS.length];
  return getComputedStyle(document.documentElement).getPropertyValue(v).trim() || "#888";
}

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------
async function load() {
  const [data, geo] = await Promise.all([
    fetch("data.json").then((r) => r.json()),
    fetch("neighbourhoods.geojson").then((r) => r.json()),
  ]);
  state.data = data;
  state.geo = geo;
  // for each segment, find the highest share it reaches in any neighbourhood,
  // so the map can shade relative to that peak (small segments stay readable)
  state.maxShareByCluster = {};
  for (const c of data.clusters) {
    let mx = 0;
    for (const name in data.neighbourhoods) {
      const nb = data.neighbourhoods[name];
      const cnt = (nb.counts && nb.counts[c.id]) ? nb.counts[c.id] : 0;
      const share = nb.total ? cnt / nb.total : 0;
      if (share > mx) mx = share;
    }
    state.maxShareByCluster[c.id] = mx;
  }
  initLegend();
  initMap();
  initPredict();
}

// ---------------------------------------------------------------------------
// legend (cluster list on the right)
// ---------------------------------------------------------------------------
function initLegend() {
  const legend = $("legend");
  const clusters = [...state.data.clusters]
    .sort((a, b) => a.median_price - b.median_price);

  for (const c of clusters) {
    const row = document.createElement("div");
    row.className = "legend-item";
    row.dataset.cluster = c.id;
    row.innerHTML = `
      <span class="legend-swatch" style="background:${colorForCluster(c.id)}"></span>
      <span class="legend-label">${c.label}</span>
      <span class="legend-meta">n=${c.n.toLocaleString()} · ${fmtPrice(c.median_price)}</span>
    `;
    row.addEventListener("click", () => toggleClusterFocus(c.id));
    legend.appendChild(row);
  }
}

function toggleClusterFocus(cid) {
  if (state.selectedCluster === cid) {
    clearClusterFocus();
  } else {
    applyClusterFocus(cid);
  }
}

function applyClusterFocus(cid) {
  state.selectedCluster = cid;
  document.querySelectorAll(".legend-item").forEach((el) => {
    el.classList.toggle("active", Number(el.dataset.cluster) === cid);
  });
  state.geoLayer.eachLayer((lyr) => state.geoLayer.resetStyle(lyr));
  state.geoLayer.setStyle(neighStyle);
  renderClusterDetail(cid);
}

function clearClusterFocus() {
  state.selectedCluster = null;
  document.querySelectorAll(".legend-item").forEach((el) => el.classList.remove("active"));
  state.geoLayer.eachLayer((lyr) => state.geoLayer.resetStyle(lyr));
  state.geoLayer.setStyle(neighStyle);
  $("cluster-detail-block").hidden = true;
}

function renderClusterDetail(cid) {
  const c = state.data.clusters.find((x) => x.id === cid);
  if (!c) return;
  const propEntries = Object.entries(c.top_property_types).slice(0, 3);
  const neighEntries = Object.entries(c.top_neighbourhoods).slice(0, 4);
  $("cluster-detail").innerHTML = `
    <div><strong style="color:${colorForCluster(c.id)}">${c.label}</strong>
         &nbsp;<span style="color:var(--ink-soft); font-size:0.85rem;">
         ${c.n.toLocaleString()} listings · ${fmtPct(c.share)} of market
         </span></div>
    <div style="font-size:0.8rem; color:var(--ink-soft); margin-top:0.25rem;">
      On the map: darker areas have more of this type.
    </div>
    <div class="stats">
      <div><span>Typical price</span><span>${fmtPrice(c.median_price)}</span></div>
      <div><span>Most are</span><span>${fmtPrice(c.p25_price)}–${fmtPrice(c.p75_price)}</span></div>
      <div><span>Guests</span><span>${c.median_accommodates}</span></div>
      <div><span>Bedrooms</span><span>${c.median_bedrooms}</span></div>
      <div><span>Beds</span><span>${c.median_beds}</span></div>
    </div>
    <div style="margin-top:0.4rem;">
      <span style="color:var(--ink-soft); font-size:0.85rem;">Top property types:</span><br/>
      ${propEntries.map(([k, v]) => `<span style="display:inline-block; margin:2px 4px 2px 0; padding:2px 7px; background:#f3f4f6; border-radius:3px; font-size:0.83rem;">${k} <em style="color:var(--ink-soft); font-style:normal;">${v}</em></span>`).join("")}
    </div>
    <div style="margin-top:0.4rem;">
      <span style="color:var(--ink-soft); font-size:0.85rem;">Where it concentrates:</span><br/>
      ${neighEntries.map(([k, v]) => `<span style="display:inline-block; margin:2px 4px 2px 0; padding:2px 7px; background:#eef2ff; border-radius:3px; font-size:0.83rem;">${k} <em style="color:var(--ink-soft); font-style:normal;">${v}</em></span>`).join("")}
    </div>
  `;
  $("cluster-detail-block").hidden = false;
}

// ---------------------------------------------------------------------------
// map
// ---------------------------------------------------------------------------
function initMap() {
  state.map = L.map("map", { zoomSnap: 0.5 }).setView([52.367, 4.895], 12);
  // drop Leaflet's default "🇺🇦 Leaflet" prefix — keep a plain credit
  state.map.attributionControl.setPrefix(
    '<a href="https://leafletjs.com" target="_blank" rel="noopener">Leaflet</a>',
  );
  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
    subdomains: "abcd",
    maxZoom: 18,
  }).addTo(state.map);

  // click anywhere on the map to set the location used by "find your segment"
  state.map.on("click", (e) => setPickedLocation(e.latlng));

  state.geoLayer = L.geoJSON(state.geo, {
    style: neighStyle,
    onEachFeature: (feature, layer) => {
      const name = feature.properties.neighbourhood;
      const nb = state.data.neighbourhoods[name];
      const labelById = Object.fromEntries(state.data.clusters.map((c) => [c.id, c.label]));
      const tooltip = nb
        ? `<strong>${name}</strong><br/>${nb.total} listings · median ${fmtPrice(nb.median_price)}<br/>dominant: ${labelById[nb.dominant_cluster]}`
        : `<strong>${name}</strong><br/>no listings in dataset`;
      layer.bindTooltip(tooltip, { sticky: true });
      layer.on({
        mouseover: (e) => {
          e.target.setStyle({ weight: 2.5, color: "#1f2937", fillOpacity: 0.85 });
        },
        mouseout: (e) => state.geoLayer.resetStyle(e.target),
        click: () => selectNeigh(name),
      });
    },
  }).addTo(state.map);

  // tighten zoom to the data
  state.map.fitBounds(state.geoLayer.getBounds(), { padding: [8, 8] });
}

function neighStyle(feature) {
  const name = feature.properties.neighbourhood;
  const nb = state.data.neighbourhoods[name];
  if (!nb) {
    return { color: "#bbb", weight: 1, fillColor: "#e2e2e2", fillOpacity: 0.35 };
  }
  const focused = state.selectedCluster;

  // default view: colour each area by its most common segment
  if (focused === null) {
    return {
      color: "#5b6470",
      weight: 1,
      fillColor: colorForCluster(nb.dominant_cluster),
      fillOpacity: 0.72,
    };
  }

  // focused view: shade every area by HOW MUCH of the chosen segment it has,
  // so even small segments (hotel/shared, spacious private) show where they live
  const total = nb.total || 0;
  const count = (nb.counts && nb.counts[focused]) ? nb.counts[focused] : 0;
  const share = total ? count / total : 0;
  // shade relative to this segment's peak share, so small segments read clearly
  const peak = state.maxShareByCluster[focused] || 1;
  const rel = peak > 0 ? share / peak : 0;
  return {
    color: share > 0 ? "#5b6470" : "#cfd5df",
    weight: share > 0 ? 1 : 0.5,
    fillColor: colorForCluster(focused),
    fillOpacity: share === 0 ? 0.05 : 0.18 + rel * 0.67,
  };
}

function selectNeigh(name) {
  state.selectedNeigh = name;
  const nb = state.data.neighbourhoods[name];
  const detail = $("neigh-detail");
  if (!nb) {
    detail.className = "neigh-detail empty";
    detail.textContent = `${name}: no listings in the dataset.`;
    return;
  }
  detail.className = "neigh-detail";
  const labelById = Object.fromEntries(state.data.clusters.map((c) => [c.id, c.label]));

  // composition bar
  const total = nb.total;
  const segs = Object.entries(nb.counts)
    .map(([cid, count]) => ({ cid: Number(cid), count, share: count / total }))
    .filter((s) => s.count > 0)
    .sort((a, b) => b.count - a.count);

  const bar = segs.map((s) =>
    `<div title="${labelById[s.cid]}: ${s.count} (${fmtPct(s.share)})"
          style="width:${s.share * 100}%; background:${colorForCluster(s.cid)}"></div>`,
  ).join("");

  const legendBits = segs.map((s) =>
    `<span><span style="display:inline-block; width:9px; height:9px; background:${colorForCluster(s.cid)}; border-radius:2px; margin-right:3px;"></span>${labelById[s.cid]} ${fmtPct(s.share)}</span>`,
  ).join("");

  detail.innerHTML = `
    <div><strong>${name}</strong>
         &nbsp;<span style="color:var(--ink-soft); font-size:0.85rem;">
         ${total} listings · median ${fmtPrice(nb.median_price)}
         </span></div>
    <div style="font-size:0.85rem; color:var(--ink-soft); margin-top:0.2rem;">
      Dominant segment: <strong style="color:${colorForCluster(nb.dominant_cluster)}">
      ${labelById[nb.dominant_cluster]}</strong>
    </div>
    <div class="neigh-bar">${bar}</div>
    <div class="neigh-bar-legend">${legendBits}</div>
  `;
}

// ---------------------------------------------------------------------------
// nearest-centroid predict — only structural features used in clustering
// ---------------------------------------------------------------------------
function stratumOf(roomType) {
  for (const [name, types] of Object.entries(state.data.stratum_room_types)) {
    if (types.includes(roomType)) return name;
  }
  return "specialty";
}

function predictCluster(input) {
  const stratum = stratumOf(input.roomType);
  const cents = state.data.centroids_by_stratum[stratum];
  if (!cents) {
    // tiny specialty stratum has no k-means — return the single specialty cluster
    const specialty = state.data.clusters.find((c) => c.stratum === "specialty");
    return specialty;
  }
  const order = cents.feature_order;
  const x = order.map((feat) => {
    switch (feat) {
      case "accommodates":    return input.accommodates;
      case "bedrooms":        return input.bedrooms;
      case "beds":            return input.beds;
      case "bathrooms_n":     return input.bathrooms;
      case "amenities_count": return input.amenities;
      case "latitude":        return input.lat;
      case "longitude":       return input.lon;
      default: return 0;
    }
  });
  // scale: (x - mean) / scale
  const xs = x.map((v, i) => (v - cents.scaler_mean[i]) / cents.scaler_scale[i]);
  // also need centroids in scaled space; we stored them in raw space, so re-scale
  let bestIdx = 0;
  let bestDist = Infinity;
  for (let i = 0; i < cents.centroids_raw.length; i++) {
    const c = cents.centroids_raw[i];
    let d = 0;
    for (let j = 0; j < c.length; j++) {
      const cs = (c[j] - cents.scaler_mean[j]) / cents.scaler_scale[j];
      const diff = xs[j] - cs;
      d += diff * diff;
    }
    if (d < bestDist) { bestDist = d; bestIdx = i; }
  }
  const globalId = cents.global_id_offset + bestIdx;
  return state.data.clusters.find((c) => c.id === globalId);
}

// remember a clicked map point and show a marker for it
function setPickedLocation(latlng) {
  state.pickedLatLng = latlng;
  if (state.pickMarker) {
    state.pickMarker.setLatLng(latlng);
  } else {
    state.pickMarker = L.marker(latlng).addTo(state.map);
  }
  const el = $("loc-indicator");
  if (el) {
    el.classList.add("set");
    el.textContent = "Location set ✓  (click the map again to move it)";
  }
}

function initPredict() {
  $("predict-btn").addEventListener("click", () => {
    const loc = state.pickedLatLng || CITY_CENTER;
    const input = {
      roomType: $("in-room").value,
      accommodates: Number($("in-acc").value),
      bedrooms: Number($("in-bed").value),
      beds: Number($("in-beds").value),
      bathrooms: Number($("in-bath").value),
      amenities: DEFAULT_AMENITIES,
      lat: loc.lat,
      lon: loc.lng,
    };
    const c = predictCluster(input);
    const result = $("predict-result");
    result.hidden = false;
    result.innerHTML = `
      <strong style="color:${colorForCluster(c.id)}">${c.label}</strong>
      Typically ${fmtPrice(c.median_price)} / night
      &nbsp;(most are ${fmtPrice(c.p25_price)}–${fmtPrice(c.p75_price)}).
      <div style="margin-top:0.3rem; font-size:0.8rem; color:var(--ink-soft);">
        Highlighted on the map — darker areas have more of this type.
      </div>
    `;
    // show where this segment lives on the map
    applyClusterFocus(c.id);
  });
}

load().catch((err) => {
  console.error(err);
  document.body.insertAdjacentHTML(
    "afterbegin",
    `<div style="padding:1rem; background:#fee2e2; color:#7f1d1d;">
       Failed to load data — open this from a static server, not file:// (Chrome blocks fetch on local files).
       Run: <code>python -m http.server</code> in the <code>web/</code> folder.
     </div>`,
  );
});
