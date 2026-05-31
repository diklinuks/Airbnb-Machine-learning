/* Amsterdam Airbnb segmentation — portfolio redesign
 *
 * Loads:   web/data.json + web/neighbourhoods.geojson
 * Renders: animated hero, Leaflet map, segment cards, Chart.js charts, predict form
 */

const COLOR_VARS = ["--c0", "--c1", "--c2", "--c3", "--c4", "--c5", "--c6", "--c7"];
const DEFAULT_AMENITIES = 28;
const CITY_CENTER = { lat: 52.370, lng: 4.895 };

const state = {
  data: null,
  geo: null,
  map: null,
  geoLayer: null,
  selectedCluster: null,
  pickedLatLng: null,
  pickMarker: null,
  maxShareByCluster: {},
};

const $ = (id) => document.getElementById(id);
const fmtPrice = (v) => `€${Math.round(v)}`;
const fmtPct = (v) => `${Math.round(v * 100)}%`;
const colorForCluster = (cid) =>
  getComputedStyle(document.documentElement)
    .getPropertyValue(COLOR_VARS[cid % COLOR_VARS.length]).trim() || "#888";

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------
async function boot() {
  const [data, geo] = await Promise.all([
    fetch("data.json").then((r) => r.json()),
    fetch("neighbourhoods.geojson").then((r) => r.json()),
  ]);
  state.data = data;
  state.geo = geo;

  // pre-compute the peak share each segment reaches in any neighbourhood,
  // so the focused-map shading stays readable for small segments
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

  initRevealOnScroll();
  initCounters();
  initLegend();
  initSegmentCards();
  initMap();
  initCharts();
  initPredict();
}

// ---------------------------------------------------------------------------
// scroll reveals — fade + lift each .reveal as it enters the viewport
// ---------------------------------------------------------------------------
function initRevealOnScroll() {
  const els = document.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window)) {
    els.forEach((el) => el.classList.add("in"));
    return;
  }
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e, i) => {
      if (e.isIntersecting) {
        // stagger siblings a little for polish
        setTimeout(() => e.target.classList.add("in"), i * 60);
        io.unobserve(e.target);
      }
    });
  }, { threshold: 0.12 });
  els.forEach((el) => io.observe(el));
}

// ---------------------------------------------------------------------------
// hero stat counters — ease 0 → target once visible
// ---------------------------------------------------------------------------
function initCounters() {
  const els = document.querySelectorAll(".stat .num");
  const animate = (el) => {
    const target = Number(el.dataset.count);
    const prefix = el.dataset.prefix || "";
    const start = performance.now();
    const dur = 1200;
    const tick = (t) => {
      const p = Math.min(1, (t - start) / dur);
      const eased = 1 - Math.pow(1 - p, 3);
      const val = Math.round(target * eased);
      el.textContent = prefix + val.toLocaleString();
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };
  if (!("IntersectionObserver" in window)) { els.forEach(animate); return; }
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) { animate(e.target); io.unobserve(e.target); }
    });
  }, { threshold: 0.4 });
  els.forEach((el) => io.observe(el));
}

// ---------------------------------------------------------------------------
// legend
// ---------------------------------------------------------------------------
function initLegend() {
  const legend = $("legend");
  legend.innerHTML = "";
  const clusters = [...state.data.clusters].sort((a, b) => a.median_price - b.median_price);
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
  if (state.selectedCluster === cid) clearClusterFocus();
  else applyClusterFocus(cid);
}

function applyClusterFocus(cid) {
  state.selectedCluster = cid;
  document.querySelectorAll(".legend-item").forEach((el) =>
    el.classList.toggle("active", Number(el.dataset.cluster) === cid),
  );
  document.querySelectorAll(".seg-card").forEach((el) =>
    el.classList.toggle("active", Number(el.dataset.cluster) === cid),
  );
  state.geoLayer.eachLayer((lyr) => state.geoLayer.resetStyle(lyr));
  state.geoLayer.setStyle(neighStyle);
  renderClusterDetail(cid);
}

function clearClusterFocus() {
  state.selectedCluster = null;
  document.querySelectorAll(".legend-item").forEach((el) => el.classList.remove("active"));
  document.querySelectorAll(".seg-card").forEach((el) => el.classList.remove("active"));
  state.geoLayer.eachLayer((lyr) => state.geoLayer.resetStyle(lyr));
  state.geoLayer.setStyle(neighStyle);
  $("cluster-detail-block").hidden = true;
}

function renderClusterDetail(cid) {
  const c = state.data.clusters.find((x) => x.id === cid);
  if (!c) return;
  const props = Object.entries(c.top_property_types).slice(0, 3);
  const neighs = Object.entries(c.top_neighbourhoods).slice(0, 4);
  $("cluster-detail").innerHTML = `
    <div><strong style="color:${colorForCluster(c.id)}">${c.label}</strong>
      <span style="color:var(--ink-soft); font-size:0.85rem;">
        · ${c.n.toLocaleString()} listings · ${fmtPct(c.share)} of market
      </span>
    </div>
    <div style="font-size:0.8rem; color:var(--ink-soft); margin-top:0.3rem;">
      Map is now shaded by share of this segment.
    </div>
    <div class="stats-grid">
      <div><span>Typical price</span><span>${fmtPrice(c.median_price)}</span></div>
      <div><span>Most are</span><span>${fmtPrice(c.p25_price)}–${fmtPrice(c.p75_price)}</span></div>
      <div><span>Guests</span><span>${c.median_accommodates}</span></div>
      <div><span>Bedrooms</span><span>${c.median_bedrooms}</span></div>
      <div><span>Beds</span><span>${c.median_beds}</span></div>
      <div><span>Bathrooms</span><span>${c.median_bathrooms}</span></div>
    </div>
    <div style="margin-top:0.6rem;">
      <div style="font-size:0.75rem; color:var(--ink-soft); text-transform:uppercase; letter-spacing:0.1em;">Top property types</div>
      ${props.map(([k, v]) => `<span class="chip">${k}<em>${v.toLocaleString()}</em></span>`).join("")}
    </div>
    <div style="margin-top:0.6rem;">
      <div style="font-size:0.75rem; color:var(--ink-soft); text-transform:uppercase; letter-spacing:0.1em;">Where it concentrates</div>
      ${neighs.map(([k, v]) => `<span class="chip">${k}<em>${v.toLocaleString()}</em></span>`).join("")}
    </div>
  `;
  $("cluster-detail-block").hidden = false;
}

// ---------------------------------------------------------------------------
// segment cards
// ---------------------------------------------------------------------------
function initSegmentCards() {
  const wrap = $("segment-cards");
  const all = [...state.data.clusters].sort((a, b) => a.median_price - b.median_price);
  const maxP75 = Math.max(...all.map((c) => c.p75_price));
  for (const c of all) {
    const color = colorForCluster(c.id);
    const card = document.createElement("div");
    card.className = "seg-card";
    card.dataset.cluster = c.id;
    card.style.setProperty("--seg-color", color);
    const leftPct = (c.p25_price / maxP75) * 100;
    const widthPct = ((c.p75_price - c.p25_price) / maxP75) * 100;
    card.innerHTML = `
      <div class="seg-name">${c.label}</div>
      <div class="seg-count">${c.n.toLocaleString()} listings · ${fmtPct(c.share)} of market</div>
      <div class="seg-price">${fmtPrice(c.median_price)}<small>typical / night</small></div>
      <div class="seg-range">
        <div class="seg-range-fill" style="left:${leftPct}%; width:${widthPct}%"></div>
      </div>
      <div class="seg-range-cap">
        <span>${fmtPrice(c.p25_price)}</span>
        <span>${fmtPrice(c.p75_price)}</span>
      </div>
      <div class="seg-meta">
        <span><strong>${c.median_accommodates}</strong>guests</span>
        <span><strong>${c.median_bedrooms}</strong>bedroom${c.median_bedrooms === 1 ? "" : "s"}</span>
        <span><strong>${c.median_beds}</strong>bed${c.median_beds === 1 ? "" : "s"}</span>
      </div>
    `;
    card.addEventListener("click", () => {
      toggleClusterFocus(c.id);
      document.getElementById("map-section").scrollIntoView({ behavior: "smooth", block: "start" });
    });
    wrap.appendChild(card);
  }
}

// ---------------------------------------------------------------------------
// map
// ---------------------------------------------------------------------------
function initMap() {
  state.map = L.map("map", { zoomSnap: 0.5, scrollWheelZoom: false })
    .setView([52.367, 4.895], 12);
  // remove Leaflet's default 🇺🇦 prefix
  state.map.attributionControl.setPrefix(
    '<a href="https://leafletjs.com" target="_blank" rel="noopener">Leaflet</a>',
  );
  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
    subdomains: "abcd",
    maxZoom: 18,
  }).addTo(state.map);

  state.map.on("click", (e) => setPickedLocation(e.latlng));

  state.geoLayer = L.geoJSON(state.geo, {
    style: neighStyle,
    onEachFeature: (feature, layer) => {
      const name = feature.properties.neighbourhood;
      const nb = state.data.neighbourhoods[name];
      const labelById = Object.fromEntries(state.data.clusters.map((c) => [c.id, c.label]));
      const tooltip = nb
        ? `<strong>${name}</strong><br/>${nb.total} listings · ${fmtPrice(nb.median_price)}<br/>dominant: ${labelById[nb.dominant_cluster]}`
        : `<strong>${name}</strong><br/>no listings in dataset`;
      layer.bindTooltip(tooltip, { sticky: true, opacity: 0.96 });
      layer.on({
        mouseover: (e) => e.target.setStyle({ weight: 2.5, color: "#0e1116", fillOpacity: 0.9 }),
        mouseout: (e) => state.geoLayer.resetStyle(e.target),
        click: () => selectNeigh(name),
      });
    },
  }).addTo(state.map);
  state.map.fitBounds(state.geoLayer.getBounds(), { padding: [10, 10] });
}

function neighStyle(feature) {
  const name = feature.properties.neighbourhood;
  const nb = state.data.neighbourhoods[name];
  if (!nb) {
    return { color: "#bbb", weight: 1, fillColor: "#e2e2e2", fillOpacity: 0.35 };
  }
  const focused = state.selectedCluster;
  if (focused === null) {
    return {
      color: "#5b6470", weight: 1,
      fillColor: colorForCluster(nb.dominant_cluster),
      fillOpacity: 0.72,
    };
  }
  const total = nb.total || 0;
  const count = (nb.counts && nb.counts[focused]) ? nb.counts[focused] : 0;
  const share = total ? count / total : 0;
  const peak = state.maxShareByCluster[focused] || 1;
  const rel = peak > 0 ? share / peak : 0;
  return {
    color: share > 0 ? "#5b6470" : "#cfd5df",
    weight: share > 0 ? 1 : 0.5,
    fillColor: colorForCluster(focused),
    fillOpacity: share === 0 ? 0.05 : 0.18 + rel * 0.72,
  };
}

function selectNeigh(name) {
  const nb = state.data.neighbourhoods[name];
  const detail = $("neigh-detail");
  if (!nb) {
    detail.className = "neigh-detail empty";
    detail.textContent = `${name}: no listings in the dataset.`;
    return;
  }
  detail.className = "neigh-detail";
  const labelById = Object.fromEntries(state.data.clusters.map((c) => [c.id, c.label]));
  const total = nb.total;
  const segs = Object.entries(nb.counts)
    .map(([cid, count]) => ({ cid: Number(cid), count, share: count / total }))
    .filter((s) => s.count > 0)
    .sort((a, b) => b.count - a.count);
  const bar = segs.map((s) =>
    `<div title="${labelById[s.cid]}: ${s.count} (${fmtPct(s.share)})"
          style="width:${s.share * 100}%; background:${colorForCluster(s.cid)}"></div>`,
  ).join("");
  const legend = segs.map((s) =>
    `<span><span style="display:inline-block; width:9px; height:9px; background:${colorForCluster(s.cid)}; border-radius:2px; margin-right:4px;"></span>${labelById[s.cid]} ${fmtPct(s.share)}</span>`,
  ).join("");
  detail.innerHTML = `
    <div><strong>${name}</strong>
      <span style="color:var(--ink-soft); font-size:0.85rem;">
        · ${total} listings · median ${fmtPrice(nb.median_price)}
      </span>
    </div>
    <div style="font-size:0.85rem; color:var(--ink-soft); margin-top:0.2rem;">
      Mostly <strong style="color:${colorForCluster(nb.dominant_cluster)}">${labelById[nb.dominant_cluster]}</strong>
    </div>
    <div class="neigh-bar">${bar}</div>
    <div class="neigh-bar-legend">${legend}</div>
  `;
}

// ---------------------------------------------------------------------------
// charts (Chart.js)
// ---------------------------------------------------------------------------
function initCharts() {
  Chart.defaults.font.family = "Inter, system-ui, sans-serif";
  Chart.defaults.color = "#4d5562";
  Chart.defaults.borderColor = "#e7e9ee";

  const clusters = [...state.data.clusters].sort((a, b) => a.median_price - b.median_price);
  const labels = clusters.map((c) => c.label);
  const colors = clusters.map((c) => colorForCluster(c.id));
  const colorsSoft = clusters.map((c) => colorForCluster(c.id) + "40");

  // ----- price chart: floating bars for the P25–P75 range, dot for the median -----
  const iqrRanges = clusters.map((c) => [c.p25_price, c.p75_price]);
  const medianPoints = clusters.map((c) => c.median_price);
  new Chart($("chart-price"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        { type: "bar", label: "Where most fall (P25–P75)", data: iqrRanges,
          backgroundColor: colorsSoft, borderColor: colors, borderWidth: 2,
          borderRadius: 8, barPercentage: 0.55, categoryPercentage: 0.85, order: 2 },
        { type: "line", label: "Median", data: medianPoints,
          showLine: false, pointRadius: 7, pointHoverRadius: 9,
          pointBackgroundColor: "#0e1116", pointBorderColor: "#fff", pointBorderWidth: 2,
          order: 1 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 900, easing: "easeOutQuart" },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => clusters[items[0].dataIndex].label,
            label: (ctx) => {
              const c = clusters[ctx.dataIndex];
              return [
                `Median  ${fmtPrice(c.median_price)} / night`,
                `Range  ${fmtPrice(c.p25_price)}–${fmtPrice(c.p75_price)}`,
              ];
            },
          },
        },
      },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 11 }, autoSkip: false, maxRotation: 0,
          callback: function (v) { const s = this.getLabelForValue(v); return s.length > 14 ? s.slice(0, 12) + "…" : s; } } },
        y: { beginAtZero: true, grid: { color: "#eef0f4" }, ticks: { callback: (v) => "€" + v } },
      },
    },
  });

  // ----- share chart: doughnut -----
  new Chart($("chart-share"), {
    type: "doughnut",
    data: {
      labels,
      datasets: [{
        data: clusters.map((c) => c.n),
        backgroundColor: colors,
        borderColor: "#fff",
        borderWidth: 3,
        hoverOffset: 8,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: "60%",
      animation: { animateRotate: true, animateScale: true, duration: 1100 },
      plugins: {
        legend: { position: "bottom", labels: { boxWidth: 10, padding: 12, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.label}: ${ctx.parsed.toLocaleString()} listings (${fmtPct(ctx.parsed / state.data.n_listings)})`,
          },
        },
      },
    },
  });

  // ----- neighbourhood composition: horizontal stacked bar -----
  const nbEntries = Object.entries(state.data.neighbourhoods)
    .sort((a, b) => b[1].total - a[1].total)
    .slice(0, 8);
  const nbLabels = nbEntries.map(([n]) => n);
  const datasets = clusters.map((c) => ({
    label: c.label,
    data: nbEntries.map(([, nb]) => (nb.counts && nb.counts[c.id]) ? nb.counts[c.id] : 0),
    backgroundColor: colorForCluster(c.id),
    borderWidth: 0,
  }));
  new Chart($("chart-neigh"), {
    type: "bar",
    data: { labels: nbLabels, datasets },
    options: {
      indexAxis: "y", responsive: true, maintainAspectRatio: false,
      animation: { duration: 900 },
      plugins: {
        legend: { position: "bottom", labels: { boxWidth: 10, padding: 10, font: { size: 11 } } },
        tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.x.toLocaleString()}` } },
      },
      scales: {
        x: { stacked: true, grid: { color: "#eef0f4" }, ticks: { precision: 0 } },
        y: { stacked: true, grid: { display: false } },
      },
    },
  });
}

// ---------------------------------------------------------------------------
// predict (nearest-centroid)
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
  if (!cents) return state.data.clusters.find((c) => c.stratum === "specialty");
  const order = cents.feature_order;
  const x = order.map((f) => {
    switch (f) {
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
  const xs = x.map((v, i) => (v - cents.scaler_mean[i]) / cents.scaler_scale[i]);
  let bestIdx = 0, bestDist = Infinity;
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

function setPickedLocation(latlng) {
  state.pickedLatLng = latlng;
  if (state.pickMarker) state.pickMarker.setLatLng(latlng);
  else state.pickMarker = L.marker(latlng).addTo(state.map);
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
      lat: loc.lat, lon: loc.lng,
    };
    const c = predictCluster(input);
    const result = $("predict-result");
    result.hidden = false;
    result.innerHTML = `
      <div class="pr-label">Closest segment</div>
      <div class="pr-name" style="color:${colorForCluster(c.id)}">${c.label}</div>
      <div class="pr-price">
        Typically <strong>${fmtPrice(c.median_price)}</strong> / night ·
        most fall between <strong>${fmtPrice(c.p25_price)}</strong> and <strong>${fmtPrice(c.p75_price)}</strong>
      </div>
      <div class="pr-tip">↑ Scroll up — the map is now shaded by where this segment lives.</div>
    `;
    applyClusterFocus(c.id);
  });
}

// ---------------------------------------------------------------------------
boot().catch((err) => {
  console.error(err);
  document.body.insertAdjacentHTML("afterbegin",
    `<div style="padding:1rem; background:#fee2e2; color:#7f1d1d; font-family:Inter,sans-serif;">
       Failed to load data. Serve via http (not file://): <code>python -m http.server -d web 8000</code>
     </div>`);
});
