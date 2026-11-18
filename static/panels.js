const PHASE_COLORS = { R: "#f18478", G1: "#abdf90", G2: "#7cd4b9", B: "#86a9f5", "Edge response": "#bef39c", "MTF": "#86a9f5" };
export const number = (value, places = 4) => value == null ? "—" : Number(value).toLocaleString("en-US", { maximumFractionDigits: places });
export const escape = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
const esc = escape;

export function select(id, label, entries, val) {
  const choices = entries.map((item) => Array.isArray(item) ? item : [item, item]);
  return `<label class="field"><span>${esc(label)}</span><select id="${id}">${choices.map(([value, text, disabled]) => `<option value="${esc(value)}"${String(value) === String(val) ? " selected" : ""}${disabled ? " disabled" : ""}>${esc(text)}</option>`).join("")}</select></label>`;
}
export function input(id, label, value = "", type = "number", extra = "") {
  return `<label class="field"><span>${esc(label)}</span><input id="${id}" type="${type}" value="${esc(value)}" ${extra}></label>`;
}
export function check(id, label, value = false) {
  return `<label class="check"><input type="checkbox" id="${id}"${value ? " checked" : ""}>${esc(label)}</label>`;
}
export function fields(html, columns) {
  return `<div class="${columns}">${html.join("")}</div>`;
}
export function group(title, body, collapse = false) {
  if (collapse) return `<details class="detail"><summary>${esc(title)}</summary>${body}</details>`;
  return `<div class="detail"><h3>${esc(title)}</h3>${body}</div>`;
}
export function actions(items) {
  return `<div class="control-actions">${items.map(([id, text, accent]) => `<button id="${id}"${accent ? ' class="accent"' : ""}>${esc(text)}</button>`).join("")}</div>`;
}
export function facts(items) {
  return `<dl class="grid-data">${items.map(([key, value]) => `<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`).join("")}</dl>`;
}
export function table(headers, rows) {
  return `<div class="table-wrap"><table><thead><tr>${headers.map((title) => `<th>${esc(title)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((value) => `<td>${esc(value)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

const METHODS = [["directional", "Edge-directed"], ["bilinear", "Bilinear"], ["nearest", "Nearest site"]];
const MEASURES = [
  ["region", "Region statistics"], ["histogram", "Exposure histogram"], ["exposure", "RAW exposure headroom"],
  ["profile", "Sensor line profile"], ["banding", "Readout periodicity"], ["spatial-noise", "Spatial noise"],
  ["paired-noise", "Temporal noise pair"], ["uniformity", "Flat-field uniformity"], ["edge", "Slanted-edge response"],
  ["chromatic", "RAW chromatic edge displacement"], ["reference", "Reference integrity"], ["mask-audit", "Defect mask audit"], ["color-audit", "Color target audit"],
];

export function developPanel(state, refs) {
  const value = state.settings;
  const wb = fields(["Red", "Green", "Blue"].map((name, n) => input(`wb-${n}`, name, value.white_balance[n], "number", "min='.125' max='8' step='.01'")), "triplet");
  const rgb = select("matrix-name", "Camera color matrix", [["identity", "Identity"], ["studio", "Studio"], ["warm-practical", "Warm light"], ["custom", "Custom"]], value.matrix_name);
  const matrix = value.matrix_name === "custom" ? fields(value.custom_matrix.map((cell, index) => input(`matrix-${index}`, `${Math.floor(index / 3) + 1}·${(index % 3) + 1}`, cell, "number", "step='.01'")), "triplet") : "";
  const refsHtml = ["dark", "flat", "defects"].map((key) => {
    const names = { dark: "Dark frame", flat: "Flat field", defects: "Defect mask" };
    const options = [["", "None"], ...(refs[key] ?? []).map((item) => [item.id, `${item.name}${item.compatible ? "" : " · incompatible"}`, !item.compatible])];
    return select(`ref-${key}`, names[key], options, value.calibration[`${key}_id`] ?? "");
  }).join("");
  return `<h2 class="control-heading">Develop controls</h2>
    ${group("Reconstruction", select("method", "Bayer interpolation", METHODS, value.method) + check("active-only", "Active sensor area only", value.active_only))}
    ${group("Exposure & white balance",input("exposure", "Exposure compensation (EV)", value.exposure_ev, "number", "min='-6' max='6' step='.1'") + wb + actions([["white-patch", "Use selected neutral patch"]]))}
    ${group("Tone & color",rgb + matrix + select("tone", "Tone curve", [["linear","Linear"],["soft","Soft shoulder"],["high-contrast","High contrast"]], value.tone) + fields([
      input("contrast", "Contrast", value.contrast, "number", "min='.25' max='3' step='.05'"), input("saturation", "Saturation", value.saturation, "number", "min='0' max='2.5' step='.05'"),
      ], "pair") + select("profile", "Export gamut", [["srgb","sRGB"],["display-p3","Display P3"]], value.profile), true)}
    ${group("Sensor calibration", refsHtml + check("scale-dark", "Scale dark exposure", value.calibration.scale_dark_exposure) + check("repair", "Detect isolated outliers", value.calibration.repair_detected) +
      check("row-offset-ob", "Row offset from masked columns", value.calibration.row_offset_ob) +
      fields([input("flat-floor","Flat floor",value.calibration.flat_floor,"number","step='.01'"),input("defect-threshold","Outlier threshold",value.calibration.defect_threshold,"number","step='1'")],"pair") +
      select("cal-map","Calibration map",[["corrected","Corrected intensity"],["gain","Flat gain"],["defects","Repaired sites"],["validity","Site validity"]],state.view.calibration_layer), true)}
    ${group("Develop recipe",actions([["save-recipe", "Save recipe"], ["clear-settings", "Reset"]]))}`;
}

function regionPanel(region, candidates) {
  const bounds = ["x", "y", "width", "height"];
  const boxes = fields(bounds.map((key) => input(`region-${key}`, key, region?.[key] ?? "", "number", "min='0' step='1'")), "quad");
  const chooser = candidates?.length ? select("sample-roi", "Sample regions", [["","Choose a reference"],...candidates.map((item, index) => [index, item.label])],"") : "";
  return boxes + actions([["apply-region","Set region"],["clear-region","Full active"]]) + chooser;
}

export function inspectPanel(state, project, prefs) {
  const kind = state.measurement;
  let options = "";
  if (kind === "histogram") {
    options = select("hist-space","Scale",[["linear","Linear"],["stops","Stops from saturation"]],prefs.space) + select("hist-bins","Bins",[16,32,64,128,256],prefs.bins);
  } else if (kind === "profile" || kind === "banding") {
    options = select("axis","Trace lines",[["rows","Sensor rows"],["columns","Sensor columns"]],prefs.axis);
    if (kind === "profile") options += check("profile-normalized","Normalize by phase range",prefs.normalized_profile) + check("corrected-profile","Overlay calibrated traces",prefs.corrected);
  } else if (kind === "paired-noise") {
    options = select("pair-id","Repeat capture",[["","Choose a capture"],...project.frames.filter((item) => item.id!==state.frame_id && item.role==="scene").map((item)=>[item.id,item.name])],prefs.pair_id??"");
  } else if (kind === "uniformity") {
    options = input("uniformity-tiles","Grid per axis",prefs.tiles,"number","min='2' max='12'");
  } else if (kind === "region") {
    options = check("corrected-region","Include calibration statistics",prefs.corrected);
  } else if (kind === "edge" || kind === "chromatic") {
    options = select("edge-orientation","Edge orientation",[["automatic","Automatic"],["vertical","Near vertical"],["horizontal","Near horizontal"]],prefs.edge.orientation);
    if (kind === "chromatic") options += '<p>Select one slightly tilted grayscale edge with room on both sides and no clipping.</p>';
  } else if (kind === "mask-audit") {
    options = '<p>Select a defect mask from Calibration references. Measure the full active area or a region.</p>';
  } else if (kind === "color-audit") {
    options = project.chart_swatches?.length
      ? `<p>Measures the 18 printed patches in the synthetic color target.</p>`
      : `<p>Draw one color patch and enter its reference sRGB color.</p>${input("target-color","Reference color",prefs.target_color,"color")}`;
  }
  return `<h2 class="control-heading">Sensor measurements</h2>${group("Selected region",regionPanel(state.region, project.suggested_regions))}
    ${group("Measurement",select("measurement","Analysis",MEASURES,kind)+options+actions([["measure","Measure",true]]))}
    ${group("Pixel probe", check("develop-probe","Include reconstructed RGB", prefs.include_developed) + '<p>Click the image to inspect the original code and adjacent Bayer sites.</p>')}`;
}

export function comparisonPanel(state, project) {
  const settings = state.comparison_settings;
  const o = state.comparison_options;
  const valid = project.frames.filter((item) => item.role === "scene" && item.id !== state.frame_id);
  return `<h2 class="control-heading">Compare developments</h2>${group("Reference development",
    select("compare-frame","Reference capture",[[state.frame_id,"Same capture"],...valid.map((item)=>[item.id,item.name])],state.comparison_frame_id ?? state.frame_id) +
    select("compare-method","Reference interpolation",METHODS,settings.method) +
    input("compare-ev","Reference exposure (EV)",settings.exposure_ev,"number","min='-6' max='6' step='.1'") +
    actions([["copy-settings","Use primary settings"]]))}
    ${group("Comparison view",select("compare-mode","View",[["wipe","Wipe"],["checker","Checkerboard"],["absolute","Absolute difference"],["signed-luma","Signed luma"],["chroma","Chroma difference"]],o.mode)+
    select("compare-space","Difference space",[["linear","Linear working RGB"],["display","Encoded display RGB"]],o.space)+
    (o.mode==="wipe"?input("wipe","Wipe position",o.split,"range","min='0' max='1' step='.01'"):"")+
    fields([input("compare-x","Reference x shift",o.dx),input("compare-y","Reference y shift",o.dy)],"pair")+
    fields([input("compare-threshold","Error threshold",o.threshold,"number","step='.01'"),input("compare-gain","Map gain",o.gain,"number","step='.25'")],"pair")+
    actions([["run-compare","Compare",true],["compare-export","Export review"]]))}`;
}

function chart(label, series, options = {}) {
  const legend = series.map(({name, color})=>`<span><i style="background:${color ?? PHASE_COLORS[name] ?? "#bef39c"}"></i>${esc(name)}</span>`).join("");
  return {markup:`<canvas class="chart" aria-label="${esc(label)}" role="img"></canvas><div class="chart-legend">${legend}</div>`, series, options};
}

export function drawChart(canvas, {series, options}) {
  const width = canvas.clientWidth || 400;
  const height = canvas.clientHeight || 175;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width*dpr);
  canvas.height = Math.round(height*dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr,dpr);
  const valid = series.flatMap(({points})=>points.filter(([x,y])=>Number.isFinite(x)&&Number.isFinite(y)));
  if (!valid.length) return;
  const xx = valid.map(([x])=>x), yy = valid.map(([,y])=>y);
  const xmin = Math.min(...xx), xmax = Math.max(...xx);
  const ymin = options.zero ? 0 : Math.min(...yy), ymax = Math.max(...yy);
  const pad = {top:13,right:10,bottom:24,left:43};
  const px = (x) => pad.left + (x - xmin) * (width-pad.left-pad.right)/(xmax-xmin||1);
  const py = (y) => height-pad.bottom - (y-ymin)*(height-pad.top-pad.bottom)/(ymax-ymin||1);
  ctx.font = "10px monospace";
  ctx.fillStyle = "#9ba99a";
  ctx.strokeStyle = "#354437";
  ctx.lineWidth = 1;
  for (let i=0;i<=4;i++) {
    const val = ymin+(ymax-ymin)*i/4, y = py(val);
    ctx.beginPath(); ctx.moveTo(pad.left,y); ctx.lineTo(width-pad.right,y); ctx.stroke();
    ctx.fillText(number(val,2),4,y+3);
    if (i%2===0) {
      const x = xmin+(xmax-xmin)*i/4;
      ctx.fillText(number(x,2),Math.min(px(x),width-30),height-7);
    }
  }
  for (const {name,points,color,dash} of series) {
    ctx.lineWidth = 1.7;
    ctx.strokeStyle = color ?? PHASE_COLORS[name] ?? "#bef39c";
    ctx.setLineDash(dash ?? []);
    ctx.beginPath();
    let started = false;
    for (const [x,y] of points) {
      if (!Number.isFinite(x)||!Number.isFinite(y)) { started=false; continue; }
      if (started) ctx.lineTo(px(x),py(y));
      else { ctx.moveTo(px(x),py(y));started=true; }
    }
    ctx.stroke();
  }
  ctx.setLineDash([]);
}

function rowOffsetEvidence(row) {
  if (!row) return "";
  const strips=(row.strips??[]).map((s)=>`${s.side} ${s.columns}`).join(" · ")||"none";
  let html=`<h4>Row offset from masked columns</h4>${facts([
    ["Masked columns",strips],["Minimum samples per row",row.min_samples_per_row],
    ["Sensor rows left unchanged",row.uncorrected_row_count],
  ])}`;
  html+=table(["Phase","OB samples/row","Rows corrected","Outliers rejected","RMS offset (code)"],
    (row.phases??[]).map((p)=>[p.phase,p.ob_samples_per_row,p.rows_corrected,p.outlier_samples_rejected,number(p.rms_offset_code)]));
  if (row.uncorrected_row_count) {
    const listed=(row.rows_not_corrected??[]).join(", ");
    html+=`<p>${row.uncorrected_row_count} sensor row(s) kept as captured: ${esc(listed)}${row.uncorrected_row_count>(row.rows_not_corrected??[]).length?" …":""}</p>`;
  }
  for (const warning of row.warnings??[]) html+=`<p>${esc(warning)}</p>`;
  return html;
}

export function sensorEvidence(kind, report) {
  const title = MEASURES.find(([key])=>key===kind)?.[1] ?? kind;
  let body = "", graph = null;
  const box = report.region ?? report.left_box;
  const boxInfo = box ? `<p>Raster region ${box.x}, ${box.y} · ${box.width} × ${box.height}</p>` : "";
  if (kind==="region") {
    body = table(["Phase","Sites","Median code","Norm. median","Below black","Clipped"],report.phases.map((p)=>[p.phase,p.raw.count,number(p.raw.median),number(p.normalized.median),p.under_black,p.clipped]));
    if (report.calibrated) body += table(["Calibrated phase","Usable","Repaired","Weak flat","Median"],report.calibrated.phases.map((p)=>[p.phase,p.usable,p.filled,p.weak_flat,number(p.values.median)]));
  } else if (kind==="histogram") {
    graph = chart("Counts by exposure bin",report.phases.map((p)=>({name:p.phase,points:p.counts.map((y,i)=>[(p.edges[i]+p.edges[i+1])/2,y])})),{zero:true});
    body = table(["Phase","Sites","Below range","Above range","Clipped"],report.phases.map((p)=>[p.phase,p.samples,p.below,p.above,p.clipped]));
  } else if (kind==="exposure") {
    body = table(["Phase","Sites","Near saturation ≥98%","Clipped","Headroom (EV)"],report.phases.map((p)=>[p.phase,p.samples,p.levels.find((b)=>b.threshold===.98)?.count??0,p.levels.find((b)=>b.threshold===1)?.count??0,number(p.headroom_stops_p99_5,3)]));
  } else if (kind==="profile") {
    const traces=report.traces.map((p)=>({name:p.phase,points:p.positions.map((x,i)=>[x,p.medians[i]])}));
    let calibrated="";
    if (report.calibrated) {
      for (const p of report.calibrated.traces) {
        traces.push({name:`${p.phase} calibrated`,color:PHASE_COLORS[p.phase],dash:[5,3],points:p.positions.map((x,i)=>[x,p.medians[i]])});
      }
      calibrated=rowOffsetEvidence(report.calibrated.row_offset);
    }
    graph = chart("Line median profile",traces);
    body = `<p>Sensor ${esc(report.axis)} · ${esc(report.unit)}</p>${calibrated}`;
  } else if (kind==="banding") {
    body = table(["Phase","RMS code","Peak period (sites)","Amplitude"],report.phases.map((p)=>[p.phase,number(p.rms_code),number(p.peak_periods[0]?.period_sites),number(p.peak_periods[0]?.amplitude)])) + `<p>${esc(report.warning)}</p>`;
  } else if (kind==="spatial-noise"||kind==="paired-noise") {
    const labels=kind==="paired-noise"?["Temporal σ","SNR dB"]:["Row σ","Column σ"];
    body = table(["Phase","Usable samples","Signal",...labels],report.phases.map((p)=>[p.phase,p.usable,number(p.median_signal,6),number(p.robust_temporal_sigma??p.row_sigma,7),number(p.snr_db??p.column_sigma,6)]));
    body += `<p>${esc(report.meaning??report.warnings?.join(" "))}</p>`;
  } else if (kind==="uniformity") {
    body = table(["Phase","Median code","Valid sites","Darkest tile","Brightest tile"],report.per_phase.map((p)=>[p.phase,number(p.median_code,2),p.valid,number(p.low_tile_ratio),number(p.high_tile_ratio)]))+`<p>${esc(report.meaning)}</p>`;
  } else if (kind==="edge") {
    graph = chart("Edge MTF response",[{name:"MTF",points:report.frequency.map((x,i)=>[x,report.mtf[i]])}],{zero:true});
    body = facts([["Direction",report.direction],["MTF50",`${number(report.mtf50)} cycles/pixel`],["MTF10",number(report.mtf10)],["Fit residual",`${number(report.fit_residual_px)} px`],["Edge angle",`${number(report.edge_angle_deg_from_axis)}°`]])+`<p>${esc(report.warning)}</p>`;
  } else if (kind==="chromatic") {
    graph = chart("Edge responses against green · sensor pixels",report.phases.map((p)=>({name:p.phase,points:p.response.map((y,i)=>[report.positions_px[i],y])})));
    const geometry = report.geometry;
    body = facts([["Orientation",geometry.direction],["Edge tilt",`${number(geometry.angle_from_axis,2)}°`],["Green separation",`${number(report.green_separation_px,3)} px`],["Sensor reference",`${number(geometry.reference_point.sensor_x,2)}, ${number(geometry.reference_point.sensor_y,2)}`]]) +
      table(["Phase","Offset (px)","10–90% width","Strip spread","Sites"],report.phases.map((p)=>[p.phase,number(p.offset_px,3),number(p.transition_width_px,3),number(p.strip_offset_range_px,3),p.samples])) +
      `<p>Positive offsets point toward the bright side. ${esc(report.warning)}</p>`;
  } else if (kind==="reference") {
    body = table(["Phase","Sites","Median above black","Robust σ","Near saturation"],report.phases.map((p)=>[p.phase,p.sites,number(p.median_code_above_black,2),number(p.robust_sigma_code),p.near_saturation]));
  } else if (kind==="mask-audit") {
    const groups=report.groups;
    body = facts([["Marked in region",report.marked],["Outside active area",report.outside_active],["Isolated sites",groups.singletons],["Connected groups",groups.multiple_sites],["Rows entirely marked",report.rows.entirely_marked],["Columns entirely marked",report.columns.entirely_marked]]) +
      table(["Phase","Marked","Per megapixel"],report.phases.map((phase)=>[phase.phase,phase.marked,number(phase.per_megapixel,2)]));
    const grid=report.grid.cells;
    const maximum=Math.max(1,...grid.flat().map((tile)=>tile.per_megapixel));
    const peak=report.grid.densest;
    body += `<h4>Marked sites by region</h4>${peak?`<p>Highest density ${number(peak.per_megapixel,1)} sites per megapixel · raster ${peak.box.x}, ${peak.box.y}</p>`:""}<div class="mask-grid" style="grid-template-columns:repeat(${report.grid.columns},1fr)">${grid.flat().map((tile,i)=>`<button type="button" data-mask-tile="${i}" title="Raster ${tile.box.x}, ${tile.box.y} · ${number(tile.per_megapixel,1)} sites per megapixel" style="--density:${tile.per_megapixel/maximum}">${tile.marked}</button>`).join("")}</div>`;
    if (groups.shown.length) {
      body += `<h4>Largest connected marks</h4><div class="mask-groups">${groups.shown.map((entry,i)=>`<button type="button" data-mask-group="${i}">${entry.sites} ${entry.sites===1?"site":"sites"} at ${entry.sensor_box.x}, ${entry.sensor_box.y}${entry.touches_region_boundary?" · touches edge":""}</button>`).join("")}</div>`;
    }
  } else if (kind==="color-audit") {
    body = facts([["Usable patches",`${report.usable}/${report.total}`],["Median ΔE₀₀",number(report.median_delta_e)],["Maximum ΔE₀₀",number(report.maximum_delta_e)],["Worst patch",report.worst_patch]])+
      table(["Patch","Samples","ΔE₀₀","90th pct","Notes"],report.patches.map((p)=>[p.name,p.samples,number(p.delta_e2000),number(p.p90_delta_e),p.reason??""]));
  }
  const buttons = `<div class="results-actions"><button id="measure-json">Download JSON</button>${["histogram","profile","banding","edge","chromatic","mask-audit","color-audit"].includes(kind)?'<button id="measure-csv">Download CSV</button>':""}</div>`;
  return {graph, html:`<h3>${esc(title)}</h3>${boxInfo}${graph?.markup ?? ""}${body}${buttons}`};
}

export function probeEvidence(data) {
  const hit = data.site;
  const grid = `<div class="neighborhood" style="grid-template-columns:repeat(${data.rows[0].length},minmax(33px,1fr))">${data.rows.flat().map((item)=>`<div class="site${item.selected?" selected":""}" style="color:${PHASE_COLORS[item.phase]}">${esc(item.phase)}<small>${item.value}</small></div>`).join("")}</div>`;
  return `<h3>Sensor site · ${hit.phase}</h3>${facts([["Raster coordinate",`${hit.x}, ${hit.y}`],["Sensor coordinate",`${hit.sensor_x}, ${hit.sensor_y}`],["Original code",hit.raw_code],["Black / white",`${hit.black} / ${hit.white}`],["Normalized",number(hit.normalized,6)],["Active area",hit.active?"Inside":"Outside"],...(data.corrected?[["Corrected linear",number(data.corrected.linear,6)],["Repaired",data.corrected.repaired?"Yes":"No"]]:[]),...(data.developed?[["Developed RGB",data.developed.display_rgb.join(" / ")]]:[])])}${grid}`;
}

export function comparisonEvidence(data) {
  const s=data.shared_sensor;
  const rows=data.channels.map((x)=>[x.channel,number(x.mean_error),number(x.mean_absolute_error),number(x.rmse)]);
  return `<h3>Registered comparison</h3><p>Shared sensor region ${s.x}, ${s.y} · ${s.width} × ${s.height} · ${esc(data.space)}</p>${table(["Channel","Mean","MAE","RMSE"],rows)}
    <p>${number(data.over_threshold,0)} pixels exceed ${data.threshold}. ${esc(data.notices.join(" "))}</p>`;
}
