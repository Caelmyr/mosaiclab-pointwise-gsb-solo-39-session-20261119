import {
  actions, check, comparisonEvidence, comparisonPanel, developPanel, drawChart, escape, facts, fields,
  input, inspectPanel, matrixFitEvidence, number, probeEvidence, select, sensorEvidence, table,
} from "./panels.js";

const $ = (id) => document.getElementById(id);
const esc = escape;
const savedProject = "mosaiclab.project";
const state = {
  project: null, projects: [], frame: null, inspection: null, options: null, frameLoading: false,
  references: {}, tab: "develop", tool: "probe", zoom: false, preview: null,
  seq: 0, drag: null, result: null, timer: null, toastTimer: null,
};
const defaultView = () => ({layer:"developed", width:1000, sensor_gamma:2.2, green_lattice:true, calibration_layer:"corrected"});
const defaultCompare = () => ({mode:"wipe",space:"linear",split:.5,dx:0,dy:0,threshold:.05,gain:5});
const freshSettings = () => structuredClone(state.options.defaults.settings);
const status = (message, error=false) => {
  $("toast").hidden = false;
  $("toast").textContent = message;
  $("toast").classList.toggle("error",error);
  clearTimeout(state.toastTimer);
  state.toastTimer=setTimeout(()=>{$("toast").hidden=true;}, error?6500:3600);
};
const failed = (error) => status(error.message,true);
const formatRole = (role) => ({scene:"Scene",dark:"Dark",flat:"Flat",defects:"Defect map"})[role] ?? role;
const value = (id, fallback="") => $(id)?.value ?? fallback;
const numeric = (id, fallback=0) => $(id)?.value === "" || !$(id) ? fallback : Number($(id).value);
const checked = (id, fallback=false) => $(id)?.checked ?? fallback;
const bind = (id, handler, event="click") => $(id)?.addEventListener(event,handler);
const endpoint = (path="") => `/api/projects/${state.project.id}${path}`;

async function api(path, options={}) {
  const init = {...options};
  if (init.json !== undefined) {
    init.headers = {...init.headers,"Content-Type":"application/json"};
    init.body=JSON.stringify(init.json);
    delete init.json;
  }
  const response = await fetch(path,init);
  if (!response.ok) {
    const problem = await response.json().catch(()=>({error:`Request failed (${response.status}).`}));
    throw new Error(problem.error);
  }
  if (response.status===204) return null;
  if (options.blob) return response;
  return response.json();
}

function post(path, data={}) {
  return api(endpoint(path),{method:"POST",json:{inspection:state.inspection,...data}});
}

function remember(stateValue) {
  try { localStorage.setItem(savedProject,stateValue); } catch {}
}

async function projectList() {
  const data = await api("/api/projects");
  state.projects=data.projects;
  $("projects").innerHTML=data.projects.map(({id,name})=>`<option value="${id}">${esc(name)}</option>`).join("");
  if (state.project) $("projects").value=state.project.id;
  return data.projects;
}

async function init() {
  try {
    state.options=await api("/api/options");
    let list=await projectList();
    if (!list.length) {
      const demo=await api("/api/projects",{method:"POST",json:{demo:"color-target"}});
      list=await projectList();
      await loadProject(demo.id,demo);
    } else {
      let previous;
      try { previous=localStorage.getItem(savedProject); } catch {}
      await loadProject(list.find((item)=>item.id===previous)?.id ?? list[0].id);
    }
  } catch(error) {
    showEmpty("Couldn't open the workspace");
    failed(error);
  }
}

async function loadProject(id, ready=null) {
  const project=ready ?? await api(`/api/projects/${id}`);
  state.project=project;
  remember(id);
  $("projects").value=id;
  const selection=project.selection ?? {};
  const first=project.frames.find((item)=>item.id===selection.frame_id)
    ?? project.frames.find((item)=>item.role==="scene") ?? project.frames[0];
  state.tab="develop";
  await setFrame(first?.id ?? null,selection);
}

function inspectionFor(id, saved={}) {
  const preferences = state.options.defaults.measurement_options;
  const stored = saved.measurement_options ?? {};
  return {
    frame_id:id,
    settings:saved.settings ?? freshSettings(),
    view:{...defaultView(),...saved.view},
    region:saved.region ?? null,
    probe:saved.probe ?? null,
    measurement:saved.measurement ?? "region",
    measurement_options:{...preferences,...stored,edge:{...preferences.edge,...stored.edge}},
    comparison_frame_id:saved.comparison_frame_id ?? null,
    comparison_settings:saved.comparison_settings ?? freshSettings(),
    comparison_options:{...defaultCompare(),...saved.comparison_options},
  };
}

async function setFrame(id, saved) {
  const projectId=state.project.id;
  state.seq++;
  state.frame=state.project.frames.find((item)=>item.id===id) ?? null;
  state.frameLoading=Boolean(state.frame);
  state.references={};
  state.inspection=state.frame ? inspectionFor(id,saved) : null;
  if (state.frame?.role!=="scene" && !saved?.view && state.inspection) {
    state.inspection.view.layer="mosaic";
  }
  state.result=null;
  state.preview=null;
  state.zoom=false;
  showEmpty(state.frame ? "Preparing sensor preview" : "Import a capture to start");
  paint();
  if (!state.frame) return;
  const ref=await api(endpoint(`/frames/${id}/references`)).finally(()=>{
    if (state.project.id===projectId && state.frame?.id===id) state.frameLoading=false;
  });
  if (state.project.id!==projectId || state.frame?.id!==id) return;
  state.references=ref;
  paintControls();
  if (state.tab==="compare") await refreshCompare();
  else await refreshPreview();
}

function paint() {
  const p=state.project;
  $("project-name").textContent=p.name;
  $("project-note").textContent=p.notes || (state.frame ? "Develop and inspect native sensor samples." : "Import a Bayer raster or open a synthetic sample.");
  $("project-kind").textContent=p.demo_key?"Synthetic capture":"Sensor capture";
  $("download").disabled=!state.frame;
  $("save").disabled=!state.frame;
  const chips={scene:"RAW",dark:"DK",flat:"FL",defects:"BP"};
  const render=(rows)=>rows.map((f)=>`<button class="frame${f.id===state.frame?.id?" active":""}" data-frame="${f.id}"><span class="frame-chip">${chips[f.role]}</span><span><span class="name">${esc(f.name)}</span><span class="dim">${f.width} × ${f.height} · ${esc(f.local_pattern)}</span></span></button>`).join("");
  const scenes=p.frames.filter((f)=>f.role==="scene");
  const refs=p.frames.filter((f)=>f.role!=="scene");
  $("frames").innerHTML=(scenes.length?`<p class="group">Scene frames</p>${render(scenes)}`:"")+(refs.length?`<p class="group">Calibration references</p>${render(refs)}`:"")||'<p class="empty-help">No captures yet.</p>';
  paintMetadata();
  paintTabs();
  paintControls();
  paintEvidence();
}

function paintMetadata() {
  const f=state.frame;
  $("metadata").hidden=!f;
  if (!f) return;
  const s=f.sensor, b=s.active;
  $("metadata").innerHTML='<h3>Sensor metadata</h3>'+facts([
    ["Role",formatRole(f.role)],["Encoding",f.encoding],["Sensor origin",`${s.origin_x}, ${s.origin_y}`],
    ["Bayer at origin",s.pattern],["Local phase",f.local_pattern],["Active size",`${b.width} × ${b.height}`],
    ["Code maximum",f.maximum],["Exposure",s.exposure_ms==null?"—":`${s.exposure_ms} ms`],
    ["Analog gain",s.analog_gain==null?"—":`${s.analog_gain}×`],
  ])+(f.warnings ?? []).map((warning)=>`<p>${esc(warning)}</p>`).join("")+'<button id="edit-metadata" class="quiet">Edit capture details</button>';
  bind("edit-metadata",()=>metadataDialog());
}

function paintTabs() {
  $("tabs").querySelectorAll("button").forEach((button)=>button.classList.toggle("active",button.dataset.tab===state.tab));
  const history=state.tab==="history";
  $("history").hidden=!history;
  $("screen").hidden=history;
  $("canvas-bar").hidden=history;
  $("image-info").hidden=history;
  $("layer").disabled=state.tab==="compare"||!state.frame;
  $("roi-tool").disabled=state.tab==="compare"||!state.frame;
  $("probe-tool").disabled=state.tab==="compare"||!state.frame;
  if (state.inspection) $("layer").value=state.inspection.view.layer;
  paintTool();
  if (history) paintHistory();
}

function paintControls() {
  $("controls").onchange=null;
  if (!state.inspection || state.frameLoading || state.tab==="history") {
    $("controls").hidden=true;
    return;
  }
  $("controls").hidden=false;
  const current=state.inspection;
  const previousDetails=[...$("controls").querySelectorAll("details")].map((item)=>item.open);
  if (state.tab==="develop") {
    $("controls").innerHTML=developPanel(current,state.references);
    $("controls").querySelectorAll("details").forEach((item,i)=>{item.open=previousDetails[i]??false;});
    bindDevelop();
  } else if (state.tab==="inspect") {
    $("controls").innerHTML=inspectPanel(current,state.project,current.measurement_options);
    bindInspect();
  } else {
    $("controls").innerHTML=comparisonPanel(current,state.project);
    bindCompare();
  }
}

function bindDevelop() {
  $("controls").onchange=developChange;
  bind("white-patch",async()=>{
    try {
      const info=await post("/neutral");
      state.inspection.settings.white_balance=info.gains;
      paintControls();
      await refreshPreview();
      status(info.warnings[0] ?? "White balance set from the selected patch.");
    } catch(error) { failed(error); }
  });
  bind("save-recipe",()=>nameDialog("Save development recipe","Recipe name","My RAW development",async(title)=>{
    await api(endpoint("/recipes"),{method:"POST",json:{title,settings:state.inspection.settings}});
    await updateDetail();
    status("Development recipe saved.");
  }));
  bind("clear-settings",()=>{
    state.inspection.settings=freshSettings();
    paintControls();
    refreshPreview().catch(failed);
  });
}

function developChange(event) {
  if (state.tab!=="develop") return;
  const settings=state.inspection.settings;
  settings.method=value("method",settings.method);
  settings.active_only=checked("active-only",settings.active_only);
  settings.exposure_ev=numeric("exposure",settings.exposure_ev);
  settings.white_balance=settings.white_balance.map((x,i)=>numeric(`wb-${i}`,x));
  settings.matrix_name=value("matrix-name",settings.matrix_name);
  settings.custom_matrix=settings.custom_matrix.map((x,i)=>numeric(`matrix-${i}`,x));
  settings.tone=value("tone",settings.tone);
  settings.contrast=numeric("contrast",settings.contrast);
  settings.saturation=numeric("saturation",settings.saturation);
  settings.profile=value("profile",settings.profile);
  const c=settings.calibration;
  for (const role of ["dark","flat","defects"]) c[`${role}_id`]=value(`ref-${role}`,c[`${role}_id`])||null;
  c.scale_dark_exposure=checked("scale-dark",c.scale_dark_exposure);
  c.repair_detected=checked("repair",c.repair_detected);
  c.flat_floor=numeric("flat-floor",c.flat_floor);
  c.defect_threshold=numeric("defect-threshold",c.defect_threshold);
  state.inspection.view.calibration_layer=value("cal-map",state.inspection.view.calibration_layer);
  if (event.target.id==="matrix-name") paintControls();
  scheduleRefresh();
}

function bindInspect() {
  bind("sample-roi",()=>{
    const selected=$("sample-roi").value;
    if (selected!=="") setRegion(state.project.suggested_regions[Number(selected)].box);
  },"change");
  bind("apply-region",()=>{
    const current=Object.fromEntries(["x","y","width","height"].map((k)=>[k,numeric(`region-${k}`)]));
    const f=state.frame;
    if (current.x<0||current.y<0||current.width<1||current.height<1||current.width+current.x>f.width||current.height+current.y>f.height||Object.values(current).some((v)=>!Number.isInteger(v))) {
      status("Enter a region inside the sensor raster.",true);
      return;
    }
    setRegion(current);
  });
  bind("clear-region",()=>setRegion(null));
  bind("measurement",()=>{
    pullMeasure();
    state.inspection.measurement=$("measurement").value;
    state.result=null;
    paintControls();
    paintEvidence();
  },"change");
  $("controls").onchange=(event)=>{
    if (["measurement","sample-roi"].includes(event.target.id)) return;
    pullMeasure();
    if (state.result?.type==="sensor") {
      state.result=null;
      paintEvidence();
    }
  };
  bind("measure",()=>runMeasurement());
  bind("fit-matrix",()=>fitMatrix());
}

function chartPatches() {
  const patches=state.project.chart_swatches ?? [];
  if (patches.length) return patches;
  const hex=state.inspection.measurement_options.target_color.slice(1);
  const rgb=hex.match(/../g).map((item)=>parseInt(item,16)/255);
  if (!state.inspection.region) throw new Error("Select a swatch region first.");
  return [{name:"Selected patch",region:state.inspection.region,srgb:rgb}];
}

async function fitMatrix() {
  const inspection=state.inspection;
  const context=JSON.stringify(inspection);
  try {
    const patches=chartPatches();
    const data=await post("/matrix-fit",{patches});
    if (state.tab!=="inspect"||state.inspection!==inspection||JSON.stringify(inspection)!==context) return;
    const report=data.report;
    if (report.matrix_replaced) {
      state.inspection.settings.matrix_name="custom";
      state.inspection.settings.custom_matrix=report.custom_matrix;
      paintControls();
      await refreshPreview();
      await updateDetail();
      status(`Color matrix replaced and recipe "${report.saved_recipe.title}" saved.`);
    } else {
      status(report.decision==="rejected-holdout"
        ? "Holdout check failed: existing color matrix kept."
        : "No gray-preserving fit improved the colored patches: existing matrix kept.", true);
    }
    state.result={type:"matrix-fit",data};
    paintEvidence();
  } catch(error) { failed(error); }
}

function pullMeasure() {
  const prefs=state.inspection.measurement_options;
  prefs.space=value("hist-space",prefs.space);
  prefs.bins=numeric("hist-bins",prefs.bins);
  prefs.axis=value("axis",prefs.axis);
  prefs.normalized_profile=checked("profile-normalized",prefs.normalized_profile);
  prefs.pair_id=value("pair-id",prefs.pair_id)||null;
  prefs.tiles=numeric("uniformity-tiles",prefs.tiles);
  prefs.corrected=checked("corrected-region",prefs.corrected);
  prefs.edge.orientation=value("edge-orientation",prefs.edge.orientation);
  prefs.include_developed=checked("develop-probe",prefs.include_developed);
  prefs.target_color=value("target-color",prefs.target_color);
}

async function runMeasurement() {
  pullMeasure();
  const inspection=state.inspection;
  const kind=inspection.measurement;
  const options=inspection.measurement_options;
  const context=JSON.stringify(inspection);
  try {
    let result, patches;
    if (kind==="color-audit") {
      patches=chartPatches();
      result=await post("/color-audit",{patches});
    } else {
      result=await post("/measure",{kind});
    }
    if (state.tab!=="inspect"||state.inspection!==inspection||JSON.stringify(inspection)!==context) return;
    state.result={type:"sensor",kind,data:result,patches};
    paintEvidence();
  } catch(error) { failed(error); }
}

function bindCompare() {
  const pull = () => {
    const a=state.inspection;
    a.comparison_frame_id=value("compare-frame",a.frame_id);
    a.comparison_settings.method=value("compare-method",a.comparison_settings.method);
    a.comparison_settings.exposure_ev=numeric("compare-ev",a.comparison_settings.exposure_ev);
    const o=a.comparison_options;
    o.mode=value("compare-mode",o.mode);
    o.space=value("compare-space",o.space);
    o.split=numeric("wipe",o.split);
    o.dx=numeric("compare-x",o.dx);
    o.dy=numeric("compare-y",o.dy);
    o.threshold=numeric("compare-threshold",o.threshold);
    o.gain=numeric("compare-gain",o.gain);
  };
  bind("compare-mode",()=>{pull();paintControls();},"change");
  bind("run-compare",()=>{pull();refreshCompare().catch(failed);});
  bind("wipe",()=>{pull();refreshCompare().catch(failed);},"change");
  bind("copy-settings",()=>{
    pull();
    state.inspection.comparison_settings=structuredClone(state.inspection.settings);
    if (state.inspection.comparison_frame_id!==state.inspection.frame_id) state.inspection.comparison_settings.calibration=freshSettings().calibration;
    paintControls();
  });
  bind("compare-export",()=>{pull();saveDownload("/compare/download",{}).catch(failed);});
}

function scheduleRefresh() {
  clearTimeout(state.timer);
  state.timer=setTimeout(()=>refreshPreview().catch(failed),125);
}

function showEmpty(message) {
  $("photo").hidden=true;
  $("empty").hidden=false;
  $("empty").querySelector("strong").textContent=message;
  $("image-info").replaceChildren();
}

function showImage(url, info, kind) {
  $("preview").src=url;
  $("empty").hidden=true;
  $("photo").hidden=false;
  $("photo").classList.toggle("raw",kind==="raw");
  $("photo").classList.toggle("zoomed",state.zoom);
  $("photo").classList.toggle("selectable",kind!=="comparison" && !!info.raster_box);
  const b=info.raster_box;
  const title=kind==="comparison"?"Registered difference":kind==="raw"?"Sensor raster":"Developed image";
  $("image-info").innerHTML=`<span>${esc(title)}</span><span>${info.width} × ${info.height} px</span>${b?`<span>Raster origin ${b.x}, ${b.y}</span>`:""}${info.method?`<span>${esc(info.method)}</span>`:""}`;
  state.preview=info;
  paintOverlay();
}

async function refreshPreview() {
  if (!state.frame||state.tab==="history"||state.tab==="compare") return;
  const ticket=++state.seq;
  if (!$("preview").getAttribute("src")||!state.preview) showEmpty("Preparing sensor preview");
  try {
    const data=await post("/preview");
    if (ticket!==state.seq) return;
    showImage(data.url,data.info,data.layer==="developed"?"developed":"raw");
  } catch(error) {
    if (ticket!==state.seq) return;
    showEmpty("Preview could not be rendered");
    throw error;
  }
}

async function refreshCompare() {
  if (!state.frame) return;
  const ticket=++state.seq;
  const data=await post("/compare");
  if (ticket!==state.seq) return;
  showImage(data.url,data.preview,"comparison");
  state.result={type:"comparison",data:data.report};
  paintEvidence();
}

function paintEvidence() {
  const wrap=$("evidence");
  const result=state.result;
  wrap.hidden=!result||state.tab==="history"||(state.tab==="compare"&&result.type!=="comparison");
  if (wrap.hidden) return;
  if (result.type==="site") {
    wrap.innerHTML=probeEvidence(result.data);
  } else if (result.type==="comparison") {
    wrap.innerHTML=comparisonEvidence(result.data);
  } else if (result.type==="matrix-fit") {
    wrap.innerHTML=matrixFitEvidence(result.data);
  } else {
    const resultView=sensorEvidence(result.kind,result.data);
    wrap.innerHTML=resultView.html;
    if (resultView.graph) drawChart(wrap.querySelector("canvas"),resultView.graph);
    if (result.kind==="mask-audit") {
      const tiles=result.data.grid.cells.flat();
      wrap.querySelectorAll("[data-mask-tile]").forEach((button)=>button.addEventListener("click",()=>{
        setRegion(tiles[Number(button.dataset.maskTile)].box);
      }));
      wrap.querySelectorAll("[data-mask-group]").forEach((button)=>button.addEventListener("click",()=>{
        setRegion(result.data.groups.shown[Number(button.dataset.maskGroup)].focus_region);
      }));
    }
    for (const format of ["json","csv"]) {
      bind(`measure-${format}`,()=>{
        const path=result.patches?"/color-audit/download":"/measure/download";
        const details=result.patches?{patches:result.patches}:{kind:result.kind};
        saveDownload(path,{...details,format}).catch(failed);
      });
    }
  }
}

function setRegion(region) {
  state.inspection.region=region;
  state.result=null;
  if (state.tab==="inspect") paintControls();
  paintEvidence();
  paintOverlay();
}

function rasterPoint(event) {
  if (state.tab==="compare"||!state.preview?.raster_box) return null;
  const box=state.preview.raster_box;
  const rect=$("preview").getBoundingClientRect();
  if (!rect.width||!rect.height) return null;
  const nx=Math.max(0,Math.min(.999999,(event.clientX-rect.left)/rect.width));
  const ny=Math.max(0,Math.min(.999999,(event.clientY-rect.top)/rect.height));
  return {x:box.x+Math.floor(nx*box.width),y:box.y+Math.floor(ny*box.height)};
}

function paintOverlay(dragBox=null) {
  const box=state.preview?.raster_box;
  const roi=$("roi-overlay"), dot=$("probe-overlay");
  roi.hidden=true;dot.hidden=true;
  if (!box||!state.inspection) return;
  const region=dragBox??state.inspection.region;
  if (region) {
    const x=Math.max(region.x,box.x), y=Math.max(region.y,box.y);
    const right=Math.min(region.x+region.width,box.x+box.width), bot=Math.min(region.y+region.height,box.y+box.height);
    if (right>x&&bot>y) {
      roi.hidden=false;
      Object.assign(roi.style,{left:`${100*(x-box.x)/box.width}%`,top:`${100*(y-box.y)/box.height}%`,width:`${100*(right-x)/box.width}%`,height:`${100*(bot-y)/box.height}%`});
    }
  }
  const hit=state.inspection.probe;
  if (hit&&hit.x>=box.x&&hit.y>=box.y&&hit.x<box.x+box.width&&hit.y<box.y+box.height) {
    dot.hidden=false;
    Object.assign(dot.style,{left:`${100*(hit.x-box.x+.5)/box.width}%`,top:`${100*(hit.y-box.y+.5)/box.height}%`});
  }
}

const dragBox=(from,to)=>({x:Math.min(from.x,to.x),y:Math.min(from.y,to.y),width:Math.abs(from.x-to.x)+1,height:Math.abs(from.y-to.y)+1});
function paintTool() {
  $("probe-tool").classList.toggle("active",state.tool==="probe");
  $("roi-tool").classList.toggle("active",state.tool==="region");
  $("zoom").textContent=state.zoom?"1:1":"Fit";
}

$("photo").addEventListener("pointerdown",(event)=>{
  if (event.button!==0) return;
  const point=rasterPoint(event);
  if (!point||state.tool!=="region") return;
  state.drag=point;
  $("photo").setPointerCapture(event.pointerId);
  event.preventDefault();
});
$("photo").addEventListener("pointermove",(event)=>{if(state.drag){const hit=rasterPoint(event);if(hit)paintOverlay(dragBox(state.drag,hit));}});
$("photo").addEventListener("pointercancel",()=>{state.drag=null;paintOverlay();});
$("photo").addEventListener("pointerup",async(event)=>{
  if (event.button!==0) return;
  const point=rasterPoint(event);
  if (!point) return;
  if (state.tool==="region") {
    if (state.drag) setRegion(dragBox(state.drag,point));
    state.drag=null;
  } else {
    state.inspection.probe=point;
    paintOverlay();
    pullMeasure();
    try {
      const info=await post("/probe",{radius:2,include_developed:state.inspection.measurement_options.include_developed});
      if (state.inspection.probe!==point) return;
      state.result={type:"site",data:info};
      paintEvidence();
    } catch(error){failed(error);}
  }
});

async function updateDetail() {
  const id=state.project.id;
  const data=await api(endpoint());
  if (state.project.id===id) {
    state.project=data;
    state.frame=data.frames.find((f)=>f.id===state.frame?.id) ?? null;
    paint();
  }
  await projectList();
}

function paintHistory() {
  const history=state.project.sessions??[];
  const recipes=state.project.recipes??[];
  $("history").innerHTML=`<h3>Saved inspections</h3>${history.length?history.map((item)=>`<div class="history-card"><strong>${esc(item.title)}</strong><button data-open="${item.id}">Reopen</button><button data-remove="${item.id}" aria-label="Remove inspection ${esc(item.title)}">×</button></div>`).join(""):'<p class="empty-help">Save the capture, development and region to revisit this inspection.</p>'}
    <h3 style="margin-top:20px">Development recipes</h3>${recipes.length?recipes.map((item)=>`<div class="history-card"><strong>${esc(item.title)}</strong><button data-recipe="${item.id}" ${state.frame?"":"disabled"}>Apply</button><button data-delete-recipe="${item.id}" aria-label="Remove recipe ${esc(item.title)}">×</button></div>`).join(""):'<p class="empty-help">Save development controls to reuse them on another capture.</p>'}`;
}

$("history").addEventListener("click",async(event)=>{
  const data=event.target.dataset;
  try {
    if (data.open) {
      const item=state.project.sessions.find(({id})=>id===data.open);
      state.tab=item.state.measurement==="region"?"develop":"inspect";
      await setFrame(item.state.frame_id,item.state);
      status("Saved inspection reopened.");
    } else if (data.recipe) {
      const item=state.project.recipes.find(({id})=>id===data.recipe);
      state.inspection.settings={...state.inspection.settings,...structuredClone(item.settings)};
      await setTab("develop");
      status("Development recipe applied.");
    } else if (data.remove||data.deleteRecipe) {
      const item=data.remove??data.deleteRecipe;
      await api(endpoint(`/${data.remove?"sessions":"recipes"}/${item}`),{method:"DELETE"});
      await updateDetail();
      status("Saved item removed.");
    }
  } catch(error) { failed(error); }
});

async function setTab(name) {
  state.seq++;
  state.tab=name;
  state.result=null;
  paintTabs();
  paintControls();
  paintEvidence();
  if (name==="compare") {
    showEmpty("Preparing comparison");
    await refreshCompare();
  } else if(name!=="history") await refreshPreview();
}

function modal(title, content) {
  const root=$("dialog");
  if(root.open) root.close();
  root.innerHTML=`<div class="modal-head"><h2>${esc(title)}</h2><button id="dialog-close" class="icon" aria-label="Close">×</button></div><div class="modal-content">${content}</div>`;
  bind("dialog-close",()=>root.close());
  root.showModal();
}

function formAction(handler) {
  $("dialog").querySelector("form")?.addEventListener("submit",async(event)=>{
    event.preventDefault();
    const button=event.currentTarget.querySelector("button[type=submit]");
    if (button) button.disabled=true;
    try {
      await handler(event);
      $("dialog").close();
    } catch(error) {
      failed(error);
      if (button) button.disabled=false;
    }
  });
}

function nameDialog(title,label,placeholder,handler) {
  modal(title,`<form class="modal-content">${input("modal-name",label,placeholder,"text","required maxlength='80'")}<button class="accent" type="submit">Save</button></form>`);
  value("modal-name");
  $("modal-name").select();
  formAction(()=>handler(value("modal-name")));
}

function samplesDialog() {
  modal("Synthetic sample scenes",state.options.demo.map((item)=>`<div class="sample"><div><strong>${esc(item.title)}</strong><small>${esc(item.summary)}</small></div><button data-sample="${item.key}">Open</button></div>`).join(""));
  $("dialog").querySelectorAll("[data-sample]").forEach((button)=>button.addEventListener("click",async()=>{
    try {
      button.disabled=true;
      const p=await api("/api/projects",{method:"POST",json:{demo:button.dataset.sample}});
      $("dialog").close();
      await projectList();
      await loadProject(p.id,p);
      status("Synthetic workbench opened.");
    } catch(error) { button.disabled=false; failed(error); }
  }));
}

function importDialog() {
  modal("Import sensor capture",`<form class="modal-content">
    <p>Grayscale PNG, PGM, NPY, MosaicLab capture bundle or raw raster with a declared byte layout.</p>
    <label class="field"><span>Capture file</span><input id="capture-file" type="file" required></label>
    ${input("capture-label","Capture name","","text","maxlength='80'")}
    ${select("capture-role","Capture role",[["scene","Scene"],["dark","Dark reference"],["flat","Flat reference"],["defects","Defect mask"]],"scene")}
    ${check("manual-sensor","Override sensor metadata")}${check("binary-capture","Describe a raw binary layout")}
    <div id="upload-sensor" hidden>${metadataFields({})}</div><fieldset id="upload-binary" hidden disabled>
      ${fields([input("bin-w","Width","","number","min='2' max='8192'"),input("bin-h","Height","","number","min='2' max='8192'")],"pair")}
      ${select("bin-type","Sample type",state.options.binary_encodings.map((item)=>[item.id,item.label]),"u16le")}
      <p id="bin-hint"></p>
      ${fields([input("bin-offset","Byte offset","","number","min='0'"),input("bin-stride","Row stride","","number","min='1'")],"pair")}
      ${input("bin-max","Code maximum","","number","min='1' max='65535'")}
      <button type="button" id="bin-check">Check byte layout</button>
      <div id="bin-evidence" aria-live="polite"></div>
    </fieldset><button class="accent" type="submit">Import capture</button></form>`);
  bind("manual-sensor",()=>{$("upload-sensor").hidden=!checked("manual-sensor");},"change");
  bind("binary-capture",()=>{
    $("upload-binary").hidden=!checked("binary-capture");
    $("upload-binary").disabled=!checked("binary-capture");
  },"change");
  const layout = ()=>{
    const binary={width:numeric("bin-w"),height:numeric("bin-h"),encoding:value("bin-type")};
    for (const [key,id] of [["offset","bin-offset"],["row_stride","bin-stride"],["maximum","bin-max"]]) {
      if (value(id)!=="") binary[key]=numeric(id);
    }
    return binary;
  };
  const form = $("dialog").querySelector("form");
  const evidence = $("bin-evidence");
  const layoutHint = ()=>{
    const item=state.options.binary_encodings.find((option)=>option.id===value("bin-type"));
    $("bin-hint").textContent=`${item.group_samples} ${item.group_samples===1?"site":"sites"} per ${item.group_bytes}-byte group. Maximum code ${item.maximum}.`;
  };
  layoutHint();
  let revision=0;
  form.addEventListener("input",()=>{revision++; evidence.innerHTML="";});
  bind("bin-type",layoutHint,"change");
  bind("bin-check",async()=>{
    const file=$("capture-file").files[0];
    if (!file) { status("Choose a capture file first.",true); return; }
    const button=$("bin-check");
    const generation=revision;
    const data=new FormData();
    data.set("capture",file);
    data.set("binary",JSON.stringify(layout()));
    button.disabled=true;
    evidence.textContent="Checking file…";
    try {
      const report=await api("/api/binary-check",{method:"POST",body:data});
      if (generation!==revision) return;
      evidence.innerHTML=facts([
        ["Pixel data per row",`${report.active_row_bytes} bytes`],
        ["Stored row stride",`${report.row_stride} bytes`],
        ["Omitted header / trailing padding",`${report.header_bytes} / ${report.last_row_padding_bytes} bytes`],
        ["Code range",`${report.observed_minimum}–${report.observed_maximum}`],
        ["First row begins",report.edge_sites.first_row_start.join(", ")],
        ["Last row ends",report.edge_sites.last_row_end.join(", ")],
        ["File SHA-256",report.sha256],
      ])+table(["Raster row/column parity","Min","Median","Max","At maximum"],report.raster_parity.map((sites)=>[
        `${sites.row_parity} / ${sites.column_parity}`,sites.minimum,number(sites.median,2),sites.maximum,sites.at_declared_maximum,
      ]));
    } catch(error) { if (generation===revision) evidence.textContent=error.message; }
    finally { button.disabled=false; }
  });
  formAction(async()=>{
    const form=new FormData();
    form.set("capture",$("capture-file").files[0]);
    form.set("name",value("capture-label"));
    form.set("role",value("capture-role"));
    if (checked("manual-sensor")) form.set("sensor",JSON.stringify(sensorValues()));
    if (checked("binary-capture")) form.set("binary",JSON.stringify(layout()));
    const frame=await api(endpoint("/frames"),{method:"POST",body:form});
    state.project=await api(endpoint());
    await setFrame(frame.id);
    await projectList();
    status("Capture imported.");
  });
}

function metadataFields(sensor) {
  const numericExtra="step='any'";
  return select("s-pattern","Sensor Bayer pattern",state.options.patterns,sensor.pattern??"RGGB")+
    fields([input("s-x","Sensor origin x",sensor.origin_x??0),input("s-y","Sensor origin y",sensor.origin_y??0)],"pair")+
    fields([input("s-black","Black level or 4 comma-separated",sensor.black?.join(",")??"","text"),input("s-white","White level or 4 comma-separated",sensor.white?.join(",")??"","text")],"pair")+
    fields([input("s-exposure","Exposure (ms)",sensor.exposure_ms??"","number",numericExtra),input("s-gain","Analog gain",sensor.analog_gain??"","number",numericExtra)],"pair")+
    input("s-model","Camera model",sensor.model??"","text","maxlength='80'")+
    fields(["x","y","width","height"].map((key)=>input(`s-box-${key}`,`Active ${key}`,sensor.active?.[key]??"")),"quad");
}

function sensorValues() {
  const out={pattern:value("s-pattern"),origin_x:numeric("s-x"),origin_y:numeric("s-y"),model:value("s-model")};
  for (const key of ["black","white"]) {
    const text=value(`s-${key}`).trim();
    if (text.length) out[key]=text.includes(",")?text.split(",").map(Number):Number(text);
  }
  if(value("s-exposure")!=="")out.exposure_ms=numeric("s-exposure");
  if(value("s-gain")!=="")out.analog_gain=numeric("s-gain");
  const fields=["x","y","width","height"];
  if(fields.some((key)=>value(`s-box-${key}`)!=="")) out.active=Object.fromEntries(fields.map((key)=>[key,numeric(`s-box-${key}`)]));
  return out;
}

function metadataDialog() {
  const source=state.frame;
  modal("Capture details",`<form class="modal-content">${input("edit-frame-name","Capture name",source.name,"text","required maxlength='80'")}${metadataFields(source.sensor)}<div class="control-actions"><button class="accent" type="submit">Save details</button><button id="remove-frame" type="button">Remove capture</button></div></form>`);
  formAction(async()=>{
    const sensor=sensorValues();
    sensor.exposure_ms??=null;
    sensor.analog_gain??=null;
    await api(endpoint(`/frames/${source.id}`),{method:"PATCH",json:{name:value("edit-frame-name"),sensor}});
    await updateDetail();
    await setFrame(source.id,state.inspection);
    status("Capture details saved.");
  });
  bind("remove-frame",async()=>{
    if (!window.confirm(`Remove ${source.name} from this workbench?`)) return;
    try {
      await api(endpoint(`/frames/${source.id}`),{method:"DELETE"});
      $("dialog").close();
      await loadProject(state.project.id);
      status("Capture removed.");
    } catch(error) { failed(error); }
  });
}

function workbenchDialog() {
  modal("Workbench actions",`<form class="modal-content">${input("workbench-name","Name",state.project.name,"text","required maxlength='80'")}<label class="field"><span>Notes</span><textarea id="workbench-notes" maxlength="500" rows="2">${esc(state.project.notes)}</textarea></label><div class="control-actions"><button type="submit" class="accent">Save details</button><button id="archive-project" type="button">Download archive</button><button id="import-archive" type="button">Import archive</button></div></form>`);
  formAction(async()=>{
    await api(endpoint(),{method:"PATCH",json:{name:value("workbench-name"),notes:value("workbench-notes")}});
    await updateDetail();
    status("Workbench details saved.");
  });
  bind("archive-project",async()=>{
    try {
      const file=await api(endpoint("/archive"),{blob:true});
      await receiveDownload(file);
    } catch(error){failed(error);}
  });
  bind("import-archive",()=>{
    modal("Import workbench archive",'<form class="modal-content"><label class="field"><span>Workbench archive</span><input id="archive-file" type="file" accept=".zip" required></label><button type="submit" class="accent">Open imported workbench</button></form>');
    formAction(async()=>{
      const bundle=new FormData();
      bundle.set("archive",$("archive-file").files[0]);
      const p=await api("/api/project-archives",{method:"POST",body:bundle});
      await projectList();
      await loadProject(p.id,p);
      status("Archive restored as a new workbench.");
    });
  });
}

async function receiveDownload(response) {
  const blob=await response.blob();
  const header=response.headers.get("content-disposition")??"";
  const utf=header.match(/filename\*=UTF-8''([^;]+)/i);
  const ascii=header.match(/filename="?([^";]+)"?/i);
  const name=utf?decodeURIComponent(utf[1]):ascii?.[1]??"mosaiclab";
  const anchor=document.createElement("a");
  const url=URL.createObjectURL(blob);
  anchor.download=name;
  anchor.href=url;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1500);
  status("Download prepared.");
}

async function saveDownload(path,extra) {
  const file=await api(endpoint(path),{method:"POST",json:{inspection:state.inspection,...extra},blob:true});
  await receiveDownload(file);
}

function exportDialog() {
  modal("Export current capture",`<form class="modal-content">
    ${select("export-kind","Output",[["png8","Developed PNG · 8 bit"],["png16","Developed PNG · 16 bit"],["linear16","Linear PNG · 16 bit"],["npz","Linear float arrays · NPZ"],["raw","Portable RAW capture"],["report","Development report · JSON"],["review-zip","Review package · ZIP"]],"png8")}
    ${check("export-roi","Use selected region")}
    <div id="rotation-field" hidden>
      ${select("export-turns","Clockwise rotation",[[0,"None"],[1,"90°"],[2,"180°"],[3,"270°"]],0)}
      ${select("export-encoding","Pixel encoding",[["","Portable PGM"],...state.options.binary_encodings.map((item)=>[item.id,item.label])],"")}
      <div id="alignment-field" hidden>${select("export-alignment","Binary row alignment",[[1,"Tight"],[16,"16 bytes"],[32,"32 bytes"],[64,"64 bytes"]],1)}</div>
    </div>
    <button type="submit" class="accent">Download</button></form>`);
  bind("export-kind",()=>{$("rotation-field").hidden=value("export-kind")!=="raw";},"change");
  bind("export-encoding",()=>{$("alignment-field").hidden=!value("export-encoding");},"change");
  formAction(()=>{
    const kind=value("export-kind");
    const raw=kind==="raw";
    const encoding=raw?value("export-encoding"):"";
    return saveDownload("/download",{options:{
      kind,crop_region:checked("export-roi"),rotation:raw?numeric("export-turns"):0,
      raw_encoding:encoding||null,raw_alignment:encoding?numeric("export-alignment"):1,
    }});
  });
}

$("projects").addEventListener("change",()=>loadProject($("projects").value).catch(failed));
$("frames").addEventListener("click",(event)=>{
  const id=event.target.closest("[data-frame]")?.dataset.frame;
  if(id)setFrame(id).catch(failed);
});
$("tabs").addEventListener("click",(event)=>{if(event.target.dataset.tab)setTab(event.target.dataset.tab).catch(failed);});
bind("new-project",()=>nameDialog("New workbench","Workbench name","Untitled workbench",async(name)=>{
  const p=await api("/api/projects",{method:"POST",json:{name}});
  await projectList();
  await loadProject(p.id,p);
  status("Workbench created.");
}));
bind("samples",samplesDialog);
bind("import",importDialog);
bind("download",exportDialog);
bind("project-menu",workbenchDialog);
bind("save",()=>nameDialog("Save inspection","Inspection name","Sensor review",async(title)=>{
  await post("/sessions",{title});
  await updateDetail();
  status("Inspection saved.");
}));
bind("layer",()=>{state.inspection.view.layer=$("layer").value;refreshPreview().catch(failed);},"change");
bind("probe-tool",()=>{state.tool="probe";paintTool();});
bind("roi-tool",()=>{state.tool="region";paintTool();});
bind("zoom",()=>{state.zoom=!state.zoom;$("photo").classList.toggle("zoomed",state.zoom);paintTool();});
init();
