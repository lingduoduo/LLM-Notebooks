'use strict';
// Imported labels and text enter the DOM only through textContent / text nodes.
let data = JSON.parse(document.getElementById('dashboard-data').textContent);
const $ = id => document.getElementById(id);
const ui = {tab:'infer', room:'ambiguous', selected:'room:ambiguous', edge:null,
  concept:'concept:latex', conceptSearch:'', behavior:'event:ambiguous', query:null,
  queryParams:{operation:'neighbors',node:'room:ambiguous',target:'concept:adult_content',depth:2,text:'latex',relation:''},
  last:data.last_execution, showAll:false};
const colors = {content:'#467cc2',format:'#8a65c4',interaction:'#279783',identity:'#c18a29',commerce:'#c16f95',generic:'#73839d',Room:'#245bc5',ReviewTask:'#198363',SourceField:'#7390b8'};
const edgeLabels = {IS_A:'is a',ASSIGNED_TO:'matched concept',INFERRED_AS:'inferred ancestor',HAS_REVIEW:'needs review',MAPS_TO:'maps to'};
const stages = {
 build:['Ontology structure','Browse the categories and inspect a concept’s relationships, properties and vocabulary.'],
 map:['From source records to room entities','Select a CSV record to see exactly how its fields become a room in the ontology.'],
 query:['Explore the knowledge graph','Search entities, inspect neighbors or follow a path through the connected concepts.'],
 infer:['Follow the evidence','See how “sexy latex” becomes matched concepts, inherited knowledge and a review flag.'],
 behaviors:['Give the ontology behavior','Inspect the actions, read-only functions and events registered with the room ontology.'],
 sandbox:['Try an action on the graph','Execute a real workflow behavior on a private graph copy and inspect what changed.']
};
function el(tag, cls, text) { const n=document.createElement(tag); if(cls)n.className=cls; if(text!==undefined)n.textContent=text; return n; }
function add(parent,...children) { parent.append(...children.filter(n=>n!==null&&n!==undefined)); return parent; }
function button(text, callback, cls='') { const n=el('button',cls,text); n.type='button'; n.onclick=callback; return n; }
function title(text){return el('h3','',text);}
function help(text){return el('p','help',text);}
function badge(text,cls=''){return el('span','badge '+cls,text);}
function fields(parent, entries) { const dl=el('dl'); for(const [key,value] of entries){ const row=el('div'); add(row,el('dt','',key),el('dd','',Array.isArray(value)?value.join(', ')||'None':String(value??'None')));dl.append(row);}parent.append(dl); }
function status(text,error=false){$('status').textContent=text;$('status').className='status'+(error?' error':'');}
function nodes(){return data.graph.nodes;}
function node(id){return nodes().find(n=>n.id===id);}
function label(id){const n=node(id);return n?name(n):id.replace(/^(concept|room|review):/,'');}
function name(n){if(n.type==='Room')return 'Room: '+n.properties.room_id;if(n.type==='ReviewTask')return 'Context review';return n.properties.label||n.id;}
function currentRoom(){return node('room:'+ui.room);}
function roomNodes(){return nodes().filter(n=>n.type==='Room');}
function conceptColor(n){return colors[n.type]||colors[n.properties.axis]||colors.generic;}
function behavior(){const [kind,name]=ui.behavior.split(':');return {kind,name,definition:data.behaviors[{action:'actions',function:'functions',event:'events'}[kind]][name]};}
function selectNode(id){const n=node(id); if(!n && !id.startsWith('source:'))return;ui.selected=id;ui.edge=null;if(n?.type==='Room')ui.room=n.properties.room_id;if(ui.tab==='build'){ui.concept=id;ui.showAll=false;}render();}
function selectionChoice(parent,n,callback,active,subtitle){const b=button('',callback,'choice'+(active?' active':''));b.dataset.nodeId=n.id;add(b,el('strong','',name(n)),el('small','',subtitle||n.properties.subject||n.properties.axis||n.id));parent.append(b);return b;}
function roomChoices(parent){const list=el('div','selection-list');for(const n of roomNodes()){const b=selectionChoice(list,n,()=>{ui.room=n.properties.room_id;ui.selected=n.id;ui.edge=null;ui.query=null;render();},ui.room===n.properties.room_id,n.properties.subject);if(n.properties.requires_context_review)b.append(badge('Needs context','review'));}parent.append(list);}
function selectInput(parent,id,labelText,items,value,onchange){const lab=el('label','',labelText);lab.htmlFor=id;const s=el('select');s.id=id;for(const [v,text]of items){const o=el('option','',text);o.value=v;s.append(o);}s.value=value;s.onchange=()=>onchange(s.value);add(parent,lab,s);return s;}
function textInput(parent,id,labelText,value,onchange,type='text'){const lab=el('label','',labelText);lab.htmlFor=id;const input=el('input');input.id=id;input.type=type;input.value=value;input.oninput=()=>onchange(input.value);add(parent,lab,input);return input;}
let pendingRequest=false;
function applyPending(){
 $('workspace').inert=pendingRequest;
 $('workspace').setAttribute('aria-busy',String(pendingRequest));
 for(const control of document.querySelectorAll('button,input,select'))control.disabled=pendingRequest;
}
function beginRequest(){if(pendingRequest)return false;pendingRequest=true;applyPending();return true;}
function endRequest(){pendingRequest=false;applyPending();}
async function api(path, body){const response=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const value=await response.json();if(!response.ok)throw new Error(value.error||'The request failed.');return value;}
function behaviorChoices(parent){for(const [kind,section]of Object.entries({action:'actions',function:'functions',event:'events'})){parent.append(title(kind==='action'?'Actions':kind==='function'?'Functions':'Events'));for(const [id,entry]of Object.entries(data.behaviors[section])){const b=button('',()=>{ui.behavior=kind+':'+id;render();},'choice'+(ui.behavior===kind+':'+id?' active':''));b.dataset.behavior=kind+':'+id;add(b,el('strong','',id),el('small','',entry.condition||entry.operation||entry.steps.map(s=>s.operation).join(', ')));parent.append(b);}}}
function renderControls(){const p=$('controls');p.replaceChildren();
 if(ui.tab==='build'){
  add(p,title('Concepts'),help('Choose a concept to focus its branch.'));
  const input=textInput(p,'concept-search','Search concepts',ui.conceptSearch,value=>{ui.conceptSearch=value;renderConceptList();});input.placeholder='Label or concept ID';
  p.append(button(ui.showAll?'Focus selected branch':'Show all 83 concepts',()=>{ui.showAll=!ui.showAll;render();},'wide'));
  const list=el('div','concept-list');list.id='concept-list';p.append(list);renderConceptList();
 }else if(ui.tab==='query'){
  add(p,title('Query controls'));
  selectInput(p,'query-operation','Operation',[['search','Search entities'],['neighbors','Direct neighbors'],['bfs','Explore N hops'],['path','Shortest path']],ui.queryParams.operation,value=>{ui.queryParams.operation=value;renderControls();});
  const op=ui.queryParams.operation;
  if(op==='search')textInput(p,'query-text','Search text',ui.queryParams.text,v=>ui.queryParams.text=v);
  else textInput(p,'query-node','Starting node ID',ui.queryParams.node,v=>ui.queryParams.node=v);
  if(op==='path')textInput(p,'query-target','Destination node ID',ui.queryParams.target,v=>ui.queryParams.target=v);
  if(op==='bfs'){const input=textInput(p,'query-depth','Maximum hops',ui.queryParams.depth,v=>ui.queryParams.depth=Number(v),'number');input.min='0';input.max='20';}
  if(op!=='search')selectInput(p,'query-relation','Relationship',[['','All relationships'],...Object.entries(edgeLabels).filter(([id])=>id!=='MAPS_TO').map(([id,text])=>[id,text+' ('+id+')'])],ui.queryParams.relation,v=>ui.queryParams.relation=v);
  const run=button('Run query',()=>runQuery(run),'primary wide');p.append(run);
  p.append(button('Clear highlights',()=>{ui.query=null;render();status('Showing the selected room’s graph.');},'wide'));
  add(p,title('Example'),help('Try concept:latex → concept:adult_content with “Shortest path” and “is a”.'));
  p.append(button('Load path example',()=>{ui.queryParams={...ui.queryParams,operation:'path',node:'concept:latex',target:'concept:adult_content',relation:'IS_A'};renderControls();},'wide'));
 }else if(ui.tab==='behaviors'||ui.tab==='sandbox'){
  behaviorChoices(p);if(ui.tab==='sandbox')p.append(button('Reset sandbox',()=>resetSandbox(),'wide'));
 }else{
  add(p,title('Room records'),help(ui.tab==='map'?'Actual records from the sample CSV.':'Choose a room to follow its classification.'));roomChoices(p);
  if(ui.tab==='infer'){add(p,title('Inference steps'),help('1. Match vocabulary in the source text.\n2. Follow parent relationships.\n3. Retain any context-review flag.'));}
 }
}
function renderConceptList(){const list=$('concept-list');if(!list)return;list.replaceChildren();const q=ui.conceptSearch.toLowerCase();const found=nodes().filter(n=>n.type==='Concept'&&(name(n).toLowerCase().includes(q)||n.id.toLowerCase().includes(q)));for(const n of found)selectionChoice(list,n,()=>selectNode(n.id),ui.concept===n.id);if(!found.length)list.append(help('No concepts match. Try another label or ID.'));}
async function runQuery(run){if(!beginRequest())return;try{const params={...ui.queryParams};if(!params.relation)delete params.relation;ui.query=await api('/api/query',params);ui.edge=null;if(ui.query.length)ui.selected=(ui.query[0].node||ui.query[0]).id;render();status(ui.query.length?`${ui.query.length} query result${ui.query.length===1?'':'s'}. Select a node to inspect it.`:'No results. Try another node, relationship or search term.');}catch(error){status(error.message,true);}finally{endRequest();}}
function relationKey(e){return [e.source,e.type,e.target].join('|');}
function focusIds(){
 if(ui.tab==='build'){
  if(ui.showAll)return new Set(nodes().filter(n=>n.type==='Concept').map(n=>n.id));
  const ids=new Set([ui.concept]);let n=node(ui.concept);
  while(n?.properties.parent_id){ids.add('concept:'+n.properties.parent_id);n=node('concept:'+n.properties.parent_id);}
  const selected=node(ui.concept);for(const n of nodes())if(n.type==='Concept'&&(n.properties.parent_id===selected?.properties.id||n.properties.parent_id===selected?.properties.parent_id))ids.add(n.id);
  return ids;
 }
 if(ui.tab==='query'&&ui.query!==null){const ids=new Set();for(const result of ui.query){ids.add((result.node||result).id);for(const id of result.path||[])ids.add(id);}return ids;}
 const root='room:'+ui.room;const ids=new Set([root]);for(const e of data.graph.edges)if(e.source===root)ids.add(e.target);
 return ids;
}
let camera=null, fitted=null, drag=null;
function svgEl(tag,attrs={},text){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;}
function mapPreview(){const room=currentRoom();const props=room.properties;const nodes=[{...room}];const edges=[];const values={room_id:props.room_id,subject:props.subject,tags:props.tags,self_declared_concepts:props.self_declared_concepts};for(const [field,source]of Object.entries(data.mapping)){const id='source:'+field;nodes.push({id,type:'SourceField',properties:{label:source,value:values[field],field}});edges.push({source:id,target:room.id,type:'MAPS_TO',properties:{field}});}return {nodes,edges};}
function renderGraph(){const svg=$('graph');svg.replaceChildren();const isMap=ui.tab==='map';let visible,edges;
 if(isMap){const mapped=mapPreview();visible=mapped.nodes;edges=mapped.edges;}
 else {const ids=focusIds();visible=nodes().filter(n=>ids.has(n.id));edges=data.graph.edges.filter(e=>ids.has(e.source)&&ids.has(e.target));}
 $('graph-count').textContent=`${visible.length} nodes · ${edges.length} relationships`;
 $('graph-message').hidden=visible.length>0;
 $('graph-message').textContent='No entities match this query. Change the search or clear the highlights.';
 const defs=svgEl('defs');for(const [id,color]of [['normal','#93a4ba'],['direct','#497ac8'],['inferred','#bf873f'],['review','#29856a']]){const marker=svgEl('marker',{id:'arrow-'+id,viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto-start-reverse'});marker.append(svgEl('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:color}));defs.append(marker);}svg.append(defs);
 const layers=new Map();for(const n of visible){let col;if(isMap)col=n.type==='Room'?2:0;else if(ui.tab==='build')col=n.properties.level||0;else col=n.type==='Room'?0:n.type==='ReviewTask'?1:4-(n.properties.level||0);if(!layers.has(col))layers.set(col,[]);layers.get(col).push(n);}
 const maxRows=Math.max(3,...[...layers.values()].map(a=>a.length));const height=Math.max(430,maxRows*105+80);const position=new Map();
 for(const [col,items]of layers){items.sort((a,b)=>name(a).localeCompare(name(b)));items.forEach((n,i)=>position.set(n.id,{x:125+col*230,y:((i+1)/(items.length+1))*height}));}
 const width=Math.max(500,...[...position.values()].map(p=>p.x+125));fitted={x:0,y:0,w:width,h:height};camera={...fitted};
 if(window.matchMedia('(max-width:650px)').matches&&visible.length){
  // A phone starts at a readable scale around the selection; Fit still shows the entire graph.
  const box=svg.getBoundingClientRect();const focus=position.get(ui.selected)||position.get('room:'+ui.room)||position.values().next().value;
  const mobileWidth=Math.min(width,440);const mobileHeight=mobileWidth*(box.height||380)/(box.width||366);
  camera={x:Math.max(0,focus.x-115),y:focus.y-mobileHeight/2,w:mobileWidth,h:mobileHeight};
 }
 updateCamera();
 const directEdges=edges.filter(e=>e.type!=='INFERRED_AS');const inferredEdges=edges.filter(e=>e.type==='INFERRED_AS');
 for(const e of [...inferredEdges,...directEdges]){
  const a=position.get(e.source),b=position.get(e.target);if(!a||!b)continue;
  const direction=b.x>=a.x?1:-1;const sx=a.x+direction*86,tx=b.x-direction*91;const mid=(sx+tx)/2;
  const isInferred=e.type==='INFERRED_AS';const c1=isInferred?a.y-55:a.y;const c2=isInferred?b.y-55:b.y;
  const d=`M${sx},${a.y} C${mid},${c1} ${mid},${c2} ${tx},${b.y}`;
  const kind=e.type==='HAS_REVIEW'?'review':e.type==='ASSIGNED_TO'?'direct':isInferred?'inferred':'normal';
  const g=svgEl('g',{'class':'graph-edge'+(ui.edge&&relationKey(ui.edge)===relationKey(e)?' selected':''),tabindex:0,role:'button','aria-label':`${label(e.source)} ${edgeLabels[e.type]} ${label(e.target)}`,'data-edge-type':e.type});
  g.append(svgEl('path',{d,'class':'edge-hit'}));g.append(svgEl('path',{d,'class':'edge-line',stroke:{normal:'#93a4ba',direct:'#497ac8',inferred:'#bf873f',review:'#29856a'}[kind],'stroke-dasharray':isInferred?'7 5':'none','marker-end':'url(#arrow-'+kind+')',opacity:isInferred?.65:1}));
  // Long inferred arcs are identified in the legend and inspector; label direct edges on canvas.
  if(!isInferred)g.append(svgEl('text',{x:mid,y:(a.y+b.y)/2-9,'text-anchor':'middle','class':'edge-label'},edgeLabels[e.type]));
  const choose=()=>{ui.edge=e;ui.selected=null;renderInspector();for(const group of svg.querySelectorAll('.graph-edge'))group.classList.remove('selected');g.classList.add('selected');for(const group of svg.querySelectorAll('.graph-node'))group.classList.remove('selected');};
  g.onclick=choose;g.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();choose();}};svg.append(g);
 }
 for(const n of visible){const p=position.get(n.id);const color=conceptColor(n);const g=svgEl('g',{'class':'graph-node'+(ui.selected===n.id?' selected':''),transform:`translate(${p.x-86},${p.y-33})`,tabindex:0,role:'button','aria-label':name(n),'data-node-id':n.id});
  g.append(svgEl('rect',{width:172,height:66,rx:8,fill:n.type==='ReviewTask'?'#e9f7ef':n.type==='Room'?'#edf3ff':'#fff',stroke:color}));g.append(svgEl('rect',{x:0,y:9,width:4,height:48,rx:2,fill:color,stroke:'none'}));
  const words=name(n);g.append(svgEl('text',{x:86,y:28,'text-anchor':'middle','class':'node-label'},words.length>22?words.slice(0,20)+'…':words));
  const sub=n.type==='SourceField'?String(Array.isArray(n.properties.value)?n.properties.value.join(', '):n.properties.value):n.type==='Room'?'Room record':n.type==='ReviewTask'?'Pending task':n.properties.axis;
  g.append(svgEl('text',{x:86,y:49,'text-anchor':'middle','class':'node-meta'},sub.length>25?sub.slice(0,23)+'…':sub));
  g.append(svgEl('title',{},name(n)+'\n'+n.id));g.onclick=()=>{if(n.type==='SourceField'){ui.selected=n.id;ui.edge=null;renderGraph();renderInspector();}else selectNode(n.id);};g.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();g.onclick();}};svg.append(g);
 }
}
function updateCamera(){if(camera)$('graph').setAttribute('viewBox',`${camera.x} ${camera.y} ${camera.w} ${camera.h}`);}
function zoom(factor){if(!camera||!fitted)return;const ratio=camera.w/fitted.w;if(ratio*factor<.2||ratio*factor>3)return;camera.x+=camera.w*(1-factor)/2;camera.y+=camera.h*(1-factor)/2;camera.w*=factor;camera.h*=factor;updateCamera();}
function renderLegend(){const target=$('legend');target.replaceChildren();let items;
 if(ui.tab==='map')items=[['CSV field',colors.SourceField],['Room entity',colors.Room]];
 else if(ui.tab==='build')items=Object.entries(colors).filter(([key])=>!['Room','ReviewTask','SourceField'].includes(key)).map(([name,color])=>[name,color]);
 else items=[['Room',colors.Room],['Concept',colors.content],['Direct match','#497ac8'],['Inferred ancestor (dashed)','#bf873f'],['Review task',colors.ReviewTask]];
 for(const [name,color]of items){const row=el('span');const sw=el('i','swatch');sw.style.background=color;add(row,sw,document.createTextNode(name));target.append(row);}
}
function evidenceBlock(parent,evidence,concept){for(const e of evidence||[]){const box=el('div','evidence');add(box,el('strong','',e.matched_term?'Matched “'+e.matched_term+'”':'Explicit self-declaration'),el('p','',e.text),el('small','muted',(e.field==='subject'?'Room subject':e.field)+' · '+concept));parent.append(box);}}
function showPath(parent,path){const row=el('div','path');path.forEach((id,i)=>{if(i)row.append(document.createTextNode('→'));row.append(el('span','',label(id.startsWith('concept:')?id:'concept:'+id)));});parent.append(row);}
function inspectEdge(parent,e){add(parent,title(e.type==='INFERRED_AS'?'Inference explanation':'Relationship'),el('div','inspector-title',edgeLabels[e.type]||e.type));showPath(parent,[e.source,e.target]);
 if(e.type==='ASSIGNED_TO'){parent.append(help('This direct assignment is supported by the room’s source text.'));evidenceBlock(parent,e.properties.evidence,label(e.target));}
 if(e.type==='INFERRED_AS'){parent.append(help('This ancestor follows from a directly assigned concept through the taxonomy.'));for(const path of e.properties.supporting_paths||[]){showPath(parent,path);const direct=data.graph.edges.find(d=>d.source===e.source&&d.target==='concept:'+path[0]&&d.type==='ASSIGNED_TO');if(direct)evidenceBlock(parent,direct.properties.evidence,label(direct.target));}}
 if(e.type==='IS_A'){parent.append(help(label(e.source)+' is a more specific concept under '+label(e.target)+'.'));}
 if(e.type==='HAS_REVIEW')parent.append(el('p','notice success','A pending context-review task is linked to this room.'));
 if(e.type==='MAPS_TO'){fields(parent,[['CSV column',data.mapping[e.properties.field]],['Room property',e.properties.field],['Value',currentRoom().properties[e.properties.field]]]);}
 fields(parent,[['Relationship',e.type],['Version',e.properties.taxonomy_version||data.taxonomy.version]]);
}
function inspectNode(parent,n){if(!n)return;
 add(parent,title(n.type==='Concept'?'Concept details':n.type==='Room'?'Room details':'Review task'),el('div','inspector-title',name(n)),el('div','identifier',n.id));
 if(n.type==='Concept'){
  fields(parent,[['Axis',n.properties.axis],['Category',n.properties.category_id],['Level',n.properties.level],['Parent',n.properties.parent_id?label('concept:'+n.properties.parent_id):'Root']]);
  const words=data.keywords.filter(k=>k.concept_id===n.properties.id);add(parent,title('Vocabulary'),help(words.length?words.slice(0,16).map(k=>k.keyword).join(', '):'No keywords map directly to this concept.'));
  const children=nodes().filter(c=>c.type==='Concept'&&c.properties.parent_id===n.properties.id);if(children.length){parent.append(title('More specific concepts'));for(const c of children)parent.append(button(name(c),()=>selectNode(c.id),'choice'));}
 }else if(n.type==='Room'){
  fields(parent,[['Subject',n.properties.subject||'(empty)'],['Tags',n.properties.tags],['Declared',n.properties.self_declared_concepts]]);
  if(ui.tab==='map'){add(parent,title('Mapping validation'),el('p','notice success','Valid room ID, string subject and up to five tags. Record mapped successfully.'));return;}
  const classified=data.classifications[n.properties.room_id];if(!classified)return;
  if(classified.requires_context_review)parent.append(el('p','notice','Context review required. An ambiguous keyword matched; classification alone does not resolve its meaning.'));
  else parent.append(badge('No context-review flag','success'));
  parent.append(title('Direct assignments'));
  for(const a of classified.assignments){const b=button('',()=>{const e=data.graph.edges.find(e=>e.source===n.id&&e.target==='concept:'+a.concept_id&&e.type==='ASSIGNED_TO');ui.edge=e;ui.selected=null;renderInspector();renderGraph();},'choice');add(b,el('strong','',label('concept:'+a.concept_id)),el('small','',a.evidence.map(e=>e.matched_term||'Self-declared').join(', ')));parent.append(b);}
  if(!classified.assignments.length)parent.append(help('No direct concepts matched.'));
  parent.append(title('Inferred ancestors'));add(parent,help(classified.inferred_ancestors.map(id=>label('concept:'+id)).join(', ')||'None'));
 }else if(n.type==='ReviewTask'){fields(parent,[['Room',n.properties.room_id],['Status',n.properties.status]]);parent.append(help('Created by the review action on the sandbox graph. It does not change the original CSV or taxonomy.'));}
}
function renderInspector(){const p=$('inspector');p.replaceChildren();
 if(ui.tab==='behaviors'){inspectBehavior(p);return;}
 if(ui.tab==='sandbox'){sandboxControls(p);return;}
 if(ui.tab==='query'&&ui.query!==null){p.append(title('Query results'));p.append(help(ui.query.length?`${ui.query.length} result${ui.query.length===1?'':'s'}`:'No results. Change the query or clear highlights.'));const list=el('ul','result-list');for(const result of ui.query){const n=result.node||result;const item=el('li');item.append(button(name(n),()=>{ui.selected=n.id;ui.edge=null;renderGraph();renderInspector();},'link-button'));if(result.path)showPath(item,result.path);list.append(item);}p.append(list);}
 if(ui.edge){inspectEdge(p,ui.edge);return;}
 if(ui.tab==='map'&&ui.selected?.startsWith('source:')){const field=ui.selected.slice(7);add(p,title('Source field'),el('div','inspector-title',data.mapping[field]));fields(p,[['Maps to',field],['Value',currentRoom().properties[field]]]);return;}
 inspectNode(p,node(ui.selected)||currentRoom());
}
const operationHelp={classify_room:'Classify a room and replace its direct and inherited assignments.',create_review_task:'Create one pending review task, or reuse an existing pending task.',room_assignments:'Read direct and inherited assignments for one room.',pending_reviews:'Read all pending context-review tasks without changing the graph.'};
function inspectBehavior(p){const b=behavior();add(p,title('Behavior definition'),el('div','inspector-title',b.name),badge(b.kind));fields(p,[['Parameters',b.definition.parameters],['Changes graph',b.kind==='function'?'No':'On a sandbox copy']]);
 if(b.kind==='event')add(p,title('Condition'),help(b.definition.condition),title('Dispatches action'),help(b.definition.action));
 const steps=b.kind==='action'?b.definition.steps:b.kind==='function'?[b.definition]:data.behaviors.actions[b.definition.action].steps;
 for(const step of steps)add(p,title(step.operation),help(operationHelp[step.operation]));
 p.append(button('Open in sandbox',()=>switchTab('sandbox'),'primary wide'));
}
function renderBehaviorFlow(){const target=$('behavior-flow');target.replaceChildren();const b=behavior();add(target,el('p','flow-heading',b.name),help(b.kind==='function'?'A read-only graph query.':'Inputs, conditions and steps are declared in the behavior registry.'));
 const sequence=el('div','flow-sequence');const cards=[['Parameters',b.definition.parameters.length?b.definition.parameters.join(', '):'No input parameters',b.kind]];
 if(b.kind==='event'){cards.push(['Condition',b.definition.condition+' = true','event']);cards.push(['Dispatch action',b.definition.action,'action']);}
 const steps=b.kind==='action'?b.definition.steps:b.kind==='function'?[b.definition]:data.behaviors.actions[b.definition.action].steps;
 for(const s of steps)cards.push([s.operation,operationHelp[s.operation],b.kind==='function'?'function':'action']);
 cards.push([b.kind==='function'?'Query result':'Result',b.kind==='function'?'Returned data; graph unchanged':'Updated sandbox graph and execution trace',b.kind]);
 cards.forEach(([name,desc,kind],index)=>{if(index)sequence.append(el('div','flow-connector','↓'));const card=el('div','flow-card '+kind);add(card,el('strong','',name),el('p','',desc));sequence.append(card);});target.append(sequence);
}
function sandboxControls(p){const b=behavior();add(p,title('Execute behavior'),el('div','inspector-title',b.name),badge(b.kind));
 if(b.definition.parameters.includes('room_id'))selectInput(p,'sandbox-room','Room',roomNodes().map(n=>[n.properties.room_id,n.properties.room_id+' — '+n.properties.subject]),ui.room,value=>{ui.room=value;ui.selected='room:'+value;ui.edge=null;renderGraph();renderInspector();});
 if(b.kind==='event')p.append(el('p','notice','Condition: '+b.definition.condition+'. The selected room must already be classified.'));
 else p.append(help(b.kind==='function'?'This function reads the current sandbox graph.': 'Runs against the sandbox copy. Source records remain unchanged.'));
 const run=button('Run '+b.kind,()=>runBehavior(run),'primary wide');p.append(run);
 if(ui.edge){p.append(title('Selected relationship'));inspectEdge(p,ui.edge);}
 else if(ui.selected&&node(ui.selected)?.type==='ReviewTask')inspectNode(p,node(ui.selected));
 if(ui.last){const result=ui.last;const added=result.changes.nodes.added.length;const removed=result.changes.nodes.removed.length;const updated=result.changes.nodes.updated.length;
  add(p,title('Last execution'),el('p','notice success',`${added} node${added===1?'':'s'} added · ${removed} removed · ${updated} updated`));
  for(const trace of result.trace){const row=el('div','trace-step');if(trace.event)add(row,el('strong','',trace.event),el('small','',trace.condition+' → '+(trace.matched?'matched; action dispatched':'not matched; no action')));else add(row,el('strong','',trace.operation),el('small','',trace.params.room_id?'Room: '+trace.params.room_id:'All pending review tasks'));p.append(row);}
  if(Array.isArray(result.result)){p.append(title('Returned results'));if(!result.result.length)p.append(help('No records returned.'));for(const n of result.result){if(n.type==='ReviewTask')add(p,el('p','notice success',n.id+' · '+n.properties.status));else if(n.target)p.append(help(label(n.target)+' — '+edgeLabels[n.type]));}}
  for(const n of result.changes.nodes.added){add(p,title('New review task'),el('p','notice success',n.id+' · '+n.properties.status));}
  if(!added&&result.trace.some(t=>t.operation==='create_review_task'))p.append(help('An existing pending review task was reused. No duplicate was created.'));
 }else add(p,title('Ready to run'),help('Try the ambiguous event on room “ambiguous”. A review task will appear in green on the graph.'));
}
async function runBehavior(run){if(!beginRequest())return;const b=behavior();try{const params=b.definition.parameters.includes('room_id')?{room_id:ui.room}:{};const result=await api('/api/behavior',{kind:b.kind,name:b.name,params});data.graph=result.graph;ui.last=result;ui.edge=null;ui.selected=result.changes.nodes.added[0]?.id||'room:'+ui.room;render();status(`${b.name} finished. ${result.changes.nodes.added.length} nodes added; source data unchanged.`);}catch(error){status(error.message,true);}finally{endRequest();}}
async function resetSandbox(){if(!beginRequest())return;try{data=await api('/api/reset',{});ui.last=null;ui.edge=null;ui.selected='room:'+ui.room;ui.query=null;render();status('Sandbox restored to the original classified graph.');}catch(error){status(error.message,true);}finally{endRequest();}}
function switchTab(tab){if(pendingRequest)return;ui.tab=tab;ui.edge=null;ui.query=null;ui.selected=tab==='build'?ui.concept:'room:'+ui.room;render();status('');}
function render(){
 for(const tab of document.querySelectorAll('[role=tab]')){const active=tab.dataset.tab===ui.tab;tab.setAttribute('aria-selected',String(active));tab.tabIndex=active?0:-1;}
 $('workspace').setAttribute('aria-labelledby','tab-'+ui.tab);$('stage-title').textContent=stages[ui.tab][0];$('stage-description').textContent=stages[ui.tab][1];
 $('taxonomy-version').textContent='Taxonomy v'+data.taxonomy.version;
 const counts=$('stage-counts');counts.replaceChildren();const reviews=nodes().filter(n=>n.type==='ReviewTask').length;
 for(const [value,label]of [[data.taxonomy.nodes.length,'concepts'],[roomNodes().length,'rooms'],[reviews,'review tasks']]){const box=el('div');add(box,el('strong','',value),el('span','',label));counts.append(box);}
 $('canvas-title').textContent={build:'Taxonomy branch',map:'CSV fields → Room',query:'Query graph',infer:'Room → Concepts → Ancestors',behaviors:'Behavior flow',sandbox:'Sandbox knowledge graph'}[ui.tab];
 const flow=ui.tab==='behaviors';$('graph-area').hidden=flow;$('graph-tools').hidden=flow;$('legend').hidden=flow;$('behavior-flow').hidden=!flow;
 const mapping=$('mapping-preview');mapping.replaceChildren();if(ui.tab==='map'){const strip=el('div','mapping-strip');for(const [field,column]of Object.entries(data.mapping))strip.append(el('span','',column+' → '+field));mapping.append(strip);}
 renderControls();if(flow){$('graph-count').textContent='Registered operations and conditions';renderBehaviorFlow();}else{renderGraph();renderLegend();}renderInspector();
 applyPending();
 $('canvas-hint').textContent=flow?'Select a behavior, then open it in the sandbox.':ui.tab==='infer'?'Solid edges = direct matches. Dashed amber edges = inferred ancestors.':'Select a node or relationship to inspect it.';
}
for(const tab of document.querySelectorAll('[role=tab]')){tab.onclick=()=>switchTab(tab.dataset.tab);tab.onkeydown=event=>{const all=[...document.querySelectorAll('[role=tab]')];const i=all.indexOf(tab);let next;if(event.key==='ArrowRight')next=(i+1)%all.length;else if(event.key==='ArrowLeft')next=(i+all.length-1)%all.length;else if(event.key==='Home')next=0;else if(event.key==='End')next=all.length-1;else return;event.preventDefault();switchTab(all[next].dataset.tab);all[next].focus();};}
$('zoom-in').onclick=()=>zoom(.8);$('zoom-out').onclick=()=>zoom(1.25);$('fit-graph').onclick=()=>{camera={...fitted};updateCamera();};
$('graph').addEventListener('wheel',e=>{e.preventDefault();zoom(e.deltaY>0?1.12:.89);},{passive:false});
$('graph').onpointerdown=e=>{if(e.target.closest('.graph-node,.graph-edge'))return;drag={x:e.clientX,y:e.clientY,camera:{...camera}};$('graph').setPointerCapture(e.pointerId);};
$('graph').onpointermove=e=>{if(!drag)return;const box=$('graph').getBoundingClientRect();camera.x=drag.camera.x-(e.clientX-drag.x)*camera.w/box.width;camera.y=drag.camera.y-(e.clientY-drag.y)*camera.h/box.height;updateCamera();};
$('graph').onpointerup=()=>drag=null;$('graph').onpointercancel=()=>drag=null;
render();
