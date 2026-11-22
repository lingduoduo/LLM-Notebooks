'use strict';
// Imported labels and text enter the DOM only through textContent / text nodes.
let data = JSON.parse(document.getElementById('dashboard-data').textContent);
const $ = id => document.getElementById(id);
const ui = {tab:'infer', room:'ambiguous', selected:'room:ambiguous', edge:null,
  concept:'concept:latex', conceptSearch:'', behavior:'event:ambiguous', query:null,
  queryParams:{operation:'neighbors',node:'room:ambiguous',target:'concept:adult_content',depth:2,text:'latex',relation:''},
  last:data.last_execution, showAll:false,
  // Canvas view per tab, rule-inference results, mapping draft/preview and the sandbox execution log.
  view:{build:'schema',map:'record',query:'full',infer:'focus',sandbox:'full'},
  inference:null, rule:null, queryCenter:null, mapPreview:null, mapError:null, mapOpen:false,
  mapDraft:{csv:data.csv_text,mapping:{...data.mapping}}, log:[]};
ui.queryParams.pattern="MATCH (r:Room)-[:ASSIGNED_TO]->(c:Concept) WHERE c.axis = 'content' RETURN r, c";
const kindSections={action:'actions',function:'functions',event:'events'};
const ACCENT='#e08a1e';
const colors = {content:'#467cc2',format:'#8a65c4',interaction:'#279783',identity:'#c18a29',commerce:'#c16f95',generic:'#73839d',Room:'#245bc5',ReviewTask:'#198363',SourceField:'#7390b8'};
const edgeLabels = {IS_A:'is a',ASSIGNED_TO:'matched concept',INFERRED_AS:'inferred ancestor',HAS_REVIEW:'needs review',MAPS_TO:'maps to'};
const stages = {
 build:['Ontology structure','Inspect the entity types, their properties and relationships, then browse the concept taxonomy.'],
 map:['From source records to room entities','Configure how CSV columns become room properties, preview the resulting graph and apply it.'],
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
// Graphs are replaced, never mutated, so an index keyed on the array stays valid; label() runs per edge.
let nodeIndex={list:null,byId:null};
function node(id){const list=nodes();if(nodeIndex.list!==list)nodeIndex={list,byId:new Map(list.map(n=>[n.id,n]))};return nodeIndex.byId.get(id);}
function label(id){if(id.startsWith('type:'))return id.slice(5);const n=node(id);return n?name(n):id.replace(/^(concept|room|review):/,'');}
function name(n){if(n.type==='Room')return 'Room: '+n.properties.room_id;if(n.type==='ReviewTask')return 'Context review';return n.properties.label||n.id;}
function currentRoom(){return node('room:'+ui.room);}
function roomNodes(){return nodes().filter(n=>n.type==='Room');}
function conceptColor(n){if(n.type==='Schema')return n.properties.kind==='Concept'?colors.content:colors[n.properties.kind]||colors.generic;return colors[n.type]||colors[n.properties.axis]||colors.generic;}
function typeProperties(type){const keys=new Set(data.schema.types[type]||[]);for(const n of nodes())if(n.type===type)Object.keys(n.properties).forEach(k=>keys.add(k));return [...keys];}
function behavior(){const [kind,name]=ui.behavior.split(':');return {kind,name,definition:data.behaviors[{action:'actions',function:'functions',event:'events'}[kind]][name]};}
function selectNode(id){const n=node(id); if(!n && !id.startsWith('source:'))return;ui.selected=id;ui.edge=null;if(n?.type==='Room')ui.room=n.properties.room_id;if(ui.tab==='build'){ui.concept=id;ui.showAll=false;ui.view.build='taxonomy';}render();}
function selectionChoice(parent,n,callback,active,subtitle){const b=button('',callback,'choice'+(active?' active':''));b.dataset.nodeId=n.id;add(b,el('strong','',name(n)),el('small','',subtitle||n.properties.subject||n.properties.axis||n.id));parent.append(b);return b;}
function roomChoices(parent){const list=el('div','selection-list');for(const n of roomNodes()){const b=selectionChoice(list,n,()=>{ui.room=n.properties.room_id;ui.selected=n.id;ui.edge=null;ui.query=null;if(ui.tab==='infer')ui.view.infer='focus';render();},ui.room===n.properties.room_id,n.properties.subject);if(n.properties.requires_context_review)b.append(badge('Needs context','review'));}parent.append(list);}
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
function behaviorSteps(kind,entry){return kind==='action'?entry.steps.map(s=>s.operation):kind==='function'?[entry.operation]:['if '+entry.condition,'then '+entry.action];}
function behaviorChoices(parent,showTiles=false){
 if(showTiles){const tiles=el('div','tiles');for(const [kind,section]of Object.entries(kindSections)){const t=el('div','tile');add(t,el('strong','',String(Object.keys(data.behaviors[section]).length)),el('span','',kind==='function'?'functions':kind+'s'));tiles.append(t);}parent.append(tiles);}
 for(const [kind,section]of Object.entries(kindSections)){parent.append(title(kind==='action'?'Actions':kind==='function'?'Functions':'Events'));
  for(const [id,entry]of Object.entries(data.behaviors[section])){const b=button('',()=>{ui.behavior=kind+':'+id;render();},'choice'+(ui.behavior===kind+':'+id?' active':''));b.dataset.behavior=kind+':'+id;
   const chips=el('div','chips');for(const param of entry.parameters)chips.append(el('span','chip param',param));if(!entry.parameters.length)chips.append(el('span','chip','no input'));
   const chain=el('div','chain');behaviorSteps(kind,entry).forEach((step,i)=>{if(i)chain.append(el('span','arrow','→'));chain.append(el('span','chip step '+kind,step));});
   add(b,el('strong','',id),chips,chain);parent.append(b);}}
}
// Which entity types the registered behaviors read or create (the reference's "has behavior" badges).
function behaviorTypes(parent){parent.append(title('Entity types'));const uses={Room:0,ReviewTask:0,Concept:0};
 for(const section of Object.values(kindSections))for(const entry of Object.values(data.behaviors[section])){if(entry.parameters.includes('room_id'))uses.Room++;const ops=entry.steps?entry.steps.map(s=>s.operation):[entry.operation];if(ops.some(op=>op==='create_review_task'||op==='pending_reviews'))uses.ReviewTask++;}
 for(const [type,count]of Object.entries(uses)){const row=el('div','type-row');add(row,el('strong','',type),count?badge(count+' behavior'+(count===1?'':'s'),'success'):badge('no behavior'));parent.append(row);}
}
function nodeInput(parent,id,labelText,value,onchange){const input=textInput(parent,id,labelText,value,onchange);input.setAttribute('list','node-options');input.placeholder='Type a label or ID';return input;}
function renderControls(){const p=$('controls');p.replaceChildren();
 if(ui.tab==='build'){
  add(p,title('Entity types'),help('Classes in the ontology schema with their instance and property counts.'));
  for(const type of Object.keys(data.schema.types)){const count=nodes().filter(n=>n.type===type).length;const active=ui.view.build==='schema'&&ui.selected==='type:'+type;
   const b=button('',()=>{ui.view.build='schema';ui.selected='type:'+type;ui.edge=null;render();},'choice compact'+(active?' active':''));b.dataset.type=type;
   add(b,el('strong','',type),el('small','',`${count} instance${count===1?'':'s'} · ${typeProperties(type).length} properties`));p.append(b);}
  add(p,title('Concepts'),help('Choose a concept to focus its taxonomy branch.'));
  const input=textInput(p,'concept-search','Search concepts',ui.conceptSearch,value=>{ui.conceptSearch=value;renderConceptList();});input.placeholder='Label or concept ID';
  p.append(button(ui.showAll?'Focus selected branch':'Show all 83 concepts',()=>{ui.showAll=!ui.showAll;ui.view.build='taxonomy';ui.selected=ui.concept;render();},'wide'));
  const list=el('div','concept-list');list.id='concept-list';p.append(list);renderConceptList();
 }else if(ui.tab==='query'){
  add(p,title('Query controls'));
  selectInput(p,'query-operation','Operation',[['search','Search entities'],['neighbors','Direct neighbors'],['bfs','Explore N hops'],['path','Shortest path'],['pattern','Pattern query (Cypher-like)']],ui.queryParams.operation,value=>{ui.queryParams.operation=value;renderControls();});
  const op=ui.queryParams.operation;
  if(op==='search')textInput(p,'query-text','Search text',ui.queryParams.text,v=>ui.queryParams.text=v);
  else if(op==='pattern'){const lab=el('label','','Pattern');lab.htmlFor='query-pattern';const ta=el('textarea','code-input');ta.id='query-pattern';ta.rows=5;ta.spellcheck=false;ta.value=ui.queryParams.pattern;ta.oninput=()=>ui.queryParams.pattern=ta.value;add(p,lab,ta,help('MATCH (a:Type)-[:REL]->(b) WHERE a.prop = "value" RETURN a, b. Up to three hops; conditions use = or CONTAINS.'));}
  else nodeInput(p,'query-node','Starting node ID',ui.queryParams.node,v=>ui.queryParams.node=v);
  if(op==='path')nodeInput(p,'query-target','Destination node ID',ui.queryParams.target,v=>ui.queryParams.target=v);
  if(op==='bfs'){const input=textInput(p,'query-depth','Maximum hops',ui.queryParams.depth,v=>ui.queryParams.depth=Number(v),'number');input.min='0';input.max='20';}
  if(!['search','pattern'].includes(op))selectInput(p,'query-relation','Relationship',[['','All relationships'],...Object.entries(edgeLabels).filter(([id])=>id!=='MAPS_TO').map(([id,text])=>[id,text+' ('+id+')'])],ui.queryParams.relation,v=>ui.queryParams.relation=v);
  const run=button('Run query',()=>runQuery(),'primary wide');p.append(run);
  p.append(button('Clear highlights',()=>{ui.query=null;ui.queryCenter=null;render();status('Showing the whole graph.');},'wide'));
  add(p,title('Examples'),help('Click a node in the full graph to expand its direct neighbors.'));
  p.append(button('Load path example',()=>{ui.queryParams={...ui.queryParams,operation:'path',node:'concept:latex',target:'concept:adult_content',relation:'IS_A'};renderControls();},'wide'));
  p.append(button('Load pattern example',()=>{ui.queryParams={...ui.queryParams,operation:'pattern',pattern:"MATCH (r:Room)-[:ASSIGNED_TO]->(c)-[:IS_A]->(p) WHERE r.room_id = 'clear' RETURN r, p"};renderControls();},'wide'));
 }else if(ui.tab==='behaviors'){behaviorChoices(p,true);behaviorTypes(p);
 }else if(ui.tab==='sandbox'){behaviorChoices(p);p.append(button('Reset sandbox',()=>resetSandbox(),'wide'));
 }else if(ui.tab==='map'){
  add(p,title('Target: Room entity'),help('Properties each CSV row must provide.'));
  for(const [field,type,required]of [['room_id','string',1],['subject','string',1],['tags','array of strings',1],['self_declared_concepts','array of strings',0]]){const row=el('div','type-row');add(row,el('span','',''),el('code','',field),el('small','',type),badge(required?'required':'optional',required?'review':''));row.firstChild.remove();p.append(row);}
  const count=roomNodes().length;p.append(el('p','notice success',`${count} room${count===1?'':'s'} mapped in the workspace.`));
  add(p,title('Room records'),help('Records currently in the workspace.'));roomChoices(p);
 }else{
  add(p,title('Room records'),help('Choose a room to follow its classification.'));roomChoices(p);
  add(p,title('Inference rules'),help('Conjunctive rules evaluated on the current sandbox graph. Select one to filter results.'));
  for(const rule of data.rules){const count=ui.inference?ui.inference.inferences.filter(i=>i.rule===rule.name).length:null;
   const b=button('',()=>{ui.rule=ui.rule===rule.name?null:rule.name;if(ui.inference)ui.view.infer='full';ui.selected=null;ui.edge=null;render();},'choice rule'+(ui.rule===rule.name?' active':''));b.dataset.rule=rule.name;
   add(b,el('strong','',rule.label),el('code','rule-text',rule.text));if(count!==null)b.append(badge(count+' inferred',count?'review':''));p.append(b);}
  p.append(button('Run inference',()=>runInference(),'primary wide'));
  if(ui.inference)p.append(button('Clear inference',()=>{ui.inference=null;ui.rule=null;ui.view.infer='focus';ui.selected='room:'+ui.room;render();status('');},'wide'));
 }
}
function renderConceptList(){const list=$('concept-list');if(!list)return;list.replaceChildren();const q=ui.conceptSearch.toLowerCase();const found=nodes().filter(n=>n.type==='Concept'&&(name(n).toLowerCase().includes(q)||n.id.toLowerCase().includes(q)));for(const n of found)selectionChoice(list,n,()=>selectNode(n.id),ui.concept===n.id);if(!found.length)list.append(help('No concepts match. Try another label or ID.'));}
function queryRequest(){const q=ui.queryParams,op=q.operation;const r={operation:op};
 if(op==='search')r.text=q.text;else if(op==='pattern')r.text=q.pattern;
 else{r.node=q.node;if(op==='path')r.target=q.target;if(op==='bfs')r.depth=q.depth;if(q.relation)r.relation=q.relation;}return r;}
async function runQuery(center=null){if(!beginRequest())return;try{ui.queryCenter=center;ui.query=await api('/api/query',queryRequest());ui.edge=null;if(center)ui.selected=center;else if(ui.query.length)ui.selected=(ui.query[0].node||ui.query[0]).id;render();status(ui.query.length?`${ui.query.length} query result${ui.query.length===1?'':'s'}. Select a node to inspect it.`:'No results. Try another node, relationship or search term.');}catch(error){status(error.message,true);}finally{endRequest();}}
function expandNode(id){ui.queryParams={...ui.queryParams,operation:'neighbors',node:id,relation:''};runQuery(id);}
async function runInference(){if(!beginRequest())return;try{ui.inference=await api('/api/infer',{});ui.view.infer='full';ui.selected=null;ui.edge=null;render();const n=ui.inference.inferences.length,c=ui.inference.contradictions.length;status(`Inference complete: ${n} facts inferred, ${c} contradiction${c===1?'':'s'}.`);}catch(error){status(error.message,true);}finally{endRequest();}}
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
let camera=null, fitted=null, fitScale=1, drag=null;
const NODE_W=172, NODE_H=66, COL_GAP=250, ROW_GAP=96, ARC_LIFT=30, ARC_STEP=18;
// Short on-canvas labels keep text inside the gap between columns; full names stay in the inspector.
const canvasLabels={IS_A:'is a',ASSIGNED_TO:'matches',HAS_REVIEW:'review',MAPS_TO:'maps to'};
const edgeColors={normal:'#93a4ba',direct:'#497ac8',inferred:'#bf873f',review:'#29856a'};
function edgeKind(e){return e.type==='HAS_REVIEW'?'review':e.type==='ASSIGNED_TO'?'direct':e.type==='INFERRED_AS'?'inferred':'normal';}
function svgEl(tag,attrs={},text){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;}
function mapPreview(){const room=currentRoom();if(!room)return {nodes:[],edges:[]};const props=room.properties;const nodes=[{...room}];const edges=[];const values={room_id:props.room_id,subject:props.subject,tags:props.tags,self_declared_concepts:props.self_declared_concepts};for(const [field,source]of Object.entries(data.mapping)){const id='source:'+field;nodes.push({id,type:'SourceField',properties:{label:source,value:values[field],field}});edges.push({source:id,target:room.id,type:'MAPS_TO',properties:{field}});}return {nodes,edges};}
// Build: tidy tree. Leaves take consecutive rows and each parent centres on its children.
function treeLayout(visible){
 const ids=new Set(visible.map(n=>n.id));const kids=new Map();const roots=[];
 for(const n of visible){const parent='concept:'+n.properties.parent_id;if(n.properties.parent_id&&ids.has(parent)){if(!kids.has(parent))kids.set(parent,[]);kids.get(parent).push(n);}else roots.push(n);}
 const levels=[...new Set(visible.map(n=>n.properties.level||0))].sort((a,b)=>a-b);
 const pos=new Map();let slot=0;
 const place=n=>{const c=kids.get(n.id)||[];let y;if(c.length){const ys=c.map(place);y=(ys[0]+ys[ys.length-1])/2;}else y=(slot++)*ROW_GAP;pos.set(n.id,{x:levels.indexOf(n.properties.level||0)*COL_GAP,y});return y;};
 for(const r of roots){place(r);slot+=.4;}
 return pos;
}
// Room graphs: fixed columns (room → deepest concept → root), rows ordered by the
// barycentre of already placed neighbours so parent/child links rarely cross.
function layeredLayout(visible,edges,isMap){
 const maxLevel=Math.max(0,...visible.filter(n=>n.type==='Concept').map(n=>n.properties.level||0));
 const column=n=>isMap?(n.type==='Room'?1:0):n.type==='Room'?0:n.type==='ReviewTask'?1:1+maxLevel-(n.properties.level||0);
 const cols=new Map();for(const n of visible){const c=column(n);if(!cols.has(c))cols.set(c,[]);cols.get(c).push(n);}
 const links=edges.filter(e=>e.type!=='INFERRED_AS');const pos=new Map();
 [...cols.keys()].sort((a,b)=>a-b).forEach((c,ci)=>{const items=cols.get(c);
  if(!isMap){const bary=new Map(items.map(n=>{const ys=links.flatMap(e=>e.source===n.id?[e.target]:e.target===n.id?[e.source]:[]).map(id=>pos.get(id)?.y).filter(y=>y!==undefined);return [n.id,ys.length?ys.reduce((a,b)=>a+b,0)/ys.length:Infinity];}));
   const rank=n=>n.type==='ReviewTask'?0:1;
   items.sort((a,b)=>rank(a)-rank(b)||(bary.get(a.id)-bary.get(b.id))||name(a).localeCompare(name(b)));}
  items.forEach((n,i)=>pos.set(n.id,{x:ci*COL_GAP,y:(i-(items.length-1)/2)*ROW_GAP}));});
 return pos;
}
// Spread several edges meeting one side of a node so arrows and labels do not stack.
function directRoutes(edges,pos){
 const groups=new Map();const ends=new Map();
 const push=(key,item)=>{if(!groups.has(key))groups.set(key,[]);groups.get(key).push(item);};
 for(const e of edges){if(e.type==='INFERRED_AS')continue;const a=pos.get(e.source),b=pos.get(e.target);if(!a||!b)continue;
  const dir=b.x>a.x?1:b.x<a.x?-1:0;const sSide=dir>=0?1:-1,tSide=dir>0?-1:1;
  const key=relationKey(e);ends.set(key,{e,dir,sSide,tSide});
  push(e.source+'|'+sSide,{key,end:'s',other:b.y});push(e.target+'|'+tSide,{key,end:'t',other:a.y});}
 const offset=new Map();
 for(const items of groups.values()){items.sort((p,q)=>p.other-q.other);const span=Math.min(NODE_H-22,(items.length-1)*14);items.forEach((it,i)=>offset.set(it.key+it.end,items.length>1?-span/2+i*span/(items.length-1):0));}
 const routes=[];
 for(const [key,{e,dir,sSide,tSide}]of ends){const a=pos.get(e.source),b=pos.get(e.target);
  const sx=a.x+sSide*NODE_W/2,sy=a.y+offset.get(key+'s'),tx=b.x+tSide*(NODE_W/2+5),ty=b.y+offset.get(key+'t');
  const bend=dir===0?70:Math.max(40,Math.abs(tx-sx)/2);
  const c1x=sx+sSide*bend,c2x=tx+tSide*bend;
  routes.push({e,d:`M${sx},${sy} C${c1x},${sy} ${c2x},${ty} ${tx},${ty}`,label:{x:(sx+3*c1x+3*c2x+tx)/8,y:(sy+3*sy+3*ty+ty)/8}});}
 return routes;
}
// Inferred ancestors skip several columns, so route them as nested arcs above or
// below the node rows instead of drawing them through intermediate concepts.
function inferredRoutes(edges,pos,visible){
 const ys=[...pos.values()].map(p=>p.y);const top=Math.min(...ys)-NODE_H/2,bottom=Math.max(...ys)+NODE_H/2;
 const sameCol=b=>visible.map(n=>pos.get(n.id)).filter(p=>p&&p.x===b.x).map(p=>p.y);
 const groups={up:[],down:[]};
 for(const e of edges){if(e.type!=='INFERRED_AS')continue;const a=pos.get(e.source),b=pos.get(e.target);if(!a||!b)continue;
  const col=sameCol(b);const side=b.y<=Math.min(...col)?'up':b.y>=Math.max(...col)?'down':'up';
  groups[side].push({e,a,b,middle:b.y>Math.min(...col)&&b.y<Math.max(...col)});}
 const routes=[];let lift={up:0,down:0};const r=12;
 for(const side of ['up','down']){const items=groups[side].sort((p,q)=>Math.abs(p.b.x-p.a.x)-Math.abs(q.b.x-q.a.x));const s=side==='up'?-1:1;
  items.forEach((it,k)=>{const{a,b,e}=it;const level=(side==='up'?top:bottom)+s*(ARC_LIFT+k*ARC_STEP);lift[side]=Math.max(lift[side],ARC_LIFT+k*ARC_STEP);
   const sx=a.x+(items.length>1?60-k*120/(items.length-1):40),sy=a.y+s*NODE_H/2;
   let d=`M${sx},${sy} L${sx},${level-s*r} Q${sx},${level} ${sx+r},${level}`;
   if(it.middle){const gx=b.x-NODE_W/2-24-k*4,ty=b.y+14;d+=` L${gx-r},${level} Q${gx},${level} ${gx},${level-s*r} L${gx},${ty+s*r} Q${gx},${ty} ${gx+r},${ty} L${b.x-NODE_W/2-5},${ty}`;}
   else{const ty=b.y+s*(NODE_H/2+5);d+=` L${b.x-r},${level} Q${b.x},${level} ${b.x},${level-s*r} L${b.x},${ty}`;}
   routes.push({e,d});});}
 return {routes,top:top-lift.up,bottom:bottom+lift.down};
}
// Build tab schema view: entity types as nodes, allowed relationships as edges.
function schemaModel(){const make=(kind,text)=>({id:'type:'+kind,type:'Schema',properties:{label:text||kind,kind}});
 const visible=[make('CSV','CSV record'),...Object.keys(data.schema.types).map(kind=>make(kind))];
 const edges=Object.entries(data.schema.relations).map(([type,[s,t]])=>({source:'type:'+s,target:'type:'+t,type,properties:{}}));
 edges.push({source:'type:CSV',target:'type:Room',type:'MAPS_TO',properties:{}});return {visible,edges};}
function schemaLayout(){return new Map([['type:CSV',{x:0,y:0}],['type:Room',{x:COL_GAP,y:0}],['type:Concept',{x:2*COL_GAP,y:-60}],['type:ReviewTask',{x:2*COL_GAP,y:90}]]);}
function schemaMeta(n){const kind=n.properties.kind;if(kind==='CSV')return `${roomNodes().length} rows`;const count=nodes().filter(x=>x.type===kind).length;return `${count} instances · ${typeProperties(kind).length} props`;}
// Truncate by rendered width (not character count) so labels stay inside their node box.
const measure=document.createElement('canvas').getContext('2d');
function fitText(text,font,max){measure.font=font+" 'Avenir Next','Segoe UI',sans-serif";if(measure.measureText(text).width<=max)return text;let t=text;while(t.length>1&&measure.measureText(t+'…').width>max)t=t.slice(0,-1);return t.trimEnd()+'…';}
function addMarkers(svg){const defs=svgEl('defs');for(const [id,color]of Object.entries(edgeColors)){const marker=svgEl('marker',{id:'arrow-'+id,viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto-start-reverse'});marker.append(svgEl('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:color}));defs.append(marker);}svg.append(defs);}
function renderGraph(){const svg=$('graph');svg.replaceChildren();fullCameraKey=null;const isMap=ui.tab==='map';const isSchema=ui.tab==='build'&&ui.view.build==='schema';let visible,edges;
 if(isMap){const mapped=mapPreview();visible=mapped.nodes;edges=mapped.edges;}
 else if(isSchema){const model=schemaModel();visible=model.visible;edges=model.edges;}
 else {const ids=focusIds();visible=nodes().filter(n=>ids.has(n.id));edges=data.graph.edges.filter(e=>ids.has(e.source)&&ids.has(e.target));}
 $('graph-count').textContent=`${visible.length} nodes · ${edges.length} relationships`;
 $('graph-message').hidden=visible.length>0;
 $('graph-message').textContent='No entities match this query. Change the search or clear the highlights.';
 addMarkers(svg);
 if(!visible.length){fitted=camera={x:0,y:0,w:800,h:480};updateCamera();return;}
 const position=isSchema?schemaLayout():ui.tab==='build'?treeLayout(visible):layeredLayout(visible,edges,isMap);
 const arcs=inferredRoutes(edges,position,visible);
 const xs=[...position.values()].map(p=>p.x);
 const loopRoom=isSchema?110:0;
 setCamera({x:Math.min(...xs)-NODE_W/2-30,y:arcs.top-24,w:Math.max(...xs)-Math.min(...xs)+NODE_W+60+loopRoom,h:arcs.bottom-arcs.top+48},position.get(ui.selected)||position.get('room:'+ui.room)||position.values().next().value);
 const drawEdge=(e,d,labelAt)=>{
  const kind=edgeKind(e);const isInferred=kind==='inferred';
  const g=svgEl('g',{'class':'graph-edge'+(ui.edge&&relationKey(ui.edge)===relationKey(e)?' selected':''),tabindex:0,role:'button','aria-label':`${label(e.source)} ${edgeLabels[e.type]} ${label(e.target)}`,'data-edge-type':e.type});
  g.append(svgEl('path',{d,'class':'edge-hit'}));g.append(svgEl('path',{d,'class':'edge-line',stroke:edgeColors[kind],'stroke-dasharray':isInferred?'7 5':'none','marker-end':'url(#arrow-'+kind+')',opacity:isInferred?.7:1}));
  if(labelAt&&canvasLabels[e.type])g.append(svgEl('text',{x:labelAt.x,y:labelAt.y-6,'text-anchor':'middle','class':'edge-label'},canvasLabels[e.type]));
  const choose=()=>{ui.edge=e;ui.selected=null;renderInspector();for(const group of svg.querySelectorAll('.graph-edge'))group.classList.remove('selected');g.classList.add('selected');for(const group of svg.querySelectorAll('.graph-node'))group.classList.remove('selected');};
  g.onclick=choose;g.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();choose();}};svg.append(g);
 };
 for(const {e,d}of arcs.routes)drawEdge(e,d,null);
 for(const {e,d,label:at}of directRoutes(edges,position))drawEdge(e,d,at);
 for(const n of visible){const p=position.get(n.id);const color=conceptColor(n);const g=svgEl('g',{'class':'graph-node'+(ui.selected===n.id?' selected':''),transform:`translate(${p.x-NODE_W/2},${p.y-NODE_H/2})`,tabindex:0,role:'button','aria-label':name(n),'data-node-id':n.id});
  g.append(svgEl('rect',{width:NODE_W,height:NODE_H,rx:8,fill:n.type==='ReviewTask'?'#e9f7ef':n.type==='Room'?'#edf3ff':'#fff',stroke:color}));g.append(svgEl('rect',{x:0,y:9,width:4,height:NODE_H-18,rx:2,fill:color,stroke:'none'}));
  g.append(svgEl('text',{x:NODE_W/2,y:28,'text-anchor':'middle','class':'node-label'},fitText(name(n),'600 18px',NODE_W-18)));
  const value=n.properties.value;const sub=n.type==='SourceField'?(Array.isArray(value)?value.join(', '):String(value??''))||'(empty)':n.type==='Schema'?schemaMeta(n):n.type==='Room'?'Room record':n.type==='ReviewTask'?'Pending task':n.properties.axis;
  g.append(svgEl('text',{x:NODE_W/2,y:49,'text-anchor':'middle','class':'node-meta'+(sub==='(empty)'?' empty':'')},fitText(sub,'13px',NODE_W-18)));
  g.append(svgEl('title',{},name(n)+'\n'+n.id));g.onclick=()=>{if(n.type==='SourceField'||n.type==='Schema'){ui.selected=n.id;ui.edge=null;renderGraph();renderInspector();if(n.type==='Schema')renderControls();}else selectNode(n.id);};g.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();g.onclick();}};svg.append(g);
 }
}
// The camera always matches the canvas aspect ratio, so pan and zoom maths are exact.
function canvasSize(){const box=$('graph').getBoundingClientRect();return {w:box.width||800,h:box.height||480};}
// Full-graph view (Query, Infer, Sandbox, mapping result): force-directed overview of every entity.
let forceCache={key:null,pos:new Map()},fullCameraKey=null;
function hash01(text){let h=2166136261;for(let i=0;i<text.length;i++)h=Math.imul(h^text.charCodeAt(i),16777619);return (h>>>0)/4294967296;}
function forceLayout(graph){
 const ids=graph.nodes.map(n=>n.id);const key=ids.join('|')+'#'+graph.edges.length;if(forceCache.key===key)return forceCache.pos;
 // Seeded and warm-started from the previous layout, so positions stay stable when a node is added.
 const index=new Map(ids.map((id,i)=>[id,i]));const warm=forceCache.pos;
 const pts=ids.map(id=>{const p=warm.get(id);return p?{x:p.x,y:p.y}:{x:(hash01(id)-.5)*700,y:(hash01(id+'#')-.5)*700};});
 const links=graph.edges.filter(e=>e.type!=='INFERRED_AS').map(e=>[index.get(e.source),index.get(e.target)]).filter(([a,b])=>a!==undefined&&b!==undefined&&a!==b);
 const iterations=warm.size?160:380;
 for(let it=0;it<iterations;it++){const cool=1-it/iterations;const dx=new Float64Array(pts.length),dy=new Float64Array(pts.length);
  for(let i=0;i<pts.length;i++)for(let j=i+1;j<pts.length;j++){let x=pts[i].x-pts[j].x,y=pts[i].y-pts[j].y;let d2=x*x+y*y;if(d2<1){x=hash01(ids[i]+j)-.5;y=.5;d2=1;}if(d2>250000)continue;const f=900/d2;dx[i]+=x*f;dy[i]+=y*f;dx[j]-=x*f;dy[j]-=y*f;}
  for(const [a,b]of links){const x=pts[b].x-pts[a].x,y=pts[b].y-pts[a].y;const d=Math.hypot(x,y)||1;const f=.3*(d-36)/d;dx[a]+=x*f;dy[a]+=y*f;dx[b]-=x*f;dy[b]-=y*f;}
  const step=1.5+24*cool;for(let i=0;i<pts.length;i++){dx[i]-=pts[i].x*.03;dy[i]-=pts[i].y*.03;const m=Math.hypot(dx[i],dy[i]);const k=m>step?step/m:1;pts[i].x+=dx[i]*k;pts[i].y+=dy[i]*k;}}
 const pos=new Map(ids.map((id,i)=>[id,pts[i]]));forceCache={key,pos};return pos;}
function isFullView(){return (['query','infer','sandbox'].includes(ui.tab)&&ui.view[ui.tab]==='full')||(ui.tab==='map'&&ui.view.map==='result');}
function fullGraphData(){return ui.tab==='map'&&ui.mapPreview?ui.mapPreview.graph:data.graph;}
// hit: nodes to keep at full opacity (null = all). accent: nodes ringed in orange (new or inferred).
function fullHighlight(graph){
 if(ui.tab==='query'&&ui.query){const hit=new Set();for(const r of ui.query){hit.add((r.node||r).id);for(const id of r.path||[])hit.add(id);}if(ui.queryCenter)hit.add(ui.queryCenter);return {hit,accent:new Set()};}
 if(ui.tab==='infer'&&ui.inference){const items=ui.inference.inferences.filter(i=>!ui.rule||i.rule===ui.rule);const accent=new Set(items.map(i=>i.subject));if(!ui.rule)return {hit:null,accent};const hit=new Set();for(const i of items)for(const v of Object.values(i.bindings))hit.add(v);return {hit,accent};}
 if(ui.tab==='sandbox'&&ui.last){const c=ui.last.changes;return {hit:null,accent:new Set([...c.nodes.added.map(n=>n.id),...c.nodes.updated.map(u=>u.after.id),...c.edges.added.flatMap(e=>[e.source,e.target])])};}
 if(ui.tab==='map')return {hit:null,accent:new Set(graph.nodes.filter(n=>n.type==='Room').map(n=>n.id))};
 return {hit:null,accent:new Set()};
}
function fullRadius(n){return n?.type==='Room'?11:n?.type==='ReviewTask'?10:n?.type==='Concept'&&!n.properties.level?10:7;}
function renderFullGraph(){const svg=$('graph');svg.replaceChildren();addMarkers(svg);
 const graph=fullGraphData();const pos=forceLayout(graph);const {hit,accent}=fullHighlight(graph);const byId=new Map(graph.nodes.map(n=>[n.id,n]));
 $('graph-count').textContent=`${graph.nodes.length} nodes · ${graph.edges.length} relationships`+(hit?` · ${hit.size} highlighted`:accent.size?` · ${accent.size} marked`:'');
 $('graph-message').hidden=!(hit&&hit.size===0);$('graph-message').textContent='No entities match this query. Change the search or clear the highlights.';
 const focusPts=(hit&&hit.size?[...hit]:graph.nodes.map(n=>n.id)).map(id=>pos.get(id)).filter(Boolean);
 const xs=focusPts.map(p=>p.x),ys=focusPts.map(p=>p.y);
 const key=ui.tab+'|'+forceCache.key+'|'+(hit?[...hit].sort().join(','):'all');
 if(key!==fullCameraKey||!camera){const w=Math.max(Math.max(...xs)-Math.min(...xs)+160,420),h=Math.max(Math.max(...ys)-Math.min(...ys)+90,260);const cx=(Math.max(...xs)+Math.min(...xs))/2,cy=(Math.max(...ys)+Math.min(...ys))/2;setCamera({x:cx-w/2,y:cy-h/2,w,h},null);fullCameraKey=key;}else updateCamera();
 // Marks are sized in screen terms: zoomed-out overviews get proportionally larger dots and labels.
 const k=Math.min(2.6,Math.max(1,1/(canvasSize().w/camera.w)));const radius=n=>fullRadius(n)*k;
 for(const e of graph.edges){const a=pos.get(e.source),b=pos.get(e.target);if(!a||!b)continue;const on=!hit||(hit.has(e.source)&&hit.has(e.target));const kind=edgeKind(e);
  if(kind==='inferred'&&!(hit&&on))continue;
  const len=Math.hypot(b.x-a.x,b.y-a.y)||1,ra=radius(byId.get(e.source)),rb=radius(byId.get(e.target))+4*k;
  const d=`M${a.x+(b.x-a.x)*ra/len},${a.y+(b.y-a.y)*ra/len} L${b.x-(b.x-a.x)*rb/len},${b.y-(b.y-a.y)*rb/len}`;
  const g=svgEl('g',{'class':'graph-edge full'+(on?'':' dim')+(ui.edge&&relationKey(ui.edge)===relationKey(e)?' selected':''),'data-edge-type':e.type,'aria-label':`${label(e.source)} ${edgeLabels[e.type]} ${label(e.target)}`});
  g.append(svgEl('path',{d,'class':'edge-hit'}),svgEl('path',{d,'class':'edge-line',stroke:edgeColors[kind],'stroke-width':1.3*k,'stroke-dasharray':kind==='inferred'?'5 4':'none','marker-end':on&&hit?'url(#arrow-'+kind+')':'none'}));
  if(hit&&on&&canvasLabels[e.type])g.append(svgEl('text',{x:(a.x+b.x)/2,y:(a.y+b.y)/2-5*k,'text-anchor':'middle','class':'edge-label small',style:`font-size:${11*k}px;stroke-width:${4*k}px`},canvasLabels[e.type]));
  g.onclick=()=>{ui.edge=e;ui.selected=null;renderInspector();for(const x of svg.querySelectorAll('.graph-edge,.graph-node'))x.classList.remove('selected');g.classList.add('selected');};svg.append(g);}
 const labels=svgEl('g',{'class':'label-layer','aria-hidden':'true'});
 for(const n of graph.nodes){const p=pos.get(n.id);const isHit=!hit||hit.has(n.id);const r=radius(n);
  const g=svgEl('g',{'class':'graph-node full'+(isHit?'':' dim')+(accent.has(n.id)?' accent':'')+(ui.selected===n.id?' selected':''),transform:`translate(${p.x},${p.y})`,tabindex:isHit&&(hit||n.type!=='Concept'||accent.has(n.id))?0:-1,role:'button','aria-label':name(n),'data-node-id':n.id});
  if(accent.has(n.id))g.append(svgEl('circle',{r:r+5*k,'class':'accent-ring','stroke-width':3*k}));
  g.append(svgEl('circle',{r,fill:conceptColor(n),'class':'dot','stroke-width':2*k}));
  const showLabel=hit?isHit:(n.type!=='Concept'||accent.has(n.id)||(n.properties.level??0)<=1);
  if(showLabel)labels.append(svgEl('text',{x:p.x,y:p.y+r+13*k,'text-anchor':'middle','class':'full-label'+(isHit?'':' dim'),style:`font-size:${12*k}px;stroke-width:${4*k}px`},fitText(name(n),'600 12px',150)));
  g.append(svgEl('title',{},name(n)+'\n'+n.id));
  g.onclick=()=>{if(ui.tab==='query'){expandNode(n.id);return;}ui.selected=n.id;ui.edge=null;if(n.type==='Room'&&node(n.id))ui.room=n.properties.room_id;renderFullGraph();renderInspector();};
  g.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();g.onclick();}};svg.append(g);}
 svg.append(labels);
}
function setCamera(bounds,focus){
 const {w:bw,h:bh}=canvasSize();const maxScale=1.15;
 let w=Math.max(bounds.w,bw/maxScale),h=Math.max(bounds.h,bh/maxScale);
 if(w/h<bw/bh)w=h*bw/bh;else h=w*bh/bw;
 fitted={x:bounds.x+bounds.w/2-w/2,y:bounds.y+bounds.h/2-h/2,w,h};fitScale=bw/w;camera={...fitted};
 // Large graphs start at a readable scale around the selection; Fit still shows everything.
 if(fitScale<.6&&focus){const scale=bw<500?.8:.85;const cw=bw/scale,ch=bh/scale;
  const clamp=(v,lo,size,full)=>full<=size?lo+(full-size)/2:Math.min(Math.max(v,lo),lo+full-size);
  camera={x:clamp(focus.x-cw/2,bounds.x,cw,bounds.w),y:clamp(focus.y-ch/2,bounds.y,ch,bounds.h),w:cw,h:ch};}
 updateCamera();
}
function updateCamera(){if(camera)$('graph').setAttribute('viewBox',`${camera.x} ${camera.y} ${camera.w} ${camera.h}`);}
function zoom(factor,px,py){if(!camera||!fitted)return;const scale=canvasSize().w/(camera.w*factor);
 if((factor<1&&scale>2.5)||(factor>1&&scale<Math.min(.2,fitScale*.8)))return;
 const cx=px??camera.x+camera.w/2,cy=py??camera.y+camera.h/2;
 camera={x:cx-(cx-camera.x)*factor,y:cy-(cy-camera.y)*factor,w:camera.w*factor,h:camera.h*factor};updateCamera();}
function renderLegend(){const target=$('legend');target.replaceChildren();let items;
 if(isFullView()){const graph=fullGraphData();const axes=[...new Set(graph.nodes.filter(n=>n.type==='Concept').map(n=>n.properties.axis))];
  items=[['Room',colors.Room],['Review task',colors.ReviewTask],...axes.map(axis=>[axis,colors[axis]||colors.generic])];
  const {hit,accent}=fullHighlight(graph);if(accent.size)items.push([ui.tab==='infer'?'Inferred subject':ui.tab==='map'?'Mapped room':'Changed by last run',ACCENT,'ring']);if(hit)items.push(['Faded = not in result','#c9d3df']);}
 else if(ui.tab==='build'&&ui.view.build==='schema')items=[['Entity type',colors.Room],['Relationship',edgeColors.normal,'line'],['Inferred relationship',edgeColors.inferred,'dash']];
 else if(ui.tab==='map')items=[['CSV field',colors.SourceField],['Room entity',colors.Room],['Maps to',edgeColors.normal,'line']];
 else if(ui.tab==='build')items=[...Object.entries(colors).filter(([key])=>!['Room','ReviewTask','SourceField'].includes(key)).map(([axis,color])=>[axis,color]),['is a (child → parent)',edgeColors.normal,'line']];
 else items=[['Room',colors.Room],['Concept (outline = axis)',colors.generic],['Review task',colors.ReviewTask],['Direct match',edgeColors.direct,'line'],['Inferred ancestor',edgeColors.inferred,'dash'],['is a',edgeColors.normal,'line']];
 for(const [text,color,shape]of items){const row=el('span');const sw=el('i','swatch'+(shape?' '+shape:''));sw.style.setProperty('--c',color);add(row,sw,document.createTextNode(text));target.append(row);}
}
function evidenceBlock(parent,evidence,concept){for(const e of evidence||[]){const box=el('div','evidence');add(box,el('strong','',e.matched_term?'Matched “'+e.matched_term+'”':'Explicit self-declaration'),el('p','',e.text),el('small','muted',(e.field==='subject'?'Room subject':e.field)+' · '+concept));parent.append(box);}}
function showPath(parent,path){const row=el('div','path');path.forEach((id,i)=>{if(i)row.append(document.createTextNode('→'));row.append(el('span','',label(id.includes(':')?id:'concept:'+id)));});parent.append(row);}
function inspectEdge(parent,e){
 if(e.source.startsWith('type:')){const count=e.type==='MAPS_TO'?roomNodes().length:data.graph.edges.filter(x=>x.type===e.type).length;add(parent,title('Relationship type'),el('div','inspector-title',e.type));showPath(parent,[e.source,e.target]);
  parent.append(help({IS_A:'Taxonomy hierarchy: each concept points to its parent.',ASSIGNED_TO:'A concept matched directly in the room’s text, tags or declarations.',INFERRED_AS:'An ancestor concept derived by following IS_A from a direct match.',HAS_REVIEW:'A context-review task created by the review action.',MAPS_TO:'Each CSV row is mapped into one Room entity.'}[e.type]||''));
  fields(parent,[['From',label(e.source)],['To',label(e.target)],[e.type==='MAPS_TO'?'Rows':'Instances',count]]);return;}add(parent,title(e.type==='INFERRED_AS'?'Inference explanation':'Relationship'),el('div','inspector-title',edgeLabels[e.type]||e.type));showPath(parent,[e.source,e.target]);
 if(e.type==='ASSIGNED_TO'){parent.append(help('This direct assignment is supported by the room’s source text.'));evidenceBlock(parent,e.properties.evidence,label(e.target));}
 if(e.type==='INFERRED_AS'){parent.append(help('This ancestor follows from a directly assigned concept through the taxonomy.'));for(const path of e.properties.supporting_paths||[]){showPath(parent,path);const direct=data.graph.edges.find(d=>d.source===e.source&&d.target==='concept:'+path[0]&&d.type==='ASSIGNED_TO');if(direct)evidenceBlock(parent,direct.properties.evidence,label(direct.target));}}
 if(e.type==='IS_A'){parent.append(help(label(e.source)+' is a more specific concept under '+label(e.target)+'.'));}
 if(e.type==='HAS_REVIEW')parent.append(el('p','notice success','A pending context-review task is linked to this room.'));
 if(e.type==='MAPS_TO'){fields(parent,[['CSV column',data.mapping[e.properties.field]],['Room property',e.properties.field],['Value',currentRoom().properties[e.properties.field]]]);}
 fields(parent,[['Relationship',e.type],['Version',e.properties.taxonomy_version||data.taxonomy.version]]);
}
function inspectType(p,kind){
 if(kind==='CSV'){add(p,title('Source table'),el('div','inspector-title','CSV record'));fields(p,Object.entries(data.mapping).map(([field,column])=>[column,'→ '+field]));p.append(help('Each CSV row becomes one Room entity. Configure the columns on the Map tab.'));return;}
 const instances=nodes().filter(n=>n.type===kind);add(p,title('Entity type'),el('div','inspector-title',kind),el('div','identifier',`${instances.length} instance${instances.length===1?'':'s'} in the workspace`));
 p.append(title(`Properties (${typeProperties(kind).length})`));const required=new Set(data.schema.types[kind]);const table=el('div','prop-table');
 for(const key of typeProperties(kind)){const sample=instances.find(n=>key in n.properties)?.properties[key];const type=Array.isArray(sample)?'array':sample===null?'null':sample===undefined?'—':typeof sample;
  const row=el('div','prop-row');add(row,el('code','',key),el('span','prop-type',type),required.has(key)?badge('required','review'):badge('optional'));table.append(row);}
 p.append(table);p.append(title('Relationships'));
 for(const [rel,[s,t]]of Object.entries(data.schema.relations))if(s===kind||t===kind){const count=data.graph.edges.filter(e=>e.type===rel).length;const row=el('div','path');add(row,el('span','',s),document.createTextNode(rel+' →'),el('span','',t),el('small','muted',`${count}`));p.append(row);}
}
function inspectMapping(p){const r=ui.mapPreview;add(p,title(r?'Mapping preview':'Workspace mapping'));
 if(ui.mapError)p.append(el('p','notice error','Mapping failed: '+ui.mapError));
 const stats=r?r.stats:{nodes:nodes().length,edges:data.graph.edges.length,types:new Set(nodes().map(n=>n.type)).size,alerts:null};
 const tiles=el('div','tiles four');for(const [value,text,cls]of [[stats.nodes,'nodes'],[stats.edges,'relationships'],[stats.types,'node types'],[stats.alerts??'–','quality alerts',stats.alerts?'warn':'']]){const t=el('div','tile '+(cls||''));add(t,el('strong','',String(value)),el('span','',text));tiles.append(t);}p.append(tiles);
 if(r){for(const alert of r.alerts)p.append(el('p','notice','Room “'+alert.room_id+'”: '+alert.message));if(!r.alerts.length&&!ui.mapError)p.append(el('p','notice success','No quality alerts in this preview.'));}
 p.append(help(r?'Preview only. Apply it to use the new rooms in Query, Infer and Sandbox; source files never change.':'Edit the CSV or column mapping above the canvas to preview changes.'));
}
function inspectInference(p){const r=ui.inference;const sel=ui.selected&&r.inferences.some(i=>i.subject===ui.selected)?ui.selected:null;
 const items=r.inferences.filter(i=>(!ui.rule||i.rule===ui.rule)&&(!sel||i.subject===sel));
 add(p,title(`Inference results (${r.inferences.length})`));
 p.append(r.contradictions.length?el('p','notice error',r.contradictions.map(c=>`${label(c.subject)} is both ${c.facts.join(' and ')}`).join('; ')):el('p','notice success','No contradictions between inferred facts ('+data.contradiction_pairs.map(pair=>pair.join(' ⟂ ')).join(', ')+').'));
 if(sel||ui.rule)p.append(button('Show all results',()=>{ui.selected=null;ui.rule=null;render();},'link-button'));
 for(const rule of data.rules){const list=items.filter(i=>i.rule===rule.name);if(!list.length)continue;
  const head=el('div','rule-head');add(head,el('strong','',rule.label),badge(String(list.length),'review'));p.append(head,el('code','rule-text',rule.text));
  for(const inference of list){const box=el('div','derivation');box.append(el('div','derivation-title',`${inference.fact}(${label(inference.subject)})`));const ol=el('ol');
   for(const step of inference.steps){const li=el('li');add(li,el('span','',step.premise),el('small','',step.matched==='no match'?' — no matching fact':' ⇐ '+step.matched));ol.append(li);}
   box.append(ol,el('small','muted','Bindings: '+Object.entries(inference.bindings).map(([k,v])=>k+' = '+label(v)).join(', ')));p.append(box);}}
 if(!items.length)p.append(help('No inferences match this selection.'));
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
 if(ui.tab==='build'&&ui.view.build==='schema'){if(ui.edge)inspectEdge(p,ui.edge);else inspectType(p,ui.selected?.startsWith('type:')?ui.selected.slice(5):'Room');return;}
 if(ui.tab==='infer'&&ui.inference&&ui.view.infer==='full'&&!ui.edge){inspectInference(p);return;}
 if(ui.tab==='map'&&!ui.selected?.startsWith('source:'))inspectMapping(p);
 if(ui.tab==='sandbox'){sandboxControls(p);return;}
 if(ui.tab==='query'&&ui.query!==null){p.append(title('Query results'));p.append(help(ui.query.length?`${ui.query.length} result${ui.query.length===1?'':'s'}`:'No results. Change the query or clear highlights.'));const list=el('ul','result-list');for(const result of ui.query){const n=result.node||result;const item=el('li');item.append(button(result.bindings?Object.entries(result.bindings).map(([v,id])=>v+' = '+label(id)).join(' · '):name(n),()=>{ui.selected=n.id;ui.edge=null;renderGraph();renderInspector();},'link-button'));if(result.path)showPath(item,result.path);list.append(item);}p.append(list);}
 if(ui.edge){inspectEdge(p,ui.edge);return;}
 if(ui.tab==='map'&&ui.selected?.startsWith('source:')){const field=ui.selected.slice(7);add(p,title('Source field'),el('div','inspector-title',data.mapping[field]));fields(p,[['Maps to',field],['Value',currentRoom().properties[field]]]);return;}
 inspectNode(p,node(ui.selected)||(ui.tab==='map'&&ui.mapPreview?.graph.nodes.find(n=>n.id===ui.selected))||currentRoom());
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
function sandboxControls(p){const b=behavior();add(p,title('Execution panel'));
 const tabs=el('div','segmented full-width');for(const [kind,text]of [['action','Action'],['function','Function'],['event','Event']]){const t=button(text,()=>{ui.behavior=kind+':'+Object.keys(data.behaviors[kindSections[kind]])[0];render();},b.kind===kind?'active':'');t.setAttribute('aria-pressed',String(b.kind===kind));tabs.append(t);}p.append(tabs);
 selectInput(p,'sandbox-behavior',{action:'Action',function:'Function',event:'Event'}[b.kind],Object.keys(data.behaviors[kindSections[b.kind]]).map(k=>[k,k]),b.name,v=>{ui.behavior=b.kind+':'+v;render();});
 if(b.kind==='event')p.append(el('div','payload-title','Event payload'));
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
 }else add(p,title('Ready to run'),help('Try the ambiguous event on room “ambiguous”. The new review task is ringed in orange on the graph.'));
 if(ui.log.length){p.append(title('Execution log'));const log=el('div','console');for(const line of ui.log.slice(-14))log.append(el('div','console-line '+(line.cls||''),line.text));p.append(log);}
}
function logLine(text,cls=''){ui.log.push({text,cls});}
async function runBehavior(run){if(!beginRequest())return;const b=behavior();try{const params=b.definition.parameters.includes('room_id')?{room_id:ui.room}:{};const result=await api('/api/behavior',{kind:b.kind,name:b.name,params});data.graph=result.graph;ui.last=result;ui.inference=null;
 const time=new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'});logLine(`${time} ${b.kind} ${b.name}(${params.room_id||''})`,'head');
 for(const t of result.trace)logLine(t.event?`  if ${t.condition} → ${t.matched?'matched; dispatching '+b.definition.action:'not matched; no action'}`:`  ${t.step}. ${t.operation}`,t.event&&!t.matched?'warn':'');
 const c=result.changes;logLine(`  = ${c.nodes.added.length} added · ${c.nodes.updated.length} updated · ${c.edges.added.length} edges added`,'ok');ui.edge=null;ui.selected=result.changes.nodes.added[0]?.id||'room:'+ui.room;render();status(`${b.name} finished. ${result.changes.nodes.added.length} nodes added; source data unchanged.`);}catch(error){status(error.message,true);}finally{endRequest();}}
async function resetSandbox(){if(!beginRequest())return;try{data=await api('/api/reset',{});ui.last=null;ui.inference=null;logLine('Sandbox reset to the classified graph','head');ui.edge=null;ui.selected='room:'+ui.room;ui.query=null;render();status('Sandbox restored to the original classified graph.');}catch(error){status(error.message,true);}finally{endRequest();}}
function inferDescription(){const room=currentRoom();if(!room)return stages.infer[1];const flagged=data.classifications[room.properties.room_id]?.requires_context_review;return `See how “${room.properties.subject||room.properties.room_id}” becomes matched concepts, inherited knowledge${flagged?' and a review flag':''}.`;}
function switchTab(tab){if(pendingRequest)return;ui.tab=tab;ui.edge=null;ui.query=null;ui.queryCenter=null;ui.selected=tab==='build'?(ui.view.build==='schema'?'type:Room':ui.concept):'room:'+ui.room;render();status('');}
function csvHeaderGuess(text){return (text.split(/\r?\n/)[0]||'').split(',').map(h=>h.trim().replace(/^"|"$/g,'')).filter(Boolean);}
let previewTimer=null;
function schedulePreview(){clearTimeout(previewTimer);previewTimer=setTimeout(()=>previewMapping(false,true),450);}
function renderMappingPanel(target){const draft=ui.mapDraft;const headers=csvHeaderGuess(draft.csv);
 const strip=el('div','mapping-strip');strip.id='mapping-strip';for(const [field,column]of Object.entries(draft.mapping))strip.append(el('span','',column+' → '+field));target.append(strip);
 const details=el('details','mapping-config');details.open=ui.mapOpen;details.ontoggle=()=>{ui.mapOpen=details.open;};details.append(el('summary','','Configure column mapping and edit the CSV'));
 const grid=el('div','mapping-grid');
 for(const [field,required]of [['room_id',true],['subject',true],['tags',true],['self_declared_concepts',false]]){const lab=el('label','',field+(required?' *':''));const s=el('select');s.id='map-'+field;lab.htmlFor=s.id;
  if(!required){const o=el('option','','(not mapped)');o.value='';s.append(o);}for(const h of headers){const o=el('option','',h);o.value=h;s.append(o);}
  s.value=draft.mapping[field]||'';s.onchange=()=>{if(s.value)draft.mapping[field]=s.value;else delete draft.mapping[field];const st=$('mapping-strip');st.replaceChildren(...Object.entries(draft.mapping).map(([f,c])=>el('span','',c+' → '+f)));schedulePreview();};grid.append(lab,s);}
 const ta=el('textarea','code-input csv-editor');ta.id='csv-editor';ta.rows=6;ta.spellcheck=false;ta.value=draft.csv;ta.setAttribute('aria-label','CSV source text');ta.oninput=()=>{draft.csv=ta.value;schedulePreview();};
 const actions=el('div','button-row');actions.append(button('Preview mapping',()=>previewMapping(false),'primary'),button('Apply to workspace',()=>previewMapping(true)),button('Restore sample',()=>{ui.mapDraft={csv:data.sample_csv,mapping:{...data.sample_mapping}};previewMapping(true);}));
 add(details,grid,ta,help('Edits preview live and never change rooms.csv. JSON arrays are expected in tags and declared columns.'),actions);target.append(details);
}
async function previewMapping(apply,live=false){
 if(!live&&!beginRequest())return;
 try{const result=await api('/api/mapping',{csv:ui.mapDraft.csv,mapping:ui.mapDraft.mapping,apply});ui.mapPreview=result;ui.mapError=null;
  if(apply){data=await api('/api/state');ui.inference=null;ui.last=null;ui.query=null;const rooms=roomNodes();if(!rooms.some(n=>n.properties.room_id===ui.room))ui.room=rooms[0]?.properties.room_id||'';ui.selected='room:'+ui.room;ui.mapDraft={csv:data.csv_text,mapping:{...data.mapping}};buildNodeOptions();}
  if(!live){ui.view.map='result';if(ui.selected?.startsWith('source:'))ui.selected=null;}
  status(apply?`Mapping applied: ${result.stats.rooms} rooms in the workspace. Source files unchanged.`:`Preview: ${result.stats.nodes} nodes, ${result.stats.edges} relationships, ${result.stats.alerts} quality alerts.`);
 }catch(error){ui.mapError=error.message;if(!live)status(error.message,true);}
 finally{if(!live)endRequest();}
 if(live&&ui.tab==='map'){renderCanvas();renderInspector();}else if(!live)render();
}
function renderCanvas(){if(isFullView())renderFullGraph();else renderGraph();renderLegend();}
function buildNodeOptions(){const list=$('node-options');list.replaceChildren(...nodes().map(n=>{const o=el('option');o.value=n.id;o.label=name(n);return o;}));}
const views={build:[['schema','Schema'],['taxonomy','Taxonomy']],map:[['record','Record → Room'],['result','Result graph']],query:[['full','Full graph'],['focus','Results only']],infer:[['focus','Room evidence'],['full','Full graph']],sandbox:[['full','Full graph'],['focus','Room focus']]};
function renderViewToggle(){const toggle=$('view-toggle');toggle.replaceChildren();toggle.hidden=!views[ui.tab];
 for(const [value,text]of views[ui.tab]||[]){const active=ui.view[ui.tab]===value;const b=button(text,()=>{ui.view[ui.tab]=value;ui.edge=null;if(ui.selected?.startsWith('source:'))ui.selected='room:'+ui.room;if(ui.tab==='build')ui.selected=value==='schema'?'type:Room':ui.concept;render();},active?'active':'');b.setAttribute('aria-pressed',String(active));toggle.append(b);}}
function canvasTitle(){const v=ui.view[ui.tab];return {build:v==='schema'?'Ontology schema':'Taxonomy branch',map:v==='result'?'Mapping result graph':'CSV fields → Room',query:v==='full'?'Knowledge graph':'Query results',infer:v==='full'?'Inferred facts on the graph':'Room → Concepts → Ancestors',behaviors:'Behavior flow',sandbox:v==='full'?'Sandbox knowledge graph (live)':'Sandbox room focus'}[ui.tab];}
function render(){
 for(const tab of document.querySelectorAll('[role=tab]')){const active=tab.dataset.tab===ui.tab;tab.setAttribute('aria-selected',String(active));tab.tabIndex=active?0:-1;}
 $('workspace').setAttribute('aria-labelledby','tab-'+ui.tab);$('stage-title').textContent=stages[ui.tab][0];$('stage-description').textContent=ui.tab==='infer'?inferDescription():stages[ui.tab][1];
 $('taxonomy-version').textContent='Taxonomy v'+data.taxonomy.version;
 const counts=$('stage-counts');counts.replaceChildren();const reviews=nodes().filter(n=>n.type==='ReviewTask').length;
 for(const [value,label]of [[data.taxonomy.nodes.length,'concepts'],[roomNodes().length,'rooms'],[reviews,'review tasks']]){const box=el('div');add(box,el('strong','',value),el('span','',label));counts.append(box);}
 $('canvas-title').textContent=canvasTitle();renderViewToggle();
 const flow=ui.tab==='behaviors';$('graph-area').hidden=flow;$('graph-tools').hidden=flow;$('legend').hidden=flow;$('behavior-flow').hidden=!flow;
 const mapping=$('mapping-preview');mapping.replaceChildren();if(ui.tab==='map')renderMappingPanel(mapping);
 renderControls();if(flow){$('graph-count').textContent='Registered operations and conditions';renderBehaviorFlow();}else renderCanvas();renderInspector();
 applyPending();
 $('canvas-hint').textContent=flow?'Select a behavior, then open it in the sandbox.':ui.tab==='query'&&isFullView()?'Click a node to expand its neighbors; non-results fade out.':ui.tab==='infer'&&!isFullView()?'Solid edges = direct matches. Dashed amber edges = inferred ancestors.':'Select a node or relationship to inspect it.';
}
for(const tab of document.querySelectorAll('[role=tab]')){tab.onclick=()=>switchTab(tab.dataset.tab);tab.onkeydown=event=>{const all=[...document.querySelectorAll('[role=tab]')];const i=all.indexOf(tab);let next;if(event.key==='ArrowRight')next=(i+1)%all.length;else if(event.key==='ArrowLeft')next=(i+all.length-1)%all.length;else if(event.key==='Home')next=0;else if(event.key==='End')next=all.length-1;else return;event.preventDefault();switchTab(all[next].dataset.tab);all[next].focus();};}
$('zoom-in').onclick=()=>zoom(.8);$('zoom-out').onclick=()=>zoom(1.25);$('fit-graph').onclick=()=>{camera={...fitted};updateCamera();};
$('graph').addEventListener('wheel',e=>{e.preventDefault();if(!camera)return;const box=$('graph').getBoundingClientRect();zoom(e.deltaY>0?1.12:.89,camera.x+(e.clientX-box.left)/box.width*camera.w,camera.y+(e.clientY-box.top)/box.height*camera.h);},{passive:false});
$('graph').onpointerdown=e=>{if(e.target.closest('.graph-node,.graph-edge'))return;drag={x:e.clientX,y:e.clientY,camera:{...camera}};$('graph').setPointerCapture(e.pointerId);};
$('graph').onpointermove=e=>{if(!drag)return;const box=$('graph').getBoundingClientRect();camera.x=drag.camera.x-(e.clientX-drag.x)*camera.w/box.width;camera.y=drag.camera.y-(e.clientY-drag.y)*camera.h/box.height;updateCamera();};
$('graph').onpointerup=()=>drag=null;$('graph').onpointercancel=()=>drag=null;
buildNodeOptions();
render();
