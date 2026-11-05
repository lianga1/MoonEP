"""Static links plus execution of the report's real JS formulas, without browser."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import hashlib,json,subprocess,tempfile

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/eight_rank_estimate'

class Parts(HTMLParser):
    def __init__(self):super().__init__();self.scripts=[];self.current=None;self.links=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='script':self.current=[]
        if tag=='a' and a.get('href'):self.links.append(a['href'])
    def handle_endtag(self,tag):
        if tag=='script' and self.current is not None:self.scripts.append(''.join(self.current));self.current=None
    def handle_data(self,data):
        if self.current is not None:self.current.append(data)

p=OUT/'report.html';parts=Parts();parts.feed(p.read_text());assert len(parts.scripts)==2
missing=[]
for link in parts.links:
    u=urlsplit(link)
    if not u.scheme and u.path and not (p.parent/unquote(u.path)).exists():missing.append(link)
assert not missing,missing
estimate=json.loads((OUT/'estimate_attempt03.json').read_text())
program='''const vm=require('vm');
const input=JSON.parse(require('fs').readFileSync(process.argv[2],'utf8'));
const fields={data:{textContent:input.embedded},base:{value:'native_rt0'},schedule:{value:'serial'},
dispatch:{value:'156.32264378892432'},combine:{value:'182.01002257010302'},weight:{value:'330.3146385351925'},gradient:{value:'278.1711250126369'},planner:{value:'0'},refresh:{value:'1'},interactive:{textContent:''}};
const context={document:{getElementById:id=>fields[id],querySelectorAll:()=>[]}};
vm.createContext(context);vm.runInContext(input.script,context);
let count=0,maxerr=0;
const names={serial:'training_M1_serial_ms',ideal:'training_M1_ideal_all_weights_hidden_ms',shared:'training_M1_shared_endpoint_service_ms'};
const scenarios=[...input.estimate.scenarios,...input.estimate.independent_token_bandwidth_scenarios,...input.estimate.weight_refresh_sensitivity];
for(const sc of scenarios){context.rates=sc.effective_endpoint_GBs;context.refresh=sc.weight_refreshes_per_layer||1;
for(const method of ['native_rt0','native_rt2','sample256_g2_bundle'])for(const schedule of ['serial','ideal','shared']){
context.method=method;context.schedule=schedule;
const actual=vm.runInContext('data.traffic.reduce((s,l)=>s+calc(l.methods[method],rates,schedule,refresh),0)',context);
const expected=sc.aggregate[method][names[schedule]];const err=Math.abs(actual-expected);maxerr=Math.max(maxerr,err);count++;if(err>1e-9)throw Error(JSON.stringify({sc:sc.name,method,schedule,actual,expected}));}}
fields.planner.value='250';vm.runInContext('update()',context);const result=fields.interactive.textContent;
if(!result.includes('297.504'))throw Error('Planner per-layer units: '+result);
console.log(JSON.stringify({status:'passed',formula_comparisons:count,max_absolute_error_ms:maxerr,planner_250us_per_layer_adds_2ms_over_eight_layers:true,interactive_result:result}));
'''
with tempfile.TemporaryDirectory() as d:
    js=Path(d)/'audit.js';inp=Path(d)/'input.json';js.write_text(program)
    inp.write_text(json.dumps(dict(embedded=parts.scripts[0],script=parts.scripts[1],estimate=estimate)))
    result=json.loads(subprocess.check_output(['node',str(js),str(inp)],text=True))
result.update(missing_links=missing,checked_links=len(parts.links),independent_bandwidth_scenarios_visible=True,
              forward_backward_weight_lifetime_explicit=True,scope='source-level JS execution and static links; no browser visual render',
              report_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
(OUT/'report_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
