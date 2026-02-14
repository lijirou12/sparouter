import { useEffect, useState } from 'react';
import { api } from '../lib/api';
const nav = [
  ['Providers', '/providers'],
  ['Routing', '/routing'],
  ['Classifier', '/classifier'],
  ['Generate', '/generate'],
  ['Playground', '/playground']
];


export default function RoutingPage() {
  const [groups, setGroups] = useState<any[]>([]);
  const [providers, setProviders] = useState<any[]>([]);
  const [newGroup, setNewGroup] = useState('');
  const [dep, setDep] = useState<any>({ group:'GENERAL', provider_id:'', model_name:'', weight:100, priority:100 });

  async function load() {
    const g = await api('/api/routing/groups');
    setGroups(g.groups || []);
    const p = await api('/api/providers');
    setProviders(p.providers || []);
  }

  useEffect(() => { load(); }, []);

  return <main style={{maxWidth:1100, margin:'24px auto', fontFamily:'Arial'}}>
    <h1>Routing Groups / Deployments</h1>
    <p>{nav.map(([n,h]) => <a key={h} href={h} style={{marginRight: 12}}>{n}</a>)}</p>
    <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:16}}>
      <section>
        <h3>Group 管理</h3>
        <input value={newGroup} onChange={e=>setNewGroup(e.target.value)} placeholder='new group name' />
        <button onClick={async()=>{await api('/api/routing/groups', {method:'POST', body:JSON.stringify({name:newGroup,enabled:true,tag_filter:''})}); setNewGroup(''); load();}}>新增组</button>
        <pre>{JSON.stringify(groups, null, 2)}</pre>
      </section>
      <section>
        <h3>新增 Deployment</h3>
        <input value={dep.group} onChange={e=>setDep({...dep, group:e.target.value})} placeholder='group' />
        <input value={dep.provider_id} onChange={e=>setDep({...dep, provider_id:e.target.value})} placeholder='provider id' />
        <input value={dep.model_name} onChange={e=>setDep({...dep, model_name:e.target.value})} placeholder='model name' />
        <input value={dep.weight} onChange={e=>setDep({...dep, weight:Number(e.target.value)})} placeholder='weight' />
        <input value={dep.priority} onChange={e=>setDep({...dep, priority:Number(e.target.value)})} placeholder='priority' />
        <button onClick={async()=>{
          await api(`/api/routing/groups/${dep.group}/deployments`, {method:'POST', body: JSON.stringify({
            provider_id: dep.provider_id, model_name: dep.model_name, weight: dep.weight, priority: dep.priority, enabled: true, fallback_on:['timeout','429','5xx']
          })});
          load();
        }}>保存 Deployment</button>
        <h4>Provider 参考</h4>
        <pre>{JSON.stringify(providers, null, 2)}</pre>
      </section>
    </div>
  </main>
}
