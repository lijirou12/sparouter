import { useEffect, useState } from 'react';
import { api } from '../lib/api';
const nav = [
  ['Providers', '/providers'],
  ['Routing', '/routing'],
  ['Classifier', '/classifier'],
  ['Generate', '/generate'],
  ['Playground', '/playground']
];


export default function ProvidersPage() {
  const [providers, setProviders] = useState<any[]>([]);
  const [form, setForm] = useState<any>({ name: '', base_url: '', api_key: '', tags: {} });
  const [manual, setManual] = useState('');
  const [pid, setPid] = useState('');

  async function load() {
    const data = await api('/api/providers');
    setProviders(data.providers || []);
  }

  useEffect(() => { load(); }, []);

  return (
    <main style={{maxWidth: 1100, margin: '24px auto', fontFamily: 'Arial'}}>
      <h1>Providers</h1>
      <p>{nav.map(([n,h]) => <a key={h} href={h} style={{marginRight: 12}}>{n}</a>)}</p>
      <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:16}}>
        <section>
          <h3>新增/更新 Provider</h3>
          <input placeholder='name' value={form.name} onChange={e=>setForm({...form, name:e.target.value})} />
          <input placeholder='base_url (https://xxx/v1)' value={form.base_url} onChange={e=>setForm({...form, base_url:e.target.value})} />
          <input placeholder='api_key (仅更新时填写)' value={form.api_key} onChange={e=>setForm({...form, api_key:e.target.value})} />
          <button onClick={async()=>{ await api('/api/providers', {method:'POST', body:JSON.stringify(form)}); setForm({name:'',base_url:'',api_key:'',tags:{}}); load(); }}>保存</button>
          <hr/>
          <h3>手动导入模型（fallback）</h3>
          <input placeholder='provider id' value={pid} onChange={e=>setPid(e.target.value)} />
          <textarea rows={8} value={manual} onChange={e=>setManual(e.target.value)} placeholder='一行一个模型名或JSON数组' />
          <button onClick={async()=>{
            let manual_models:any[] = [];
            try { const arr = JSON.parse(manual); manual_models = Array.isArray(arr)?arr:[]; } catch { manual_models = manual.split('\n').map(x=>x.trim()).filter(Boolean); }
            await api(`/api/providers/${pid}/sync-models`, {method:'POST', body: JSON.stringify({ manual_models })});
            alert('已导入');
          }}>导入手动模型</button>
        </section>
        <section>
          <h3>供应商列表</h3>
          <pre>{JSON.stringify(providers, null, 2)}</pre>
          <h4>操作</h4>
          <input placeholder='provider id' value={pid} onChange={e=>setPid(e.target.value)} />
          <button onClick={async()=>{ await api(`/api/providers/${pid}/sync-models`, {method:'POST', body:'{}'}); alert('同步完成'); }}>一键拉取模型</button>
          <button onClick={async()=>{ const x = await api(`/api/providers/${pid}/health-check`, {method:'POST'}); alert(JSON.stringify(x)); load(); }}>健康检查</button>
          <button onClick={async()=>{ await api(`/api/providers/${pid}`, {method:'DELETE'}); load(); }}>删除</button>
        </section>
      </div>
    </main>
  );
}
