import { useState } from 'react';
import { api } from '../lib/api';
const nav = [
  ['Providers', '/providers'],
  ['Routing', '/routing'],
  ['Classifier', '/classifier'],
  ['Generate', '/generate'],
  ['Playground', '/playground']
];


export default function GeneratePage() {
  const [files, setFiles] = useState<any>(null);
  const [token, setToken] = useState('');

  return <main style={{maxWidth:1100, margin:'24px auto', fontFamily:'Arial'}}>
    <h1>Generate / Export</h1>
    <p>{nav.map(([n,h]) => <a key={h} href={h} style={{marginRight: 12}}>{n}</a>)}</p>
    <button onClick={async()=>setFiles((await api('/api/generate', {method:'POST'})).files)}>生成配置</button>
    <h3>Apply</h3>
    <input value={token} onChange={e=>setToken(e.target.value)} placeholder='ADMIN_TOKEN' />
    <button onClick={async()=>{
      const res = await fetch((process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8080') + '/api/apply', {method:'POST', headers:{'X-Admin-Token':token}});
      alert(await res.text());
    }}>应用配置</button>
    <pre>{JSON.stringify(files, null, 2)}</pre>
  </main>
}
