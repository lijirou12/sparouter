import { useEffect, useState } from 'react';
import { api } from '../lib/api';
const nav = [
  ['Providers', '/providers'],
  ['Routing', '/routing'],
  ['Classifier', '/classifier'],
  ['Generate', '/generate'],
  ['Playground', '/playground']
];


export default function ClassifierPage() {
  const [cfg, setCfg] = useState<any>({ classifier_mode:'rules_only', classifier_provider:'', classifier_model:'', low_confidence_threshold:0.7 });
  const [providers, setProviders] = useState<any[]>([]);
  const [input, setInput] = useState('{"messages":[{"role":"user","content":"请写一个Python函数"}]}');
  const [result, setResult] = useState<any>(null);

  async function load() {
    setCfg(await api('/api/classifier/config'));
    setProviders((await api('/api/providers')).providers || []);
  }
  useEffect(()=>{load();}, []);

  return <main style={{maxWidth:1100, margin:'24px auto', fontFamily:'Arial'}}>
    <h1>Classifier / Router Model</h1>
    <p>{nav.map(([n,h]) => <a key={h} href={h} style={{marginRight: 12}}>{n}</a>)}</p>
    <h3>配置</h3>
    <select value={cfg.classifier_mode} onChange={e=>setCfg({...cfg, classifier_mode:e.target.value})}>
      <option value='rules_only'>rules_only</option>
      <option value='classifier_only'>classifier_only</option>
      <option value='hybrid'>hybrid</option>
    </select>
    <input value={cfg.classifier_provider} onChange={e=>setCfg({...cfg, classifier_provider:e.target.value})} placeholder='classifier_provider id' />
    <input value={cfg.classifier_model} onChange={e=>setCfg({...cfg, classifier_model:e.target.value})} placeholder='classifier_model' />
    <input type='number' value={cfg.low_confidence_threshold} onChange={e=>setCfg({...cfg, low_confidence_threshold:Number(e.target.value)})} />
    <button onClick={async()=>{ await api('/api/classifier/config', {method:'PUT', body:JSON.stringify(cfg)}); alert('saved');}}>保存配置</button>
    <h4>Provider 参考</h4><pre>{JSON.stringify(providers, null, 2)}</pre>

    <h3>分类器测试台</h3>
    <textarea rows={10} style={{width:'100%'}} value={input} onChange={e=>setInput(e.target.value)} />
    <button onClick={async()=>{ const body=JSON.parse(input); setResult(await api('/api/classifier/test', {method:'POST', body: JSON.stringify(body)})); }}>测试</button>
    <pre>{JSON.stringify(result, null, 2)}</pre>
  </main>
}
