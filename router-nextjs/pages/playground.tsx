import { useState } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8080';
const nav = [
  ['Providers', '/providers'],
  ['Routing', '/routing'],
  ['Classifier', '/classifier'],
  ['Generate', '/generate'],
  ['Playground', '/playground']
];

export default function PlaygroundPage() {
  const [payload, setPayload] = useState('{"model":"any","messages":[{"role":"user","content":"你好"}],"stream":false}');
  const [resp, setResp] = useState('');

  async function send() {
    const body = JSON.parse(payload);
    if (body.stream) {
      const r = await fetch(`${API_BASE}/v1/chat/completions`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
      const reader = r.body?.getReader();
      if (!reader) return;
      const dec = new TextDecoder();
      let buf = '';
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        buf += dec.decode(value, {stream:true});
        setResp(buf);
      }
    } else {
      const r = await fetch(`${API_BASE}/v1/chat/completions`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
      const t = await r.text();
      setResp(t + '\n\nheaders: ' + JSON.stringify({
        group: r.headers.get('x-selected-group'), provider: r.headers.get('x-selected-provider'), model: r.headers.get('x-selected-model'), fallback: r.headers.get('x-fallback-count')
      }, null, 2));
    }
  }

  return <main style={{maxWidth:1100, margin:'24px auto', fontFamily:'Arial'}}>
    <h1>Playground (调试台)</h1>
    <p>{nav.map(([n,h]) => <a key={h} href={h} style={{marginRight: 12}}>{n}</a>)}</p>
    <textarea rows={12} style={{width:'100%'}} value={payload} onChange={e=>setPayload(e.target.value)} />
    <button onClick={send}>发送</button>
    <pre style={{whiteSpace:'pre-wrap'}}>{resp}</pre>
  </main>
}
