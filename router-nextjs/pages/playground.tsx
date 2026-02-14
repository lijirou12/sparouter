import { useState } from 'react';
import { ConsoleLayout } from '../components/ConsoleLayout';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8080';

export default function PlaygroundPage() {
  const [payload, setPayload] = useState('{"model":"any","messages":[{"role":"user","content":"你好"}],"stream":false}');
  const [resp, setResp] = useState('');

  async function send() {
    const body = JSON.parse(payload);
    if (body.stream) {
      const r = await fetch(`${API_BASE}/v1/chat/completions`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const reader = r.body?.getReader();
      if (!reader) return;
      const dec = new TextDecoder();
      let buf = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        setResp(buf);
      }
    } else {
      const r = await fetch(`${API_BASE}/v1/chat/completions`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const t = await r.text();
      setResp(
        t +
          '\n\nheaders: ' +
          JSON.stringify(
            {
              group: r.headers.get('x-selected-group'),
              provider: r.headers.get('x-selected-provider'),
              model: r.headers.get('x-selected-model'),
              fallback: r.headers.get('x-fallback-count')
            },
            null,
            2
          )
      );
    }
  }

  return (
    <ConsoleLayout title='Playground (调试台)' description='直接请求 chat completions 接口，观察返回内容与路由头信息。'>
      <section className='panel stack'>
        <textarea rows={12} value={payload} onChange={(e) => setPayload(e.target.value)} />
        <button onClick={send}>发送请求</button>
        <pre>{resp}</pre>
      </section>
    </ConsoleLayout>
  );
}
