import { useState } from 'react';
import { ConsoleLayout } from '../components/ConsoleLayout';
import { api } from '../lib/api';

export default function GeneratePage() {
  const [files, setFiles] = useState<any>(null);
  const [token, setToken] = useState('');

  return (
    <ConsoleLayout title='Generate / Export' description='根据控制台配置生成产物，并支持带管理员令牌的一键应用。'>
      <section className='panel stack'>
        <div className='row'>
          <button onClick={async () => setFiles((await api('/api/generate', { method: 'POST' })).files)}>生成配置</button>
        </div>

        <h3>Apply</h3>
        <input value={token} onChange={(e) => setToken(e.target.value)} placeholder='ADMIN_TOKEN' />
        <button onClick={async () => {
          const res = await fetch((process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8080') + '/api/apply', {
            method: 'POST',
            headers: { 'X-Admin-Token': token }
          });
          alert(await res.text());
        }}>应用配置</button>

        <pre>{JSON.stringify(files, null, 2)}</pre>
      </section>
    </ConsoleLayout>
  );
}
