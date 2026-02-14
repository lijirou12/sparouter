import Head from 'next/head';

export default function Home() {
  return (
    <>
      <Head>
        <title>sparouter-nextjs debug router</title>
        <link rel="icon" href="/favicon.svg" />
      </Head>
      <main style={{ fontFamily: 'Arial, sans-serif', maxWidth: 900, margin: '40px auto', lineHeight: 1.6 }}>
        <h1>sparouter Next.js Debug Router</h1>
        <p>Service is running on Vercel (Node runtime API routes). If you still see 307/401 before this page renders, check Vercel Deployment Protection settings.</p>
        <h2>Available endpoints</h2>
        <ul>
          <li><code>POST /api/v1/chat/completions</code> (native Next.js API route)</li>
          <li><code>POST /v1/chat/completions</code> (rewrite to API route for OpenAI-compatible clients)</li>
          <li><code>GET /healthz</code> (rewrite to API health check)</li>
        </ul>
        <h2>Quick test</h2>
        <pre style={{ background: '#f6f8fa', padding: 12, borderRadius: 8, overflowX: 'auto' }}>{`curl -sS https://<your-vercel-domain>/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "anything",
    "messages": [{"role":"user","content":"你好"}]
  }'`}</pre>
      </main>
    </>
  );
}
