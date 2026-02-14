import type { NextApiRequest, NextApiResponse } from 'next';

const upstreamBase = (process.env.UPSTREAM_BASE_URL || 'http://localhost:4000').replace(/\/$/, '');
const litellmKey = process.env.LITELLM_MASTER_KEY || '';

function flattenText(messages: any[]): string {
  const chunks: string[] = [];
  for (const msg of messages || []) {
    const content = msg?.content;
    if (typeof content === 'string') chunks.push(content);
    if (Array.isArray(content)) {
      for (const part of content) {
        if (part?.type === 'text') chunks.push(String(part.text || ''));
      }
    }
  }
  return chunks.join('\n').toLowerCase();
}

function hasImageOrFile(messages: any[]): boolean {
  for (const msg of messages || []) {
    if (!Array.isArray(msg?.content)) continue;
    for (const part of msg.content) {
      if (part?.type === 'image_url' || part?.type === 'file') return true;
    }
  }
  return false;
}

function detectIntent(body: any): 'GENERAL' | 'CODE' | 'IMAGE' | 'VIDEO' {
  const text = flattenText(body?.messages || []);
  if (body?.video_config) return 'VIDEO';
  if (/生成视频|视频超分|超清|分辨率|时长|宽高比|帧率/i.test(text)) return 'VIDEO';
  if (hasImageOrFile(body?.messages || [])) return 'IMAGE';
  if (/生成图片|画一张|出图|图片编辑|换风格|抠图|扩图/i.test(text)) return 'IMAGE';
  if (/```|写代码|实现|bug|报错|typeerror|exception|stack\s*trace|sql|正则|算法|单元测试|项目结构|readme|部署/i.test(text)) return 'CODE';
  return 'GENERAL';
}

export const config = {
  api: {
    bodyParser: true,
    externalResolver: true,
  },
};

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'method_not_allowed' });
    return;
  }

  const body = req.body || {};
  const group = detectIntent(body);
  body.model = group;

  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'x-route-group': group,
    'x-client-id': String(req.headers['x-client-id'] || 'nextjs-debug'),
    'x-request-mode': 'nextjs-proxy',
  };

  if (litellmKey) headers.authorization = `Bearer ${litellmKey}`;

  const upstreamResp = await fetch(`${upstreamBase}/v1/chat/completions`, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
  });

  res.setHeader('X-Selected-Group', group);

  const contentType = upstreamResp.headers.get('content-type') || 'application/json';
  res.setHeader('content-type', contentType);

  if (body.stream) {
    res.status(upstreamResp.status);
    const reader = upstreamResp.body?.getReader();
    if (!reader) {
      res.end();
      return;
    }
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) res.write(Buffer.from(value));
    }
    res.end();
    return;
  }

  const text = await upstreamResp.text();
  res.status(upstreamResp.status).send(text);
}
