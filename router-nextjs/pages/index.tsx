import Link from 'next/link';
const nav = [
  ['Providers', '/providers'],
  ['Routing', '/routing'],
  ['Classifier', '/classifier'],
  ['Generate', '/generate'],
  ['Playground', '/playground']
];


export default function Home() {
  return (
    <main style={{maxWidth: 980, margin: '32px auto', fontFamily: 'Arial'}}>
      <h1>AI Router 控制台</h1>
      <p>这是控制平面，不是聊天产品。核心是 Provider 导入、路由策略、分类器和配置生成。</p>
      <ul>
        {nav.map(([name, href]) => <li key={href}><Link href={href}>{name}</Link></li>)}
      </ul>
    </main>
  );
}
