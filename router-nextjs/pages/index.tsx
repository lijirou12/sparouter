import Link from 'next/link';
import { ConsoleLayout } from '../components/ConsoleLayout';

const cards = [
  ['Providers', '/providers', '管理供应商、API Key 与模型同步。'],
  ['Routing', '/routing', '维护分组、部署权重与优先级。'],
  ['Classifier', '/classifier', '配置分类器并做请求路由验证。'],
  ['Generate', '/generate', '一键生成配置并可直接应用。'],
  ['Playground', '/playground', '直接发送 chat completion 进行联调。']
];

export default function Home() {
  return (
    <ConsoleLayout title='AI Router 控制台' description='这是控制平面：集中管理 Provider、路由策略、分类器与配置生成。'>
      <section className='grid-2'>
        {cards.map(([name, href, desc]) => (
          <Link key={href} href={href} className='panel'>
            <h3>{name}</h3>
            <p style={{ color: 'var(--muted)', marginBottom: 0 }}>{desc}</p>
          </Link>
        ))}
      </section>
    </ConsoleLayout>
  );
}
