import Link from 'next/link';
import { useRouter } from 'next/router';
import { ReactNode } from 'react';

const nav = [
  ['首页', '/'],
  ['Providers', '/providers'],
  ['Routing', '/routing'],
  ['Classifier', '/classifier'],
  ['Generate', '/generate'],
  ['Playground', '/playground']
];

export function ConsoleLayout({ title, description, children }: { title: string; description: string; children: ReactNode }) {
  const router = useRouter();

  return (
    <main className='page'>
      <section className='hero'>
        <h1>{title}</h1>
        <p>{description}</p>
        <nav className='nav'>
          {nav.map(([name, href]) => (
            <Link key={href} href={href} className={router.pathname === href ? 'active' : ''}>
              {name}
            </Link>
          ))}
        </nav>
      </section>
      {children}
    </main>
  );
}
