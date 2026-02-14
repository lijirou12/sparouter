/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/v1/chat/completions',
        destination: '/api/v1/chat/completions',
      },
      {
        source: '/healthz',
        destination: '/api/healthz',
      },
      {
        source: '/favicon.ico',
        destination: '/favicon.svg',
      },
      {
        source: '/favicon.png',
        destination: '/favicon.svg',
      },
    ];
  },
};

module.exports = nextConfig;
