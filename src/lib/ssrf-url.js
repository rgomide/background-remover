import dns from 'node:dns/promises';
import net from 'node:net';

const BLOCKED_HOSTNAMES = new Set([
  'localhost',
  '0.0.0.0',
  'metadata.google.internal',
  'metadata',
]);

function isBlockedIpv4(ip) {
  const parts = ip.split('.');
  if (parts.length !== 4) return false;
  const [a, b] = parts.map((x) => Number.parseInt(x, 10));
  if (!Number.isFinite(a) || !Number.isFinite(b)) return true;
  if (a === 0) return true;
  if (a === 10) return true;
  if (a === 127) return true;
  if (a === 169 && b === 254) return true;
  if (a === 172 && b >= 16 && b <= 31) return true;
  if (a === 192 && b === 168) return true;
  if (a === 100 && b >= 64 && b <= 127) return true;
  return false;
}

function isBlockedIpv6(ip) {
  const lower = ip.toLowerCase();
  if (lower === '::1') return true;
  if (lower.startsWith('fe80:')) return true;
  if (lower.startsWith('fc') || lower.startsWith('fd')) return true;
  if (lower.startsWith('::ffff:')) {
    const mapped = lower.slice(7);
    if (net.isIPv4(mapped)) return isBlockedIpv4(mapped);
  }
  return false;
}

function isBlockedIp(ip) {
  if (net.isIPv4(ip)) return isBlockedIpv4(ip);
  if (net.isIPv6(ip)) return isBlockedIpv6(ip);
  return false;
}

function isBlockedHostname(hostname) {
  const h = hostname.toLowerCase();
  if (BLOCKED_HOSTNAMES.has(h)) return true;
  if (h.endsWith('.local')) return true;
  if (h.endsWith('.internal')) return true;
  if (h === '127.0.0.1' || h === '::1') return true;
  if (h === '169.254.169.254') return true;
  return false;
}

function allowPrivateUrls() {
  return process.env.RMBG_ALLOW_PRIVATE_URLS === '1' || process.env.RMBG_ALLOW_PRIVATE_URLS === 'true';
}

export async function assertRemoteImageUrlAllowed(imageUrlString) {
  let parsed;
  try {
    parsed = new URL(imageUrlString);
  } catch {
    throw Object.assign(new Error('Invalid URL'), { status: 400 });
  }

  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw Object.assign(new Error('Only http and https URLs are allowed'), { status: 400 });
  }

  if (parsed.username || parsed.password) {
    throw Object.assign(new Error('URLs with credentials are not allowed'), { status: 400 });
  }

  const hostname = parsed.hostname;
  if (!hostname) {
    throw Object.assign(new Error('Missing hostname'), { status: 400 });
  }

  if (isBlockedHostname(hostname)) {
    throw Object.assign(new Error('Refusing to fetch from this host'), { status: 400 });
  }

  if (allowPrivateUrls()) {
    return;
  }

  if (net.isIPv4(hostname) || net.isIPv6(hostname)) {
    if (isBlockedIp(hostname)) {
      throw Object.assign(new Error('Refusing to fetch private or loopback addresses'), { status: 400 });
    }
    return;
  }

  let address;
  try {
    const lookup = await dns.lookup(hostname, { verbatim: true });
    address = lookup.address;
  } catch {
    throw Object.assign(new Error('Could not resolve image host'), { status: 400 });
  }

  if (isBlockedIp(address)) {
    throw Object.assign(new Error('URL resolves to a private or loopback address'), { status: 400 });
  }
}
