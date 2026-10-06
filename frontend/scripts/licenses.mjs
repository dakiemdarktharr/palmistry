import fs from 'node:fs';
import path from 'node:path';

const lock = JSON.parse(fs.readFileSync('package-lock.json', 'utf8'));
const sections = [];
for (const [location, pkg] of Object.entries(lock.packages).sort(([a],[b]) => a.localeCompare(b,'en'))) {
  if (!location || pkg.dev) continue;
  const name = location.replace(/^node_modules\//, '');
  const dir = path.resolve(location);
  if (!fs.existsSync(dir)) continue;
  const files = fs.readdirSync(dir).filter(name => /^(license|licence|copying)(\.(txt|md))?$/i.test(name));
  sections.push(`${name} ${pkg.version}\n${files.map(file => fs.readFileSync(path.join(dir,file),'utf8').replace(/\r\n/g,'\n')).join('\n')}`);
}
fs.writeFileSync('../static/labeler/THIRD_PARTY_LICENSES.txt', sections.join('\n\n'+'='.repeat(72)+'\n\n')+'\n');
