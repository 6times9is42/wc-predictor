const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const webDir = path.join(root, "web");
const publicDir = path.join(webDir, "public");
const distDir = path.join(root, "dist");

function copyDir(src, dest) {
  if (!fs.existsSync(src)) {
    return;
  }

  fs.mkdirSync(dest, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyDir(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

fs.rmSync(distDir, { recursive: true, force: true });
fs.mkdirSync(distDir, { recursive: true });

for (const fileName of ["index.html", "styles.css", "app.js"]) {
  fs.copyFileSync(path.join(webDir, fileName), path.join(distDir, fileName));
}

copyDir(publicDir, distDir);

console.log(`Built Vercel static app at ${path.relative(root, distDir)}`);
