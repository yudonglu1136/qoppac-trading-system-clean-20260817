import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..");

function monitorDataPlugin() {
  return {
    name: "ibkr-monitor-data",
    configureServer(server) {
      server.middlewares.use("/api/monitor", (_req, res) => {
        const result = spawnSync("python3", [resolve(repoRoot, "scripts/export_ibkr_monitor_data.py"), "--stdout"], {
          cwd: repoRoot,
          encoding: "utf8",
          timeout: 10000,
        });
        if (result.status !== 0) {
          res.statusCode = 500;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ error: "monitor_export_failed", detail: result.stderr || result.stdout }));
          return;
        }
        res.setHeader("Content-Type", "application/json; charset=utf-8");
        res.end(result.stdout);
      });
    },
  };
}

export default defineConfig({
  build: {
    outDir: "dist/client",
  },
  optimizeDeps: {
    include: ["react", "react-dom/client"],
  },
  server: {
    host: "127.0.0.1",
    warmup: {
      clientFiles: ["./src/main.jsx"],
    },
  },
  plugins: [react(), monitorDataPlugin()],
});
