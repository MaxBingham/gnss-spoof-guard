import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const root = fileURLToPath(new URL(".", import.meta.url));
const project = fileURLToPath(new URL("..", import.meta.url));

export default defineConfig({
  root,
  server: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: true,
    fs: { allow: [project, root] },
  },
});
