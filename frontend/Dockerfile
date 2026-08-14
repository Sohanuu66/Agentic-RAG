# =============================================================
#  Ask My Docs — Frontend Dockerfile
#  Node 18 build → nginx serve
# =============================================================

# ── Stage 1: build the React/Vite app ────────────────────────
FROM node:22-alpine AS builder

WORKDIR /app

# Install deps (layer-cached unless package files change)
COPY package.json package-lock.json ./
RUN npm ci --prefer-offline

# Copy source and build
COPY index.html ./
COPY vite.config.js ./
COPY src/ ./src/
COPY public/ ./public/

# Set VITE env for production API base
ENV VITE_API_BASE_URL=/

RUN npm run build


# ── Stage 2: serve with nginx ─────────────────────────────────
FROM nginx:1.27-alpine AS runtime

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# Custom nginx config: serve SPA, proxy /query /ingest /eval /health to backend
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -q --spider http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
