# Build stage
FROM oven/bun:1 AS builder
WORKDIR /app
# Use Chinese npm mirror for bun
ENV BUN_CONFIG_REGISTRY=https://registry.npmmirror.com
COPY package.json bun.lock* ./
RUN bun install --frozen-lockfile
COPY . .
RUN bun run build

# Serve stage
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
