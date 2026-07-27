# Build the static bundle, then serve it with nginx.
#
# Coolify's "static" build pack serves the repository as-is and never runs a
# build, so a Vite app deployed that way ships raw .tsx. Owning the build here
# keeps the deployment reproducible and independent of the host's conventions.

FROM node:22-alpine AS build
WORKDIR /app
RUN corepack enable

# Dependencies first, so edits to source do not invalidate the install layer.
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
RUN pnpm install --no-frozen-lockfile

COPY . .
RUN pnpm build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
