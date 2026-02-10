FROM node:20-alpine

WORKDIR /app

ENV NODE_ENV=development
ENV PATH=/app/node_modules/.bin:$PATH

COPY package.json package-lock.json* ./
RUN npm ci --include=dev

COPY . .

EXPOSE 5173

CMD ["node", "./node_modules/vite/bin/vite.js", "--host", "0.0.0.0", "--port", "5173"]
