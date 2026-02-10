# IntelligentMaterialCerticate_web

## Avvio rapido (Docker)
```powershell
cd C:\Users\sireb\VScodeProjects\Certificate
docker compose up --build
```

Apri:
- Frontend: `http://localhost:5173/`
- API: `http://localhost:8000/api/health`

Per fermare:
```powershell
docker compose down
```

## Note principali
- Il frontend usa proxy verso l’API (`/api`) e in Docker punta a `http://api:8000`.
- I PDF vengono caricati e salvati sul backend.
- Per usare OpenAI, inserisci la **API key** nel campo “OpenAI Key” dell’interfaccia.

## Troubleshooting (problemi già visti)
- **La pagina non parte / vite: not found**
  - Ricostruisci le immagini: `docker compose down` → `docker compose up --build`.
  - Il container frontend avvia Vite direttamente con `node` (già configurato).
- **Proxy error /api/* (ECONNREFUSED)**
  - Aspetta che l’API sia “healthy”: il compose ora attende l’healthcheck.
  - Riavvia: `docker compose down` → `docker compose up --build`.
- **Rettangoli assenti / tokens 500**
  - Nel container serve Tesseract. È già installato in Dockerfile e `TESSERACT_CMD=/usr/bin/tesseract`.
  - Ricostruisci dopo modifiche: `docker compose up --build`.
- **PDF pages = 0 in Docker**
  - In Linux i path Windows non funzionano. Il backend usa `os.path.join` (già fixato).
- **Build molto lento**
  - È normale: i pacchetti ML (torch) sono enormi. Il primo build può richiedere molti minuti.

## Avvio locale (senza Docker)
Backend:
```powershell
cd C:\Users\sireb\VScodeProjects\Certificate\backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Frontend:
```powershell
cd C:\Users\sireb\VScodeProjects\Certificate\frontend
npm install
npm run dev
```

Apri:
- `http://localhost:5173/`

## Struttura
- `frontend/`: React + Vite
- `backend/`: FastAPI
- `docker-compose.yml`: avvio completo
