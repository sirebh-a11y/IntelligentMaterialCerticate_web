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
