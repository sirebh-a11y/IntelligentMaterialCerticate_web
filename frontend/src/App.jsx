import { useEffect, useMemo, useRef, useState } from "react";

export default function App() {
  const [status, setStatus] = useState("checking...");
  const [statusMsg, setStatusMsg] = useState("Pronto");
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadTotalProgress, setUploadTotalProgress] = useState(0);
  const [pdfSearch, setPdfSearch] = useState("");
  const [pdfs, setPdfs] = useState([]);
  const [selectedPdf, setSelectedPdf] = useState("");
  const [pageCount, setPageCount] = useState(0);
  const [pageIdx, setPageIdx] = useState(0);
  const [pagesLoading, setPagesLoading] = useState(false);
  const [tokens, setTokens] = useState([]);
  const [tokenImageSize, setTokenImageSize] = useState({ width: 0, height: 0 });
  const [imgMeta, setImgMeta] = useState({
    naturalW: 0,
    naturalH: 0,
    displayW: 0,
    displayH: 0,
  });
  const [imageLoaded, setImageLoaded] = useState(false);
  const [tokensLoading, setTokensLoading] = useState(false);
  const [selection, setSelection] = useState(null);
  const [selectedTokenIds, setSelectedTokenIds] = useState([]);
  const [excludedTokenIds, setExcludedTokenIds] = useState([]);
  const [usedTokenIds, setUsedTokenIds] = useState([]);
  const [activeField, setActiveField] = useState("azienda");
  const [fields, setFields] = useState({
    azienda: "",
    data_certificato: "",
    materiale: "",
    trattamento_termico: "",
    composizione_chimica: "",
    proprieta_meccaniche: "",
  });
  const [fieldObjects, setFieldObjects] = useState({});
  const [tableHints, setTableHints] = useState({});
  const [openaiPrimaryFields, setOpenaiPrimaryFields] = useState(null);
  const [lastPrompt, setLastPrompt] = useState("");
  const [lastPromptTitle, setLastPromptTitle] = useState("");
  const [promptLog, setPromptLog] = useState([]);
  const [statusLog, setStatusLog] = useState([]);
  const [chemRows, setChemRows] = useState([]);
  const [chemEl, setChemEl] = useState("");
  const [chemVal, setChemVal] = useState("");
  const [mode, setMode] = useState("select"); // select | delete | catch | pick_el | pick_val
  const [fieldMeta, setFieldMeta] = useState({});
  const [validatedTokenIds, setValidatedTokenIds] = useState([]);
  const [zoom, setZoom] = useState(1);
  const [aiBusy, setAiBusy] = useState({
    extract: false,
    full: false,
    pdf: false,
    refine: false,
    trained: false,
  });
  const [deletingPdf, setDeletingPdf] = useState("");
  const [apiKey, setApiKey] = useState("");
  const imgRef = useRef(null);
  const canvasRef = useRef(null);
  const tokensCacheRef = useRef(new Map());
  const prefetchingRef = useRef(new Set());
  const imageCacheRef = useRef(new Set());
  const lastUploadRef = useRef({ key: "", ts: 0 });
  const deletingRef = useRef(false);
  const isDraggingRef = useRef(false);
  const dragStartRef = useRef({ x: 0, y: 0 });
  const dragMovedRef = useRef(false);
  const currentAiAction = useMemo(() => {
    if (aiBusy.extract) return "AI Extract";
    if (aiBusy.full) return "OpenAI Full";
    if (aiBusy.pdf) return "OpenAI PDF";
    if (aiBusy.refine) return "OpenAI Refine";
    if (aiBusy.trained) return "AI Trained";
    return "";
  }, [aiBusy]);
  const debugInfo = useMemo(() => {
    const imgEl = imgRef.current;
    const nw = imgEl?.naturalWidth || 0;
    const nh = imgEl?.naturalHeight || 0;
    return `img:${nw}x${nh} tokens:${tokenImageSize.width}x${tokenImageSize.height} zoom:${Math.round(
      zoom * 100
    )}%`;
  }, [tokenImageSize.width, tokenImageSize.height, zoom]);

  useEffect(() => {
    fetch("/api/health")
      .then((res) => res.json())
      .then((data) => setStatus(data.status || "unknown"))
      .catch(() => setStatus("error"));
  }, []);

  useEffect(() => {
    fetch("/api/pdfs")
      .then((res) => res.json())
      .then((data) => {
        const list = data.pdfs || [];
        setPdfs(list);
        if (list.length > 0) {
          setSelectedPdf(list[0]);
        }
      })
      .catch(() => setPdfs([]));
  }, []);

  useEffect(() => {
    if (pdfs.length > 0 && (!selectedPdf || !pdfs.includes(selectedPdf))) {
      setSelectedPdf(pdfs[0]);
    }
  }, [pdfs, selectedPdf]);

  const refreshPdfs = () => {
    fetch("/api/pdfs")
      .then((res) => res.json())
      .then((data) => {
        const list = data.pdfs || [];
        setPdfs(list);
        if (list.length > 0 && !selectedPdf) {
          setSelectedPdf(list[0]);
        }
      })
      .catch(() => setPdfs([]));
  };

  const uploadFile = (file) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setStatusMsg("Solo PDF");
      return;
    }
    setUploading(true);
    setStatusMsg("Upload in corso...");
    setUploadProgress(0);
    const form = new FormData();
    form.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/pdfs/upload", true);
    xhr.upload.onprogress = (evt) => {
      if (!evt.lengthComputable) return;
      const pct = Math.round((evt.loaded / evt.total) * 100);
      setUploadProgress(pct);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        setStatusMsg("Upload completato");
        try {
          const resp = JSON.parse(xhr.responseText || "{}");
          if (resp.filename) {
            selectPdf(resp.filename);
          }
        } catch {}
        refreshPdfs();
      } else {
        setStatusMsg("Errore upload");
      }
      setUploading(false);
    };
    xhr.onerror = () => {
      setStatusMsg("Errore upload");
      setUploading(false);
    };
    xhr.send(form);
  };

  const uploadFiles = async (files) => {
    if (!files || files.length === 0) return;
    const rawList = Array.from(files);
    const seen = new Set();
    const list = [];
    rawList.forEach((f) => {
      const key = `${f.name}|${f.size}|${f.lastModified}`;
      if (seen.has(key)) return;
      seen.add(key);
      list.push(f);
    });
    if (list.length === 0) return;
    const batchKey = list
      .map((f) => `${f.name}|${f.size}|${f.lastModified}`)
      .sort()
      .join("||");
    const now = Date.now();
    if (lastUploadRef.current.key === batchKey && now - lastUploadRef.current.ts < 1500) {
      return;
    }
    lastUploadRef.current = { key: batchKey, ts: now };
    setUploading(true);
    setUploadProgress(0);
    setUploadTotalProgress(0);
    setStatusMsg(`Upload 0/${list.length}...`);

    const progressMap = new Map();
    const updateTotal = () => {
      let sum = 0;
      progressMap.forEach((v) => {
        sum += v;
      });
      const avg = Math.round(sum / Math.max(1, progressMap.size));
      setUploadTotalProgress(avg);
    };

    await Promise.all(
      list.map((f, idx) => {
        return new Promise((resolve) => {
          if (!f.name.toLowerCase().endsWith(".pdf")) {
            resolve();
            return;
          }
          const form = new FormData();
          form.append("file", f);
        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/api/pdfs/upload", true);
          xhr.upload.onprogress = (evt) => {
            if (!evt.lengthComputable) return;
            const pct = Math.round((evt.loaded / evt.total) * 100);
            progressMap.set(idx, pct);
            setUploadProgress(pct);
            updateTotal();
            setStatusMsg(`Upload ${progressMap.size}/${list.length}...`);
          };
        xhr.onload = () => {
          progressMap.set(idx, 100);
          updateTotal();
          if (!(xhr.status >= 200 && xhr.status < 300)) {
            setStatusMsg("Errore upload");
          } else {
            try {
              const resp = JSON.parse(xhr.responseText || "{}");
              if (resp.filename) {
                selectPdf(resp.filename);
              }
            } catch {}
          }
          resolve();
        };
          xhr.onerror = () => {
            progressMap.set(idx, 100);
            updateTotal();
            setStatusMsg("Errore upload");
            resolve();
          };
          xhr.send(form);
        });
      })
    );

    setUploading(false);
    setStatusMsg("Upload completato");
    refreshPdfs();
  };

  const onDrop = (evt) => {
    evt.preventDefault();
    evt.stopPropagation();
    const files = evt.dataTransfer.files;
    uploadFiles(files);
  };

  useEffect(() => {
    const onWinDragOver = (e) => e.preventDefault();
    const onWinDrop = (e) => {
      if (e.defaultPrevented) return;
      const target = e.target;
      if (target && target.closest && target.closest(".dropzone")) {
        // Drop handled by the dropzone itself.
        return;
      }
      e.preventDefault();
      const files = e.dataTransfer && e.dataTransfer.files;
      if (files && files.length) {
        uploadFiles(files);
      }
    };
    window.addEventListener("dragover", onWinDragOver);
    window.addEventListener("drop", onWinDrop);
    return () => {
      window.removeEventListener("dragover", onWinDragOver);
      window.removeEventListener("drop", onWinDrop);
    };
  }, []);

  useEffect(() => {
    const onWheel = (e) => {
      if (e.ctrlKey) {
        // Prevent browser zoom so PDF zoom doesn't resize the whole UI.
        e.preventDefault();
      }
    };
    window.addEventListener("wheel", onWheel, { passive: false });
    return () => window.removeEventListener("wheel", onWheel);
  }, []);

  const selectPdf = (name) => {
    if (!name) return;
    setSelectedPdf(name);
    setPageIdx(0);
    setPageCount(0);
    setTokens([]);
  };

  const deletePdf = (name) => {
    if (!name) return;
    if (deletingPdf) return;
    if (!window.confirm(`Eliminare ${name}?`)) return;
    deletingRef.current = true;
    setStatusMsg("Eliminazione...");
    setDeletingPdf(name);
    setPdfs((list) => list.filter((p) => p !== name));
    if (selectedPdf === name) {
      setSelectedPdf("");
      setPageIdx(0);
      setPageCount(0);
      setTokens([]);
      setImageLoaded(false);
      setTokensLoading(false);
      tokensCacheRef.current = new Map();
      prefetchingRef.current = new Set();
      imageCacheRef.current = new Set();
    }
    fetch(`/api/pdfs/${encodeURIComponent(name)}`, { method: "DELETE" })
      .then((res) => res.json())
      .then(() => {
        setStatusMsg("Eliminato");
        refreshPdfs();
      })
      .catch(() => {
        setStatusMsg("Errore eliminazione");
        refreshPdfs();
      })
      .finally(() => {
        setDeletingPdf("");
        deletingRef.current = false;
      });
  };

  useEffect(() => {
    if (!selectedPdf) return;
    if (deletingRef.current || deletingPdf) return;
    setStatusMsg("Caricamento pagine...");
    setPagesLoading(true);
    setImageLoaded(false);
    setTokenImageSize({ width: 0, height: 0 });
    tokensCacheRef.current = new Map();
    prefetchingRef.current = new Set();
    imageCacheRef.current = new Set();
    setFieldObjects({});
    setTableHints({});
    setOpenaiPrimaryFields(null);
    setLastPrompt("");
    setLastPromptTitle("");
    setPromptLog([]);
    setStatusLog([]);
    setUsedTokenIds([]);
    setValidatedTokenIds([]);
    setSelectedTokenIds([]);
    setExcludedTokenIds([]);
    setChemRows([]);
    fetch(`/api/pdfs/${encodeURIComponent(selectedPdf)}/pages`)
      .then((res) => res.json())
      .then((data) => {
        const count = data.pages || 0;
        setPageCount(count);
        setPageIdx(0);
        setPagesLoading(false);
        setStatusMsg("Pagine caricate");
      })
      .catch(() => {
        setPageCount(0);
        setPagesLoading(false);
        setStatusMsg("Errore caricamento pagine");
      });
  }, [selectedPdf]);

  useEffect(() => {
    if (!selectedPdf || pageCount <= 0) return;
    if (deletingRef.current || deletingPdf) return;
    const currentPdf = selectedPdf;
    const cache = tokensCacheRef.current;
    const prefetching = prefetchingRef.current;
    const imgCache = imageCacheRef.current;
    const run = async () => {
      for (let i = 0; i < pageCount; i += 1) {
        if (selectedPdf !== currentPdf) break;
        const key = `${currentPdf}::${i}`;
        if (!imgCache.has(key)) {
          const img = new Image();
          img.src = `/api/pdfs/${encodeURIComponent(currentPdf)}/pages/${i}/image`;
          imgCache.add(key);
        }
        if (cache.has(key) || prefetching.has(key)) continue;
        prefetching.add(key);
        try {
          const res = await fetch(`/api/pdfs/${encodeURIComponent(currentPdf)}/pages/${i}/tokens`);
          if (!res.ok) continue;
          const data = await res.json();
          cache.set(key, data);
        } catch {
          // ignore prefetch errors
        } finally {
          prefetching.delete(key);
        }
      }
    };
    run();
  }, [selectedPdf, pageCount]);

  useEffect(() => {
    setImageLoaded(false);
  }, [selectedPdf, pageIdx]);

  useEffect(() => {
    if (pageCount > 0 && pageIdx >= pageCount) {
      setPageIdx(0);
    }
  }, [pageCount, pageIdx]);

  useEffect(() => {
    if (!selectedPdf) return;
    if (pagesLoading || pageCount === 0 || pageIdx >= pageCount) return;
    if (deletingRef.current || deletingPdf) return;
    setStatusMsg("Caricamento token...");
    const currentPdf = selectedPdf;
    const currentPage = pageIdx;
    setTokens([]);
    const cacheKey = `${selectedPdf}::${pageIdx}`;
    const cached = tokensCacheRef.current.get(cacheKey);
    if (cached) {
      setTokens(cached.tokens || cached);
      if (cached.image_size) {
        setTokenImageSize(cached.image_size);
      }
      setTokensLoading(false);
      setStatusMsg("Pronto");
      return;
    }
    setTokensLoading(true);
    fetch(`/api/pdfs/${encodeURIComponent(selectedPdf)}/pages/${pageIdx}/tokens`)
      .then((res) => res.json())
      .then((data) => {
        if (selectedPdf !== currentPdf || pageIdx !== currentPage) return;
        const nextTokens = data.tokens || [];
        const payload = { tokens: nextTokens, image_size: data.image_size || null };
        tokensCacheRef.current.set(cacheKey, payload);
        setTokens(nextTokens);
        if (data.image_size) {
          setTokenImageSize(data.image_size);
        }
        setTokensLoading(false);
        setStatusMsg("Pronto");
      })
      .catch(() => {
        if (selectedPdf !== currentPdf || pageIdx !== currentPage) return;
        setTokens([]);
        setTokensLoading(false);
        setStatusMsg("Errore caricamento token");
      });
  }, [selectedPdf, pageIdx, pageCount, pagesLoading]);

  const imageUrl = useMemo(() => {
    if (!selectedPdf) return "";
    return `/api/pdfs/${encodeURIComponent(selectedPdf)}/pages/${pageIdx}/image`;
  }, [selectedPdf, pageIdx]);

  const updateImgMeta = () => {
    const imgEl = imgRef.current;
    if (!imgEl) return;
    const naturalW = imgEl.naturalWidth || tokenImageSize.width || 0;
    const naturalH = imgEl.naturalHeight || tokenImageSize.height || 0;
    setImgMeta({
      naturalW,
      naturalH,
      displayW: naturalW * zoom,
      displayH: naturalH * zoom,
    });
  };

  useEffect(() => {
    updateImgMeta();
    const handler = () => updateImgMeta();
    window.addEventListener("resize", handler);
    return () => window.removeEventListener("resize", handler);
  }, []);
  useEffect(() => {
    updateImgMeta();
  }, [zoom]);
  useEffect(() => {
    const imgEl = imgRef.current;
    if (!imgEl || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => updateImgMeta());
    ro.observe(imgEl);
    return () => ro.disconnect();
  }, [imageUrl]);
  useEffect(() => {
    const raf = requestAnimationFrame(updateImgMeta);
    return () => cancelAnimationFrame(raf);
  }, [imageUrl, zoom]);
  useEffect(() => {
    const imgEl = imgRef.current;
    if (imgEl && imgEl.complete && imgEl.naturalWidth > 0) {
      setImageLoaded(true);
    }
  }, [imageUrl]);

  const aiTokenIds = useMemo(() => {
    const ids = new Set();
    Object.values(fieldObjects || {}).forEach((v) => {
      if (!v) return;
      if (v.tokens_by_page) {
        const arr = v.tokens_by_page[pageIdx] || [];
        (arr || []).forEach((id) => ids.add(id));
      } else if (v.tokens) {
        (v.tokens || []).forEach((id) => ids.add(id));
      }
    });
    return Array.from(ids);
  }, [fieldObjects, pageIdx]);

  const greenTokenIds = useMemo(() => {
    const ids = new Set();
    aiTokenIds.forEach((id) => ids.add(id));
    usedTokenIds.forEach((id) => ids.add(id));
    validatedTokenIds.forEach((id) => ids.add(id));
    return Array.from(ids);
  }, [aiTokenIds, usedTokenIds, validatedTokenIds]);

  const drawOverlay = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const imgEl = imgRef.current;
    if (imgEl && !imgEl.complete) return;
    const naturalW = tokenImageSize.width || imgMeta.naturalW;
    const naturalH = tokenImageSize.height || imgMeta.naturalH;
    const displayW = naturalW * zoom;
    const displayH = naturalH * zoom;
    if (!naturalW || !naturalH || !displayW || !displayH) return;
    canvas.width = displayW;
    canvas.height = displayH;
    canvas.style.width = `${displayW}px`;
    canvas.style.height = `${displayH}px`;
    ctx.clearRect(0, 0, displayW, displayH);

    const sx = zoom;
    const sy = zoom;

    // draw tokens (default red)
    ctx.strokeStyle = "rgba(220, 38, 38, 0.7)";
    ctx.lineWidth = 1;
    tokens.forEach((t) => {
      const [x0, y0, x1, y1] = t.bbox;
      const rx = x0 * sx;
      const ry = y0 * sy;
      const rw = (x1 - x0) * sx;
      const rh = (y1 - y0) * sy;
      ctx.strokeRect(rx, ry, rw, rh);
    });

    // highlight selected tokens (blue = user selection)
    if (selectedTokenIds.length > 0) {
      ctx.fillStyle = "rgba(59, 130, 246, 0.35)";
      tokens.forEach((t) => {
        if (!selectedTokenIds.includes(t.id)) return;
        const [x0, y0, x1, y1] = t.bbox;
        const rx = x0 * sx;
        const ry = y0 * sy;
        const rw = (x1 - x0) * sx;
        const rh = (y1 - y0) * sy;
        ctx.fillRect(rx, ry, rw, rh);
      });
    }

    if (greenTokenIds.length > 0) {
      ctx.fillStyle = "rgba(16, 185, 129, 0.35)";
      tokens.forEach((t) => {
        if (!greenTokenIds.includes(t.id)) return;
        const [x0, y0, x1, y1] = t.bbox;
        const rx = x0 * sx;
        const ry = y0 * sy;
        const rw = (x1 - x0) * sx;
        const rh = (y1 - y0) * sy;
        ctx.fillRect(rx, ry, rw, rh);
      });
    }

    if (excludedTokenIds.length > 0) {
      ctx.fillStyle = "rgba(220, 20, 60, 0.25)";
      tokens.forEach((t) => {
        if (!excludedTokenIds.includes(t.id)) return;
        const [x0, y0, x1, y1] = t.bbox;
        const rx = x0 * sx;
        const ry = y0 * sy;
        const rw = (x1 - x0) * sx;
        const rh = (y1 - y0) * sy;
        ctx.fillRect(rx, ry, rw, rh);
      });
    }

    // draw selection box
    if (selection) {
      const { x0, y0, x1, y1 } = selection;
      ctx.strokeStyle = mode === "catch" || mode === "pick_el" || mode === "pick_val"
        ? "rgba(0, 120, 255, 0.8)"
        : "rgba(255, 0, 0, 0.7)";
      ctx.lineWidth = 2;
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    }
  };

  useEffect(() => {
    drawOverlay();
  }, [tokens, imgMeta, selection, selectedTokenIds, greenTokenIds, zoom]);

  const toCanvasPoint = (evt) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = evt.clientX - rect.left;
    const y = evt.clientY - rect.top;
    return { x, y };
  };

  const onMouseDown = (evt) => {
    if (!canvasRef.current) return;
    isDraggingRef.current = true;
    dragMovedRef.current = false;
    const { x, y } = toCanvasPoint(evt);
    dragStartRef.current = { x, y };
    setSelection({ x0: x, y0: y, x1: x, y1: y });
  };

  const onMouseMove = (evt) => {
    if (!isDraggingRef.current) return;
    const { x, y } = toCanvasPoint(evt);
    const { x: sx, y: sy } = dragStartRef.current;
    const x0 = Math.min(sx, x);
    const y0 = Math.min(sy, y);
    const x1 = Math.max(sx, x);
    const y1 = Math.max(sy, y);
    dragMovedRef.current = Math.abs(x - sx) > 3 || Math.abs(y - sy) > 3;
    setSelection({ x0, y0, x1, y1 });
  };

  const onMouseUp = (evt) => {
    isDraggingRef.current = false;
    if (!dragMovedRef.current) {
      // click: toggle exclude in delete mode
      const naturalW = tokenImageSize.width || imgMeta.naturalW;
      const naturalH = tokenImageSize.height || imgMeta.naturalH;
      const displayW = naturalW * zoom;
      const displayH = naturalH * zoom;
      if (!naturalW || !naturalH || !displayW || !displayH) return;
      const { x, y } = toCanvasPoint(evt);
      const sx = naturalW / displayW;
      const sy = naturalH / displayH;
      const px = x * sx;
      const py = y * sy;
      const hit = tokens.find((t) => {
        const [x0, y0, x1, y1] = t.bbox;
        return px >= x0 && px <= x1 && py >= y0 && py <= y1;
      });
      if (hit && mode === "delete") {
        setExcludedTokenIds((prev) =>
          prev.includes(hit.id) ? prev.filter((id) => id !== hit.id) : [...prev, hit.id]
        );
      }
      return;
    }
    if (!selection || tokens.length === 0) return;
    const naturalW = tokenImageSize.width || imgMeta.naturalW;
    const naturalH = tokenImageSize.height || imgMeta.naturalH;
    const displayW = naturalW * zoom;
    const displayH = naturalH * zoom;
    if (!naturalW || !naturalH || !displayW || !displayH) return;
    const sx = naturalW / displayW;
    const sy = naturalH / displayH;
    const sel = {
      x0: selection.x0 * sx,
      y0: selection.y0 * sy,
      x1: selection.x1 * sx,
      y1: selection.y1 * sy,
    };
    const picked = tokens
      .filter((t) => {
        const [x0, y0, x1, y1] = t.bbox;
        const cx = (x0 + x1) / 2;
        const cy = (y0 + y1) / 2;
        return cx >= sel.x0 && cx <= sel.x1 && cy >= sel.y0 && cy <= sel.y1;
      })
      .map((t) => t.id);
    setSelectedTokenIds(picked);
    if (mode === "catch") {
      const text = tokens
        .filter((t) => picked.includes(t.id))
        .sort((a, b) => (a.bbox[1] !== b.bbox[1] ? a.bbox[1] - b.bbox[1] : a.bbox[0] - b.bbox[0]))
        .map((t) => t.text)
        .join(" ");
      setFields((f) => ({ ...f, [activeField]: text }));
      setFieldMeta((m) => ({
        ...m,
        [activeField]: {
          page: pageIdx,
          bbox: [sel.x0, sel.y0, sel.x1, sel.y1],
          tokens: picked,
        },
      }));
      setValidatedTokenIds((prev) => Array.from(new Set([...prev, ...picked])));
      setSelectedTokenIds([]);
    }
    if (mode === "pick_el") {
      const text = tokens
        .filter((t) => picked.includes(t.id))
        .sort((a, b) => (a.bbox[1] !== b.bbox[1] ? a.bbox[1] - b.bbox[1] : a.bbox[0] - b.bbox[0]))
        .map((t) => t.text)
        .join(" ");
      setChemEl(text);
    }
    if (mode === "pick_val") {
      const text = tokens
        .filter((t) => picked.includes(t.id))
        .sort((a, b) => (a.bbox[1] !== b.bbox[1] ? a.bbox[1] - b.bbox[1] : a.bbox[0] - b.bbox[0]))
        .map((t) => t.text)
        .join(" ");
      setChemVal(text);
    }
  };

  const sortedSelectedTokens = useMemo(() => {
    const selected = tokens.filter((t) => selectedTokenIds.includes(t.id));
    return selected.sort((a, b) => {
      if (a.bbox[1] !== b.bbox[1]) return a.bbox[1] - b.bbox[1];
      return a.bbox[0] - b.bbox[0];
    });
  }, [tokens, selectedTokenIds]);

  const applySelectionToField = () => {
    const text = sortedSelectedTokens.map((t) => t.text).join(" ");
    setFields((f) => ({ ...f, [activeField]: text }));
    setValidatedTokenIds((prev) => Array.from(new Set([...prev, ...selectedTokenIds])));
    setSelectedTokenIds([]);
  };

  const clearField = () => {
    setFields((f) => ({ ...f, [activeField]: "" }));
  };

  const runAI = () => {
    if (!selectedPdf || aiBusy.extract) return;
    const currentPdf = selectedPdf;
    setAiBusy((b) => ({ ...b, extract: true }));
    setStatusMsg("Elaborazione AI...");
    fetch("/api/ai/extract", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(apiKey ? { "X-OpenAI-Key": apiKey } : {}),
      },
      body: JSON.stringify({ pdf_name: selectedPdf }),
    })
      .then((res) => res.json())
      .then((data) => {
        if (selectedPdf !== currentPdf) return;
        const r = data.result || {};
        setFields((f) => ({
          ...f,
          azienda: r.azienda || "",
          data_certificato: r.data_certificato || "",
          materiale: r.materiale || "",
          trattamento_termico: r.trattamento_termico || "",
          composizione_chimica: r.composizione_chimica
            ? JSON.stringify(r.composizione_chimica)
            : "",
          proprieta_meccaniche: r.proprieta_meccaniche || "",
        }));
        setStatusMsg("AI pronta");
      })
      .catch(() => setStatusMsg("Errore AI"))
      .finally(() => setAiBusy((b) => ({ ...b, extract: false })));
  };

  const applyFieldsFromResult = (resultFields) => {
    if (!resultFields) return;
    setFieldObjects(resultFields);
    const next = { ...fields };
    Object.keys(next).forEach((k) => {
      const v = resultFields[k];
      next[k] = v && v.text ? v.text : "";
    });
    setFields(next);
  };

  const runAIFull = () => {
    if (!selectedPdf || aiBusy.full) return;
    const currentPdf = selectedPdf;
    setAiBusy((b) => ({ ...b, full: true }));
    setStatusMsg("OpenAI Full in corso...");
    fetch("/api/ai/full", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(apiKey ? { "X-OpenAI-Key": apiKey } : {}),
      },
      body: JSON.stringify({ pdf_name: selectedPdf }),
    })
      .then(async (res) => {
        if (!res.ok) {
          const txt = await res.text();
          throw new Error(txt || "AI Full error");
        }
        return res.json();
      })
      .then((data) => {
        if (selectedPdf !== currentPdf) return;
        applyFieldsFromResult(data.fields);
        setTableHints(data.table_hints || {});
        setOpenaiPrimaryFields(data.fields || null);
        setLastPrompt(data.prompt || "");
        setLastPromptTitle("OpenAI Full Prompt");
        setPromptLog((log) => [
          { title: "OpenAI Full Prompt", text: data.prompt || "", ts: new Date().toISOString() },
          ...log,
        ]);
        if (data.logs) {
          setStatusLog((log) => [...data.logs, ...log]);
        }
        setStatusMsg("OpenAI Full pronto");
      })
      .catch((e) => setStatusMsg(`Errore OpenAI Full: ${e.message}`))
      .finally(() => setAiBusy((b) => ({ ...b, full: false })));
  };

  const runAIPdf = () => {
    if (!selectedPdf || aiBusy.pdf) return;
    const currentPdf = selectedPdf;
    setAiBusy((b) => ({ ...b, pdf: true }));
    setStatusMsg("OpenAI PDF in corso...");
    fetch("/api/ai/pdf", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(apiKey ? { "X-OpenAI-Key": apiKey } : {}),
      },
      body: JSON.stringify({ pdf_name: selectedPdf }),
    })
      .then(async (res) => {
        if (!res.ok) {
          const txt = await res.text();
          throw new Error(txt || "AI PDF error");
        }
        return res.json();
      })
      .then((data) => {
        if (selectedPdf !== currentPdf) return;
        applyFieldsFromResult(data.fields);
        setLastPrompt(data.prompt || "");
        setLastPromptTitle("OpenAI PDF Prompt");
        setPromptLog((log) => [
          { title: "OpenAI PDF Prompt", text: data.prompt || "", ts: new Date().toISOString() },
          ...log,
        ]);
        if (data.logs) {
          setStatusLog((log) => [...data.logs, ...log]);
        }
        setStatusMsg("OpenAI PDF pronto");
      })
      .catch((e) => setStatusMsg(`Errore OpenAI PDF: ${e.message}`))
      .finally(() => setAiBusy((b) => ({ ...b, pdf: false })));
  };

  const runAIRefine = () => {
    if (!selectedPdf || aiBusy.refine) return;
    const currentPdf = selectedPdf;
    setAiBusy((b) => ({ ...b, refine: true }));
    setStatusMsg("OpenAI Refine in corso...");
    fetch("/api/ai/refine", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(apiKey ? { "X-OpenAI-Key": apiKey } : {}),
      },
      body: JSON.stringify({
        pdf_name: selectedPdf,
        fields: fieldObjects,
        table_hints: tableHints,
      }),
    })
      .then(async (res) => {
        if (!res.ok) {
          const txt = await res.text();
          throw new Error(txt || "AI Refine error");
        }
        return res.json();
      })
      .then((data) => {
        if (selectedPdf !== currentPdf) return;
        applyFieldsFromResult(data.fields);
        setLastPrompt(data.prompt || "");
        setLastPromptTitle("OpenAI Refine Prompt");
        setPromptLog((log) => [
          { title: "OpenAI Refine Prompt", text: data.prompt || "", ts: new Date().toISOString() },
          ...log,
        ]);
        if (data.logs) {
          setStatusLog((log) => [...data.logs, ...log]);
        }
        setStatusMsg("OpenAI Refine pronto");
      })
      .catch((e) => setStatusMsg(`Errore OpenAI Refine: ${e.message}`))
      .finally(() => setAiBusy((b) => ({ ...b, refine: false })));
  };

  const runAITrained = () => {
    if (!selectedPdf || aiBusy.trained) return;
    const currentPdf = selectedPdf;
    setAiBusy((b) => ({ ...b, trained: true }));
    setStatusMsg("AI Trained in corso...");
    fetch("/api/ai/trained", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(apiKey ? { "X-OpenAI-Key": apiKey } : {}),
      },
      body: JSON.stringify({ pdf_name: selectedPdf }),
    })
      .then(async (res) => {
        if (!res.ok) {
          const txt = await res.text();
          throw new Error(txt || "AI Trained error");
        }
        return res.json();
      })
      .then((data) => {
        if (selectedPdf !== currentPdf) return;
        applyFieldsFromResult(data.fields);
        setLastPrompt("");
        setLastPromptTitle("");
        if (data.logs) {
          setStatusLog((log) => [...data.logs, ...log]);
        }
        setStatusMsg("AI Trained pronta");
      })
      .catch((e) => setStatusMsg(`Errore AI Trained: ${e.message}`))
      .finally(() => setAiBusy((b) => ({ ...b, trained: false })));
  };

  const restorePrimary = () => {
    if (!openaiPrimaryFields) return;
    applyFieldsFromResult(openaiPrimaryFields);
  };

  const runChemExtract = () => {
    if (sortedSelectedTokens.length === 0) return;
    const payload = sortedSelectedTokens.map((t) => ({
      id: t.id,
      text: t.text,
      bbox: t.bbox,
    }));
    fetch("/api/chem/extract", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tokens: payload }),
    })
      .then((res) => res.json())
      .then((data) => {
        const bindings = data.bindings || [];
        const rows = bindings.map((b) => ({
          elemento: b.elemento || "",
          valore: b.valore || "",
          token_ids: {
            el: b.el_ids || [],
            val: b.val_ids || [],
          },
        }));
        setChemRows(rows);
        setUsedTokenIds(data.used_token_ids || []);
        setActiveField("composizione_chimica");
      })
      .catch(() => {});
  };

  const saveResult = () => {
    if (!selectedPdf) return;
    setStatusMsg("Salvataggio...");
    const payload = {
      pdf: selectedPdf,
      page_idx: pageIdx,
      fields,
      fields_struct: fieldObjects,
      chem_rows: chemRows,
      field_meta: fieldMeta,
      updated_at: new Date().toISOString(),
    };
    fetch(`/api/pdfs/${encodeURIComponent(selectedPdf)}/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then(() => setStatusMsg("Salvato"))
      .catch(() => setStatusMsg("Errore salvataggio"));
  };

  useEffect(() => {
    setFields((f) => ({
      ...f,
      composizione_chimica: chemRows.length
        ? JSON.stringify(chemRows.map((r) => ({ elemento: r.elemento, valore: r.valore })))
        : "",
    }));
  }, [chemRows]);

  const addChemRow = () => {
    if (!chemEl.trim() && !chemVal.trim()) return;
    setChemRows((rows) => [
      ...rows,
      {
        elemento: chemEl.trim(),
        valore: chemVal.trim(),
        token_ids: {
          el: mode === "pick_el" ? selectedTokenIds : [],
          val: mode === "pick_val" ? selectedTokenIds : [],
        },
      },
    ]);
    setChemEl("");
    setChemVal("");
  };

  const updateChemRow = (idx, key, value) => {
    setChemRows((rows) =>
      rows.map((r, i) => (i === idx ? { ...r, [key]: value } : r))
    );
  };

  const removeChemRow = (idx) => {
    setChemRows((rows) => rows.filter((_, i) => i !== idx));
  };

  return (
    <div className="app">
      <header className="topbar">
        <h1>Certificate</h1>
        <div className="status">
          API: {status}
          {currentAiAction ? ` | In corso: ${currentAiAction}` : ""}
        </div>
        <div className="progress">
          {statusMsg}
          {tokensLoading ? " | Caricamento rettangoli..." : ""}
        </div>
        <div className="debug">{debugInfo}</div>
      </header>

      <section className="controls">
        <label>
          Cerca
          <input
            value={pdfSearch}
            onChange={(e) => setPdfSearch(e.target.value)}
            placeholder="Filtra PDF"
          />
        </label>
        <label>
          PDF
          <input value={selectedPdf} readOnly placeholder="Seleziona dalla lista" />
        </label>
        <label>
          OpenAI Key
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="sk-..."
          />
        </label>
        <button className="danger" onClick={() => deletePdf(selectedPdf)} disabled={!selectedPdf}>
          Elimina PDF
        </button>

        <div className="pager">
          <button
            onClick={() => setPageIdx((p) => Math.max(0, p - 1))}
            disabled={pageIdx <= 0}
          >
            Prev
          </button>
          <span>
            Page {pageIdx + 1} / {pageCount || 0}
          </span>
          <button
            onClick={() => setPageIdx((p) => Math.min(pageCount - 1, p + 1))}
            disabled={pageCount === 0 || pageIdx >= pageCount - 1}
          >
            Next
          </button>
        </div>
        <div className="zoom-controls">
          <button onClick={() => setZoom((z) => Math.max(0.5, +(z - 0.1).toFixed(2)))}>-</button>
          <span>{Math.round(zoom * 100)}%</span>
          <button onClick={() => setZoom((z) => Math.min(3, +(z + 0.1).toFixed(2)))}>+</button>
          <button onClick={() => setZoom(1)}>Reset</button>
        </div>

        <button className="ai-btn" onClick={runAI} disabled={!selectedPdf || aiBusy.extract}>
          AI Extract
        </button>
        <button className="ai-btn" onClick={runAIFull} disabled={!selectedPdf || aiBusy.full}>
          AI OpenAI Full
        </button>
        <button className="ai-btn" onClick={runAIPdf} disabled={!selectedPdf || aiBusy.pdf}>
          AI OpenAI PDF
        </button>
        <button className="ai-btn" onClick={runAIRefine} disabled={!selectedPdf || aiBusy.refine}>
          AI OpenAI Refine
        </button>
        <button className="ai-btn" onClick={runAITrained} disabled={!selectedPdf || aiBusy.trained}>
          AI Trained
        </button>
        <button className="save-btn" onClick={restorePrimary} disabled={!openaiPrimaryFields}>
          Restore Base
        </button>
        <button className="save-btn" onClick={saveResult} disabled={!selectedPdf}>
          Save JSON
        </button>
      </section>

      <main className="layout">
        <section className="viewer-area">
          <div
            className="viewer"
            onWheel={(e) => {
              e.preventDefault();
              const dir = e.deltaY < 0 ? 0.1 : -0.1;
              setZoom((z) => {
                const nz = Math.min(3, Math.max(0.5, +(z + dir).toFixed(2)));
                return nz;
              });
            }}
          >
            <div
              className="dropzone"
              onDragOver={(e) => e.preventDefault()}
              onDrop={onDrop}
            >
              <div>Drag & drop PDF qui</div>
              <label className="upload-btn">
                {uploading
                  ? `Uploading... ${uploadProgress}% (tot ${uploadTotalProgress}%)`
                  : "Seleziona PDF"}
                <input
                  type="file"
                  accept="application/pdf"
                  multiple
                  onChange={(e) => uploadFiles(e.target.files)}
                />
              </label>
            </div>
            {imageUrl ? (
              <div
                className="canvas-wrap"
                style={{
                  width: imgMeta.naturalW ? `${imgMeta.naturalW * zoom}px` : "auto",
                  height: imgMeta.naturalH ? `${imgMeta.naturalH * zoom}px` : "auto",
                }}
              >
                <img
                  ref={imgRef}
                  key={imageUrl}
                  src={imageUrl}
                  alt="PDF page"
                  onLoad={() => {
                    setImageLoaded(true);
                    updateImgMeta();
                  }}
                  onError={() => setImageLoaded(false)}
                  style={{
                    width: imgMeta.naturalW ? `${imgMeta.naturalW * zoom}px` : undefined,
                    height: imgMeta.naturalH ? `${imgMeta.naturalH * zoom}px` : undefined,
                    opacity: imageLoaded && !tokensLoading ? 1 : 0,
                  }}
                />
                <canvas
                  ref={canvasRef}
                  onMouseDown={onMouseDown}
                  onMouseMove={onMouseMove}
                  onMouseUp={onMouseUp}
                  style={{
                    opacity: imageLoaded && !tokensLoading ? 1 : 0,
                  }}
                />
                {(!imageLoaded || tokensLoading) && (
                  <div className="loading-overlay">
                    <div className="spinner" />
                    <div>Caricamento PDF + rettangoli...</div>
                  </div>
                )}
              </div>
            ) : (
              <div className="empty">No PDF found in the hardcoded folder.</div>
            )}
          </div>
        </section>

        <aside className="sidebar">
          <div className="panel-title">PDF</div>
          <div className="pdf-list">
            {pdfs
              .filter((p) => p.toLowerCase().includes(pdfSearch.toLowerCase()))
              .map((p) => (
                <div key={p} className={`pdf-item ${selectedPdf === p ? "active" : ""}`}>
                  <button onClick={() => selectPdf(p)}>{p}</button>
                  <button className="danger" onClick={() => deletePdf(p)}>
                    Elimina
                  </button>
                </div>
              ))}
          </div>

          <div className="panel-title">Campi</div>
          <div className="mode-row">
            <button className={mode === "select" ? "active" : ""} onClick={() => setMode("select")}>
              Select
            </button>
            <button className={mode === "delete" ? "active" : ""} onClick={() => setMode("delete")}>
              Elimina
            </button>
            <button className={mode === "catch" ? "active" : ""} onClick={() => setMode("catch")}>
              Catch
            </button>
            <button className={mode === "pick_el" ? "active" : ""} onClick={() => setMode("pick_el")}>
              Pick Elemento
            </button>
            <button className={mode === "pick_val" ? "active" : ""} onClick={() => setMode("pick_val")}>
              Pick Valore
            </button>
          </div>
          {Object.keys(fields).map((k) => (
            <button
              key={k}
              className={`field-btn ${activeField === k ? "active" : ""}`}
              onClick={() => setActiveField(k)}
            >
              {k}
            </button>
          ))}

          <div className="field-actions">
            <button onClick={applySelectionToField} disabled={selectedTokenIds.length === 0}>
              Catch
            </button>
            <button onClick={clearField}>Clear</button>
            <button onClick={runChemExtract} disabled={sortedSelectedTokens.length === 0}>
              Chem Extract
            </button>
          </div>

          {activeField === "composizione_chimica" ? (
            <div className="chem-editor">
              <div className="label">Tabella Chimica</div>
              <div className="mode-row">
                <button
                  className={mode === "select" ? "active" : ""}
                  onClick={() => setMode("select")}
                >
                  Select
                </button>
                <button
                  className={mode === "pick_el" ? "active" : ""}
                  onClick={() => setMode("pick_el")}
                >
                  Pick Elemento
                </button>
                <button
                  className={mode === "pick_val" ? "active" : ""}
                  onClick={() => setMode("pick_val")}
                >
                  Pick Valore
                </button>
              </div>
              <div className="chem-inputs">
                <input
                  placeholder="Elemento"
                  value={chemEl}
                  onChange={(e) => setChemEl(e.target.value)}
                />
                <input
                  placeholder="Valore"
                  value={chemVal}
                  onChange={(e) => setChemVal(e.target.value)}
                />
                <button onClick={addChemRow}>Add</button>
              </div>
              <div className="chem-rows">
                {chemRows.map((r, idx) => (
                  <div key={`${r.elemento}-${idx}`} className="chem-row">
                    <input
                      value={r.elemento || ""}
                      onChange={(e) => updateChemRow(idx, "elemento", e.target.value)}
                    />
                    <input
                      value={r.valore || ""}
                      onChange={(e) => updateChemRow(idx, "valore", e.target.value)}
                    />
                    <button onClick={() => removeChemRow(idx)}>Del</button>
                  </div>
                ))}
              </div>
            </div>
          ) : null}

          <div className="field-value">
            <div className="label">Valore</div>
            <textarea value={fields[activeField]} readOnly />
          </div>

          <div className="field-value">
            <div className="label">Token selezionati</div>
            <textarea
              value={sortedSelectedTokens.map((t) => t.text).join(" ")}
              readOnly
            />
          </div>
          <div className="prompt-panel">
            <div className="panel-title">Prompt Viewer</div>
            <div className="label">{lastPromptTitle || "Nessun prompt"}</div>
            <textarea value={lastPrompt} readOnly placeholder="Esegui un flusso OpenAI per vedere il prompt." />
            <div className="label">Log Prompt</div>
            <div className="prompt-log">
              {promptLog.map((p, idx) => (
                <button
                  key={`${p.ts}-${idx}`}
                  onClick={() => {
                    setLastPrompt(p.text);
                    setLastPromptTitle(p.title);
                  }}
                >
                  {p.title} — {p.ts}
                </button>
              ))}
            </div>
            <div className="label">Status Log</div>
            <div className="status-log">
              {statusLog.map((l, idx) => (
                <div key={`${idx}-${l.msg}`}>
                  [{l.ts}] {l.msg}
                </div>
              ))}
            </div>
          </div>
        </aside>
      </main>

      <footer className="statusbar">
        <div>{statusMsg}</div>
        {currentAiAction ? <div>In corso: {currentAiAction}</div> : null}
      </footer>
    </div>
  );
}
