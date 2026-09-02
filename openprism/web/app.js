(() => {
  "use strict";

  const PRESETS = Object.freeze({
    navigate: { name: "Navigate", readout: "FUSED NAVIGATION" },
    search: { name: "Search", readout: "FUSED TARGET SEARCH" },
    terrain: { name: "Terrain", readout: "TERRAIN VIEW" },
    integrity: { name: "Integrity", readout: "FUSION SUPPORT" },
    atlas: { name: "Atlas", readout: "TACTICAL ATLAS" },
  });
  const MAX_THERMAL_GAIN = 2.5;
  const ATLAS_STATUS_POLL_MS = 5000;
  const ATLAS_FRESHNESS_TICK_MS = 1000;
  const ATLAS_FUTURE_PUBLICATION_TOLERANCE_S = 5;
  // Do not connect detections across a telemetry/dropout gap longer than five
  // seconds. A new segment is safer than implying an unobserved trajectory.
  const ATLAS_TRACK_MAX_GAP_NS = 5_000_000_000n;

  const elements = {
    datasetSelect: document.querySelector("#datasetSelect"),
    splitSelect: document.querySelector("#splitSelect"),
    previousFrame: document.querySelector("#previousFrame"),
    nextFrame: document.querySelector("#nextFrame"),
    frameIndex: document.querySelector("#frameIndex"),
    frameTotal: document.querySelector("#frameTotal"),
    catalogState: document.querySelector("#catalogState"),
    catalogStateText: document.querySelector("#catalogStateText"),
    presetButtons: [...document.querySelectorAll("[data-preset]")],
    sceneShell: document.querySelector("#sceneShell"),
    sceneStage: document.querySelector("#sceneStage"),
    canvas: document.querySelector("#sceneCanvas"),
    sampleReadout: document.querySelector("#sampleReadout"),
    dimensionReadout: document.querySelector("#dimensionReadout"),
    coordinateReadout: document.querySelector("#coordinateReadout"),
    presetReadout: document.querySelector("#presetReadout"),
    loadingState: document.querySelector("#loadingState"),
    loadingDetail: document.querySelector("#loadingDetail"),
    errorState: document.querySelector("#errorState"),
    errorDetail: document.querySelector("#errorDetail"),
    retryButton: document.querySelector("#retryButton"),
    atlasState: document.querySelector("#atlasState"),
    atlasStateDetail: document.querySelector("#atlasStateDetail"),
    atlasRetryButton: document.querySelector("#atlasRetryButton"),
    sceneDescription: document.querySelector("#sceneDescription"),
    overlayDescription: document.querySelector("#overlayDescription"),
    thermalGain: document.querySelector("#thermalGain"),
    thermalOutput: document.querySelector("#thermalOutput"),
    autoFusionToggle: document.querySelector("#autoFusionToggle"),
    aiModeBadge: document.querySelector("#aiModeBadge"),
    aiSummary: document.querySelector("#aiSummary"),
    aiMetrics: document.querySelector("#aiMetrics"),
    aiReasons: document.querySelector("#aiReasons"),
    labelsToggle: document.querySelector("#labelsToggle"),
    labelsTitle: document.querySelector("#labelsTitle"),
    labelsDetail: document.querySelector("#labelsDetail"),
    lensToggle: document.querySelector("#lensToggle"),
    lensDetail: document.querySelector("#lensDetail"),
    evidenceInstructions: document.querySelector("#evidenceInstructions"),
    healthSummary: document.querySelector("#healthSummary"),
    sensorGrid: document.querySelector("#sensorGrid"),
    registrationValue: document.querySelector("#registrationValue"),
    registrationMeter: document.querySelector("#registrationMeter"),
    registrationBar: document.querySelector("#registrationBar"),
    machineChannels: document.querySelector("#machineChannels"),
    captureTimeStatus: document.querySelector("#captureTimeStatus"),
    sourceModeBadge: document.querySelector("#sourceModeBadge"),
    atlasLayerPanel: document.querySelector("#atlasLayerPanel"),
    atlasLayerButtons: [...document.querySelectorAll("[data-atlas-layer]")],
    terrainLegend: document.querySelector("#terrainLegend"),
    terrainLegendTitle: document.querySelector("#terrainLegendTitle"),
    terrainClassCount: document.querySelector("#terrainClassCount"),
    terrainClassList: document.querySelector("#terrainClassList"),
    provenanceCard: document.querySelector("#provenanceCard"),
    provenanceTitle: document.querySelector("#provenanceTitle"),
    provenanceDetail: document.querySelector("#provenanceDetail"),
    sceneA11ySummary: document.querySelector("#sceneA11ySummary"),
  };

  const context = elements.canvas.getContext("2d", { alpha: false });
  const state = {
    catalog: [],
    datasetId: "",
    splitId: "",
    index: 0,
    count: 0,
    thermalGain: Number(elements.thermalGain.value),
    autoFusion: elements.autoFusionToggle.checked,
    preset: "navigate",
    labelsVisible: elements.labelsToggle.checked,
    lensEnabled: elements.lensToggle.checked,
    frame: null,
    media: {},
    loading: true,
    catalogFailed: false,
    frameRequest: null,
    frameError: null,
    requestSerial: 0,
    atlas: null,
    atlasMedia: {},
    atlasStatus: "idle",
    atlasLayer: "composite",
    atlasRequest: null,
    atlasRequestSerial: 0,
    atlasStatusRequest: null,
    renderQueued: false,
    thermalTimer: null,
    pointer: { inside: false, x: 0, y: 0 },
    lastFit: null,
  };

  function clamp(value, minimum, maximum) {
    return Math.min(maximum, Math.max(minimum, value));
  }

  function finiteNumber(value, fallback = 0) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  function cleanText(value, fallback = "") {
    if (typeof value === "string" || typeof value === "number") {
      const text = String(value).trim();
      return text || fallback;
    }
    return fallback;
  }

  function taiNanoseconds(value) {
    const text = cleanText(value);
    if (!/^\d+$/.test(text)) return null;
    try {
      return BigInt(text);
    } catch (_error) {
      return null;
    }
  }

  function compareAtlasObservationTime(first, second) {
    const firstTime = taiNanoseconds(first.timestamp_tai_ns);
    const secondTime = taiNanoseconds(second.timestamp_tai_ns);
    if (firstTime === null && secondTime === null) return 0;
    if (firstTime === null) return -1;
    if (secondTime === null) return 1;
    if (firstTime < secondTime) return -1;
    if (firstTime > secondTime) return 1;
    return 0;
  }

  function atlasTrackPointsAreContinuous(first, second) {
    const firstTime = taiNanoseconds(first.timestamp_tai_ns);
    const secondTime = taiNanoseconds(second.timestamp_tai_ns);
    if (firstTime === null || secondTime === null || secondTime < firstTime) return false;
    return secondTime - firstTime <= ATLAS_TRACK_MAX_GAP_NS;
  }

  function atlasFreshness(meta, nowMs = Date.now()) {
    const declared = cleanText(meta?.freshness_status, "unverified").toLowerCase();
    const publishedMs = Date.parse(cleanText(meta?.published_at_utc));
    const suppliedTolerance = Number(meta?.future_publication_tolerance_s);
    const futureToleranceS = Number.isFinite(suppliedTolerance) && suppliedTolerance >= 0
      ? suppliedTolerance
      : ATLAS_FUTURE_PUBLICATION_TOLERANCE_S;
    if (!Number.isFinite(publishedMs)) {
      return { status: "unverified", ageS: null, futureToleranceS };
    }

    const rawAgeS = (nowMs - publishedMs) / 1000;
    if (declared === "future" || rawAgeS < -futureToleranceS) {
      return { status: "future", ageS: rawAgeS, futureToleranceS };
    }
    if (declared === "stale") {
      return { status: "stale", ageS: Math.max(0, rawAgeS), futureToleranceS };
    }
    if (declared === "snapshot" || declared === "unverified") {
      return { status: declared, ageS: Math.max(0, rawAgeS), futureToleranceS };
    }

    const ttlS = Number(meta?.freshness_ttl_s);
    if (!Number.isFinite(ttlS) || ttlS <= 0) {
      return { status: "unverified", ageS: Math.max(0, rawAgeS), futureToleranceS };
    }
    const ageS = Math.max(0, rawAgeS);
    return {
      status: ageS <= ttlS ? "fresh" : "stale",
      ageS,
      futureToleranceS,
    };
  }

  function sourceKind(source) {
    const value = cleanText(source).toLowerCase().replace(/[._-]+/g, " ");
    if (!value) return "none";
    if (/\b(ground\s*truth|gt|human|manual|reference)\b/.test(value) || /dataset\s+annot/.test(value)) {
      return "ground-truth";
    }
    if (/\b(model|inference|prediction|detector|network)\b/.test(value)) {
      return "model";
    }
    return "declared";
  }

  function formatConfidence(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "";
    const normalized = number <= 1 ? number * 100 : number;
    return `${clamp(normalized, 0, 100).toFixed(normalized < 10 ? 1 : 0)}%`;
  }

  function selectedDataset() {
    return state.catalog.find((dataset) => dataset.id === state.datasetId) || null;
  }

  function selectedSplit() {
    const dataset = selectedDataset();
    return dataset?.splits.find((split) => split.id === state.splitId) || null;
  }

  function normalizeCatalog(payload) {
    if (!payload || !Array.isArray(payload.datasets)) {
      throw new Error("Catalog response does not contain a datasets list.");
    }

    return payload.datasets
      .map((dataset) => {
        const id = cleanText(dataset?.id);
        if (!id) return null;
        const splits = Array.isArray(dataset.splits)
          ? dataset.splits
              .map((split) => {
                const splitId = cleanText(split?.id);
                if (!splitId) return null;
                return {
                  id: splitId,
                  label: cleanText(split.label, splitId),
                  count: Math.max(0, Math.trunc(finiteNumber(split.count))),
                };
              })
              .filter(Boolean)
          : [];
        return { id, label: cleanText(dataset.label, id), splits };
      })
      .filter((dataset) => dataset && dataset.splits.length > 0);
  }

  function makeOption(value, label) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    return option;
  }

  function populateDatasetSelect() {
    elements.datasetSelect.replaceChildren(
      ...state.catalog.map((dataset) => makeOption(dataset.id, dataset.label)),
    );
    elements.datasetSelect.value = state.datasetId;
    elements.datasetSelect.disabled = state.catalog.length === 0 || state.preset === "atlas";
  }

  function populateSplitSelect() {
    const dataset = selectedDataset();
    const splits = dataset?.splits || [];
    elements.splitSelect.replaceChildren(...splits.map((split) => makeOption(split.id, split.label)));
    elements.splitSelect.value = state.splitId;
    elements.splitSelect.disabled = splits.length === 0 || state.preset === "atlas";
  }

  function setCatalogStatus(mode, text) {
    elements.catalogState.classList.toggle("is-ready", mode === "ready");
    elements.catalogState.classList.toggle("is-error", mode === "error");
    elements.catalogStateText.textContent = text;
  }

  function setLoading(loading, detail = "") {
    state.loading = loading;
    if (detail) elements.loadingDetail.textContent = detail;
    syncSceneState();
    updateNavigation();
  }

  function showError(error, scope = "frame") {
    const message = error instanceof Error ? error.message : cleanText(error, "Unknown error.");
    state.catalogFailed = scope === "catalog";
    state.frameError = message;
    setLoading(false);
    elements.errorDetail.textContent = message;
    syncSceneState();
    if (scope === "catalog") setCatalogStatus("error", "Catalog error");
  }

  function clearError() {
    state.catalogFailed = false;
    state.frameError = null;
    syncSceneState();
  }

  function syncSceneState() {
    const atlasMode = state.preset === "atlas";
    const atlasLoading = atlasMode && state.atlasStatus === "loading";
    const atlasUnavailable = atlasMode && (state.atlasStatus === "unavailable" || state.atlasStatus === "error");
    elements.sceneShell.setAttribute("aria-busy", String(atlasMode ? atlasLoading : state.loading));
    elements.loadingState.hidden = atlasMode ? !atlasLoading : !state.loading;
    elements.errorState.hidden = atlasMode || !state.frameError;
    elements.atlasState.hidden = !atlasUnavailable;
  }

  function updateNavigation() {
    const usable = state.count > 0 && state.preset !== "atlas";
    elements.datasetSelect.disabled = state.catalog.length === 0 || state.preset === "atlas";
    elements.splitSelect.disabled = !selectedDataset()?.splits.length || state.preset === "atlas";
    elements.frameIndex.disabled = !usable || state.loading;
    elements.frameIndex.min = "1";
    elements.frameIndex.max = String(Math.max(1, state.count));
    elements.frameIndex.value = String(usable ? state.index + 1 : 1);
    elements.frameTotal.textContent = `/ ${usable ? state.count.toLocaleString() : "—"}`;
    elements.previousFrame.disabled = !usable || state.loading || state.index <= 0;
    elements.nextFrame.disabled = !usable || state.loading || state.index >= state.count - 1;
  }

  async function loadCatalog() {
    clearError();
    setCatalogStatus("loading", "Connecting");
    setLoading(true, "Reading the dataset catalog…");

    try {
      const response = await fetch("/api/catalog", {
        headers: { Accept: "application/json" },
        credentials: "same-origin",
      });
      if (!response.ok) throw new Error(`Catalog request failed (${response.status}).`);
      state.catalog = normalizeCatalog(await response.json());
      if (state.catalog.length === 0) throw new Error("No dataset splits are available in the catalog.");

      state.datasetId = state.catalog[0].id;
      state.splitId = state.catalog[0].splits[0].id;
      state.index = 0;
      state.count = state.catalog[0].splits[0].count;
      populateDatasetSelect();
      populateSplitSelect();
      updateNavigation();
      setCatalogStatus("ready", "Catalog ready");
      await loadFrame();
    } catch (error) {
      showError(error, "catalog");
    }
  }

  function trustedImageSource(value) {
    return typeof value === "string" && /^data:image\/[a-z0-9.+-]+(?:;[a-z0-9=+/_-]+)*;base64,/i.test(value);
  }

  function loadImage(source, key) {
    return new Promise((resolve, reject) => {
      if (!trustedImageSource(source)) {
        resolve(null);
        return;
      }
      const image = new Image();
      image.decoding = "async";
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error(`The ${key} image could not be decoded.`));
      image.src = source;
    });
  }

  async function decodeImages(images) {
    const keys = ["fused", "visible", "thermal", "support", "semantic"];
    const results = await Promise.allSettled(keys.map((key) => loadImage(images?.[key], key)));
    const media = {};
    const failures = [];
    results.forEach((result, index) => {
      const key = keys[index];
      if (result.status === "fulfilled" && result.value) media[key] = result.value;
      if (result.status === "rejected") failures.push(result.reason.message);
    });
    if (!media.fused && !media.visible && !media.thermal && !media.semantic && !media.support) {
      throw new Error(failures[0] || "The frame does not contain a usable image data URL.");
    }
    return media;
  }

  async function decodeAtlasImages(images) {
    const keys = ["rgb", "thermal", "support"];
    const results = await Promise.allSettled(keys.map((key) => loadImage(images?.[key], `atlas ${key}`)));
    const media = {};
    const failures = [];
    results.forEach((result, index) => {
      const key = keys[index];
      if (result.status === "fulfilled" && result.value) media[key] = result.value;
      if (result.status === "rejected") failures.push(result.reason.message);
    });
    if (keys.some((key) => !media[key])) {
      throw new Error(failures[0] || "The atlas bundle does not contain all three verified previews.");
    }
    const grids = new Set(keys.map((key) => `${media[key].naturalWidth}x${media[key].naturalHeight}`));
    if (grids.size !== 1) throw new Error("Atlas previews do not share one pixel grid.");
    return media;
  }

  function normalizeAtlas(payload) {
    if (!payload || typeof payload !== "object") throw new Error("Atlas response is empty.");
    if (payload.available !== true) {
      return {
        available: false,
        reason: cleanText(payload.reason, "No exported atlas bundle is available."),
      };
    }
    const meta = payload.meta && typeof payload.meta === "object" ? payload.meta : {};
    const grid = meta.grid && typeof meta.grid === "object" ? meta.grid : {};
    const coordinateReference = meta.coordinate_reference && typeof meta.coordinate_reference === "object"
      ? meta.coordinate_reference
      : {};
    const missionId = cleanText(meta.mission_id);
    if (!missionId || meta.survey_grade !== false) {
      throw new Error("Atlas identity or tactical-grade declaration is invalid.");
    }
    const declaredOrigin = cleanText(payload.origin_status || meta.origin_status).toLowerCase();
    const originStatus = ["captured_evidence", "synthetic_demo", "unverified"].includes(declaredOrigin)
      ? declaredOrigin
      : "unverified";
    const objects = Array.isArray(payload.objects)
      ? payload.objects.slice(0, 10000).map((item) => {
        if (!item || typeof item !== "object") return null;
        const eastM = Number(item.east_m);
        const northM = Number(item.north_m);
        const objectId = cleanText(item.object_id);
        const label = cleanText(item.label);
        if (!Number.isFinite(eastM) || !Number.isFinite(northM) || !objectId || !label) return null;
        const confidence = Number(item.confidence);
        const horizontalUncertaintyM = Number(item.horizontal_uncertainty_m);
        return {
          object_id: objectId,
          label,
          east_m: eastM,
          north_m: northM,
          confidence: Number.isFinite(confidence) ? clamp(confidence, 0, 1) : null,
          horizontal_uncertainty_m: Number.isFinite(horizontalUncertaintyM) && horizontalUncertaintyM >= 0
            ? horizontalUncertaintyM
            : null,
          timestamp_tai_ns: cleanText(item.timestamp_tai_ns),
        };
      }).filter(Boolean)
      : [];
    return {
      available: true,
      meta: {
        ...meta,
        mission_id: missionId,
        origin_status: originStatus,
        synthetic: originStatus === "synthetic_demo",
        width: Math.max(1, Math.trunc(finiteNumber(meta.width, 1))),
        height: Math.max(1, Math.trunc(finiteNumber(meta.height, 1))),
        grid,
        coordinate_reference: coordinateReference,
        layers: Array.isArray(meta.layers) ? meta.layers.map((item) => cleanText(item)).filter(Boolean) : [],
      },
      images: payload.images && typeof payload.images === "object" ? payload.images : {},
      objects,
      provenance: payload.provenance && typeof payload.provenance === "object" ? payload.provenance : {},
    };
  }

  async function loadAtlas(force = false) {
    if (!force && state.atlasStatus === "available") return;
    state.atlasRequest?.abort();
    const controller = new AbortController();
    state.atlasRequest = controller;
    const serial = ++state.atlasRequestSerial;
    state.atlasStatus = "loading";
    elements.loadingDetail.textContent = "Reading the latest exported mission atlas…";
    syncSceneState();
    scheduleRender();

    try {
      const response = await fetch("/api/atlas", {
        headers: { Accept: "application/json" },
        credentials: "same-origin",
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`Atlas request failed (${response.status}).`);
      const atlas = normalizeAtlas(await response.json());
      if (serial !== state.atlasRequestSerial) return;
      state.atlas = atlas;
      if (!atlas.available) {
        state.atlasMedia = {};
        state.atlasStatus = "unavailable";
        elements.atlasStateDetail.textContent = `${atlas.reason} Export a mission bundle to output/openprism_atlas/latest.`;
      } else {
        state.atlasMedia = await decodeAtlasImages(atlas.images);
        if (serial !== state.atlasRequestSerial) return;
        state.atlasStatus = "available";
      }
      if (state.preset === "atlas") updateAtlasInterface();
      syncSceneState();
      scheduleRender();
    } catch (error) {
      if (error?.name === "AbortError" || serial !== state.atlasRequestSerial) return;
      state.atlas = null;
      state.atlasMedia = {};
      state.atlasStatus = "error";
      elements.atlasStateDetail.textContent = error instanceof Error ? error.message : "The atlas could not be loaded.";
      if (state.preset === "atlas") updateAtlasInterface();
      syncSceneState();
      scheduleRender();
    }
  }

  async function pollAtlasStatus() {
    if (
      state.preset !== "atlas"
      || state.atlasStatus === "loading"
      || state.atlasStatusRequest
      || document.visibilityState === "hidden"
    ) return;

    const controller = new AbortController();
    state.atlasStatusRequest = controller;
    try {
      const response = await fetch("/api/atlas/status", {
        headers: { Accept: "application/json" },
        credentials: "same-origin",
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`Atlas status request failed (${response.status}).`);
      const status = await response.json();
      if (!status || typeof status !== "object") throw new Error("Atlas status response is empty.");

      if (status.available !== true) {
        state.atlas = {
          available: false,
          reason: cleanText(status.reason, "The current atlas generation is unavailable."),
        };
        state.atlasMedia = {};
        state.atlasStatus = "unavailable";
        elements.atlasStateDetail.textContent = `${state.atlas.reason} Export a mission bundle to output/openprism_atlas/latest.`;
      } else {
        const revisionId = cleanText(status.revision_id);
        const currentRevisionId = cleanText(state.atlas?.meta?.revision_id);
        if (!revisionId || revisionId !== currentRevisionId) {
          await loadAtlas(true);
          return;
        }
        // Refresh the server's independent clock assessment while the client
        // continuously advances age locally between status polls.
        for (const key of [
          "published_at_utc",
          "publication_age_s",
          "freshness_ttl_s",
          "freshness_status",
          "future_publication_tolerance_s",
        ]) {
          state.atlas.meta[key] = status[key];
        }
      }
      updateAtlasInterface();
      syncSceneState();
      scheduleRender();
    } catch (error) {
      // Retain the last verified generation on a transient status transport
      // failure. Its locally recomputed TTL will still age into a stale state.
      if (error?.name !== "AbortError") scheduleRender();
    } finally {
      if (state.atlasStatusRequest === controller) state.atlasStatusRequest = null;
    }
  }

  function normalizeFrame(payload) {
    if (!payload || typeof payload !== "object") throw new Error("Frame response is empty.");
    const meta = payload.meta && typeof payload.meta === "object" ? payload.meta : {};
    const images = payload.images && typeof payload.images === "object" ? payload.images : {};
    const detections = Array.isArray(payload.detections)
      ? payload.detections.filter((item) => item && typeof item === "object")
      : [];
    const terrainClasses = Array.isArray(payload.terrain_classes)
      ? payload.terrain_classes
          .map((terrainClass, position) => {
            if (!terrainClass || typeof terrainClass !== "object") return null;
            const id = cleanText(terrainClass.id, String(position));
            const label = cleanText(terrainClass.label, `Class ${id}`);
            const suppliedColor = cleanText(terrainClass.color);
            const color = /^#(?:[0-9a-f]{3,4}|[0-9a-f]{6}|[0-9a-f]{8})$/i.test(suppliedColor)
              ? suppliedColor
              : "#7f988b";
            const rawCoverage = Number(terrainClass.coverage);
            const coverage = Number.isFinite(rawCoverage) ? clamp(rawCoverage, 0, 1) : 0;
            return { id, label, color, coverage };
          })
          .filter(Boolean)
          .sort((left, right) => right.coverage - left.coverage)
      : [];
    const ai = payload.ai && typeof payload.ai === "object" ? payload.ai : null;
    return { meta, images, detections, terrainClasses, ai };
  }

  async function loadFrame() {
    if (!state.datasetId || !state.splitId) return;
    if (state.count <= 0) {
      showError(new Error("This split contains no frames."));
      return;
    }

    state.frameRequest?.abort();
    const controller = new AbortController();
    state.frameRequest = controller;
    const serial = ++state.requestSerial;
    clearError();
    setLoading(true, `Loading frame ${state.index + 1} of ${state.count.toLocaleString()}…`);

    const query = new URLSearchParams({
      dataset: state.datasetId,
      split: state.splitId,
      index: String(state.index),
      thermal_gain: ((state.thermalGain / 100) * MAX_THERMAL_GAIN).toFixed(2),
      automatic_control: String(state.autoFusion),
    });

    try {
      const response = await fetch(`/api/frame?${query}`, {
        headers: { Accept: "application/json" },
        credentials: "same-origin",
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`Frame request failed (${response.status}).`);
      const frame = normalizeFrame(await response.json());
      const media = await decodeImages(frame.images);
      if (serial !== state.requestSerial) return;

      state.frame = frame;
      state.media = media;
      const aiControl = frame.ai?.control;
      const recommendation = aiControl?.recommendation;
      if (state.autoFusion && recommendation && typeof recommendation === "object") {
        const appliedGain = Number(aiControl.applied_thermal_gain);
        if (Number.isFinite(appliedGain)) {
          state.thermalGain = clamp(Math.round((appliedGain / MAX_THERMAL_GAIN) * 100), 0, 100);
          elements.thermalGain.value = String(state.thermalGain);
          elements.thermalOutput.value = `${state.thermalGain}%`;
          elements.thermalOutput.textContent = `${state.thermalGain}%`;
        }
        const recommendedPreset = cleanText(recommendation.operator_preset).toLowerCase();
        if (PRESETS[recommendedPreset] && recommendedPreset !== "atlas") {
          setPreset(recommendedPreset);
        }
      }
      const responseCount = Math.trunc(finiteNumber(frame.meta.count));
      if (responseCount > 0) state.count = responseCount;
      const responseIndex = Math.trunc(finiteNumber(frame.meta.index, state.index));
      state.index = clamp(responseIndex, 0, Math.max(0, state.count - 1));
      if (state.preset !== "atlas") updateFrameInterface();
      setLoading(false);
      setCatalogStatus("ready", "Catalog ready");
      scheduleRender();
    } catch (error) {
      if (error?.name === "AbortError" || serial !== state.requestSerial) return;
      showError(error);
    }
  }

  function frameDimensions() {
    if (state.preset === "atlas" && state.atlasStatus === "available") {
      const meta = state.atlas?.meta || {};
      const fallback = state.atlasMedia.rgb || state.atlasMedia.thermal || state.atlasMedia.support;
      return {
        width: Math.max(1, Math.trunc(finiteNumber(meta.width, fallback?.naturalWidth || 1))),
        height: Math.max(1, Math.trunc(finiteNumber(meta.height, fallback?.naturalHeight || 1))),
      };
    }
    const meta = state.frame?.meta || {};
    const fallback = state.media.fused || state.media.visible || state.media.thermal || state.media.semantic || state.media.support;
    return {
      width: Math.max(1, Math.trunc(finiteNumber(meta.width, fallback?.naturalWidth || 1))),
      height: Math.max(1, Math.trunc(finiteNumber(meta.height, fallback?.naturalHeight || 1))),
    };
  }

  function displayMachineChannels(value) {
    if (Array.isArray(value)) {
      return value.map((item) => cleanText(item)).filter(Boolean).join(" · ") || "—";
    }
    if (value && typeof value === "object") {
      return Object.entries(value)
        .filter(([, available]) => Boolean(available))
        .map(([channel]) => cleanText(channel))
        .filter(Boolean)
        .join(" · ") || "—";
    }
    const declared = cleanText(value);
    if (declared) return declared;
    return [state.media.semantic && "semantic", state.media.support && "support"]
      .filter(Boolean)
      .join(" · ") || "—";
  }

  function annotationSource() {
    const declared = cleanText(state.frame?.meta?.annotation_source);
    if (declared) return declared;
    const detectionSources = [...new Set(
      (state.frame?.detections || []).map((detection) => cleanText(detection.source)).filter(Boolean),
    )];
    return detectionSources.join(" + ");
  }

  function updateMediaAvailability() {
    const channels = [
      { id: "RGB", detail: "visible view", available: Boolean(state.media.visible) },
      { id: "TIR", detail: "normalized view", available: Boolean(state.media.thermal) },
      { id: "FUS", detail: "derived fusion", available: Boolean(state.media.fused) },
    ];
    elements.sensorGrid.replaceChildren(
      ...channels.map((channel) => {
        const chip = document.createElement("div");
        chip.className = `sensor-chip${channel.available ? "" : " is-offline"}`;
        const name = document.createElement("span");
        const detail = document.createElement("small");
        name.textContent = channel.id;
        detail.textContent = channel.available ? channel.detail : "unavailable";
        chip.append(name, detail);
        return chip;
      }),
    );

    const availableCount = channels.filter((channel) => channel.available).length;
    elements.healthSummary.textContent = availableCount ? `${availableCount}/3 available` : "No media";
    elements.healthSummary.classList.toggle("is-ready", availableCount === 3);
  }

  function updateRegistrationStatus() {
    const meta = state.frame?.meta || {};
    const scoreSource = meta.registration_confidence;
    const score = Number(scoreSource);
    const scoreAvailable = scoreSource !== null
      && scoreSource !== undefined
      && scoreSource !== ""
      && Number.isFinite(score);
    const percentage = scoreAvailable ? clamp(score <= 1 ? score * 100 : score, 0, 100) : 0;
    const status = cleanText(meta.registration_status).toLowerCase().replace(/[._-]+/g, " ");
    const unmeasured = /\b(unmeasured|not\s+measured|declared|assumed)\b/.test(status);
    const measured = !unmeasured && /\bmeasured\b/.test(status);
    const estimated = !unmeasured && /\b(estimated|computed|derived)\b/.test(status);

    elements.registrationMeter.classList.toggle("is-declared", unmeasured);
    elements.registrationMeter.classList.toggle("is-estimated", estimated || (scoreAvailable && !measured));
    elements.registrationMeter.removeAttribute("aria-valuenow");
    elements.registrationMeter.removeAttribute("aria-valuemin");
    elements.registrationMeter.removeAttribute("aria-valuemax");
    elements.registrationBar.style.width = "0%";

    if (unmeasured) {
      elements.registrationValue.textContent = "Publisher-declared · unmeasured";
      elements.registrationValue.title = cleanText(meta.registration_status, "Publisher-declared; not measured");
      elements.registrationMeter.setAttribute("role", "img");
      elements.registrationMeter.setAttribute("aria-label", "Publisher-declared registration; confidence was not empirically measured");
      elements.registrationMeter.removeAttribute("aria-valuetext");
      return;
    }

    if (!scoreAvailable) {
      elements.registrationValue.textContent = "Not supplied";
      elements.registrationValue.removeAttribute("title");
      elements.registrationMeter.setAttribute("role", "img");
      elements.registrationMeter.setAttribute("aria-label", "Registration evidence not supplied");
      elements.registrationMeter.removeAttribute("aria-valuetext");
      return;
    }

    const descriptor = measured ? "Measured" : estimated ? "Estimated" : "Supplied score";
    const formatted = `${percentage.toFixed(percentage < 10 ? 1 : 0)}%`;
    elements.registrationValue.textContent = `${descriptor} · ${formatted}`;
    elements.registrationValue.title = status || "Measurement status not declared";
    elements.registrationBar.style.width = `${percentage}%`;
    elements.registrationMeter.setAttribute("role", "meter");
    elements.registrationMeter.setAttribute("aria-label", `${descriptor} registration score`);
    elements.registrationMeter.setAttribute("aria-valuemin", "0");
    elements.registrationMeter.setAttribute("aria-valuemax", "100");
    elements.registrationMeter.setAttribute("aria-valuenow", String(Math.round(percentage)));
    elements.registrationMeter.setAttribute("aria-valuetext", `${descriptor}, ${percentage.toFixed(0)} percent`);
  }

  function updatePlaybackStatus() {
    const meta = state.frame?.meta || {};
    const explicitTime = cleanText(meta.capture_time || meta.capture_timestamp);
    const status = cleanText(meta.capture_time_status).toLowerCase().replace(/[._-]+/g, " ");
    let captureLabel;
    if (explicitTime) {
      captureLabel = `Capture time ${explicitTime}`;
    } else if (/unavailable.*extracted|extracted.*unavailable/.test(status)) {
      captureLabel = "Capture time unavailable in extracted pair";
    } else if (/\b(unavailable|missing|unknown|not\s+available)\b/.test(status)) {
      captureLabel = "Capture time unavailable";
    } else if (status) {
      captureLabel = `Capture time: ${status}`;
    } else {
      captureLabel = "Capture time not supplied";
    }

    const synchronizationState = cleanText(meta.synchronization_state, "unknown")
      .toLowerCase()
      .replace(/[._-]+/g, " ");
    const synchronizationBasis = cleanText(meta.synchronization_basis, "unknown")
      .toLowerCase();
    const pixelFusionApplied = meta.pixel_fusion_applied === true;
    const synchronizationLabel = synchronizationBasis === "declared"
      ? `Declared replay alignment · ${pixelFusionApplied ? "pixel fusion active" : "pixel fusion off"}`
      : `${synchronizationState} · ${pixelFusionApplied ? "pixel fusion active" : "pixel fusion off"}`;
    elements.sourceModeBadge.textContent = "REPLAY";
    elements.sourceModeBadge.classList.remove("is-atlas", "is-synthetic", "is-unverified");
    elements.captureTimeStatus.textContent = `${captureLabel} · ${synchronizationLabel}`;
    elements.captureTimeStatus.title = elements.captureTimeStatus.textContent;
  }

  function updateProvenance() {
    const source = annotationSource();
    const kind = sourceKind(source);
    const detections = state.frame?.detections || [];
    elements.provenanceCard.classList.toggle("is-ground-truth", kind === "ground-truth");
    elements.provenanceCard.classList.toggle("is-model", kind === "model");
    elements.provenanceCard.classList.remove("is-atlas");

    if (!source && detections.length === 0) {
      elements.provenanceTitle.textContent = "No annotations";
      elements.provenanceDetail.textContent = "The browser renders supplied evidence only. It does not run or imply model inference.";
      elements.overlayDescription.textContent = "No annotation overlay supplied.";
      return;
    }

    if (kind === "ground-truth") {
      elements.provenanceTitle.textContent = `GROUND TRUTH · ${source}`;
      elements.provenanceDetail.textContent = "Dataset-supplied ground-truth annotations. No inference runs in this browser.";
      elements.overlayDescription.textContent = `GROUND TRUTH overlay · ${detections.length} annotation${detections.length === 1 ? "" : "s"}.`;
      return;
    }

    if (kind === "model") {
      elements.provenanceTitle.textContent = `MODEL OUTPUT · ${source}`;
      elements.provenanceDetail.textContent = "Externally supplied model output; inference does not run in this browser.";
      elements.overlayDescription.textContent = `MODEL OUTPUT overlay · ${detections.length} detection${detections.length === 1 ? "" : "s"}.`;
      return;
    }

    elements.provenanceTitle.textContent = source ? `DECLARED SOURCE · ${source}` : "DECLARED OVERLAY";
    elements.provenanceDetail.textContent = "The supplied source is shown verbatim and is not reclassified as model inference or ground truth.";
    elements.overlayDescription.textContent = `Declared-source overlay · ${detections.length} item${detections.length === 1 ? "" : "s"}.`;
  }

  function terrainCoverageLabel(coverage) {
    const percentage = clamp(finiteNumber(coverage) * 100, 0, 100);
    return `${percentage.toFixed(percentage > 0 && percentage < 10 ? 1 : 0)}%`;
  }

  function updateTerrainLegend() {
    const terrainClasses = state.frame?.terrainClasses || [];
    const visible = state.preset === "terrain" && terrainClasses.length > 0;
    elements.terrainLegend.hidden = !visible;
    if (!visible) return;

    const kind = sourceKind(annotationSource());
    elements.terrainLegendTitle.textContent = kind === "ground-truth"
      ? "GROUND TRUTH TERRAIN"
      : kind === "model"
        ? "MODEL TERRAIN OUTPUT"
        : kind === "declared"
          ? "DECLARED TERRAIN"
          : "TERRAIN CLASSES";
    elements.terrainClassCount.textContent = `${terrainClasses.length} class${terrainClasses.length === 1 ? "" : "es"}`;

    elements.terrainClassList.replaceChildren(
      ...terrainClasses.map((terrainClass) => {
        const item = document.createElement("li");
        item.className = "terrain-class-item";
        const swatch = document.createElement("span");
        const label = document.createElement("span");
        const coverage = document.createElement("span");
        swatch.className = "terrain-class-swatch";
        swatch.style.backgroundColor = terrainClass.color;
        swatch.setAttribute("aria-hidden", "true");
        label.className = "terrain-class-label";
        label.textContent = terrainClass.label;
        label.title = terrainClass.label;
        coverage.className = "terrain-class-coverage";
        coverage.textContent = terrainCoverageLabel(terrainClass.coverage);
        item.setAttribute("aria-label", `${terrainClass.label}, ${coverage.textContent} coverage`);
        item.append(swatch, label, coverage);
        return item;
      }),
    );
  }

  function updateAIAdvisor() {
    const metricValues = [...elements.aiMetrics.querySelectorAll("dd")];
    elements.aiModeBadge.classList.remove("is-manual", "is-override");
    if (state.preset === "atlas") {
      elements.autoFusionToggle.disabled = true;
      elements.autoFusionToggle.closest(".switch-row")?.classList.add("is-disabled");
      elements.aiModeBadge.textContent = "ATLAS";
      elements.aiSummary.textContent = "Mission Atlas keeps its layer and freshness controls separate from the per-frame fusion policy.";
      ["atlas", "map", "—"].forEach((value, index) => {
        if (metricValues[index]) metricValues[index].textContent = value;
      });
      elements.aiReasons.replaceChildren();
      return;
    }

    elements.autoFusionToggle.disabled = false;
    elements.autoFusionToggle.closest(".switch-row")?.classList.remove("is-disabled");
    const digest = state.frame?.ai;
    const control = digest?.control;
    const recommendation = control?.recommendation;
    if (!digest || !control || !recommendation) {
      elements.aiModeBadge.textContent = state.autoFusion ? "AUTO" : "MANUAL";
      elements.aiModeBadge.classList.toggle("is-manual", !state.autoFusion);
      elements.aiSummary.textContent = "Waiting for a validated machine-readable scene digest.";
      ["—", "—", "—"].forEach((value, index) => {
        if (metricValues[index]) metricValues[index].textContent = value;
      });
      elements.aiReasons.replaceChildren();
      return;
    }

    const override = cleanText(recommendation.status) === "safety_override";
    const learnedActive = digest?.learned_fusion?.active === true;
    elements.aiModeBadge.textContent = override
      ? "SAFE"
      : learnedActive && state.autoFusion
        ? "EGT"
        : state.autoFusion
          ? "AUTO"
          : "ADVICE";
    elements.aiModeBadge.classList.toggle("is-manual", !state.autoFusion && !override);
    elements.aiModeBadge.classList.toggle("is-override", override);
    elements.aiSummary.textContent = cleanText(
      digest.summary,
      "The policy produced a control recommendation from the current evidence.",
    );
    const preset = cleanText(recommendation.operator_preset, "integrity");
    const strength = Number(recommendation.thermal_strength_percent);
    const confidence = Number(recommendation.confidence);
    const values = [
      preset,
      Number.isFinite(strength) ? `${clamp(strength, 0, 100).toFixed(0)}%` : "—",
      Number.isFinite(confidence) ? `${clamp(confidence, 0, 1).toFixed(2)}` : "—",
    ];
    values.forEach((value, index) => {
      if (metricValues[index]) metricValues[index].textContent = value;
    });
    const reasons = Array.isArray(recommendation.reasons)
      ? recommendation.reasons.map((item) => cleanText(item)).filter(Boolean).slice(0, 4)
      : [];
    elements.aiReasons.replaceChildren(
      ...reasons.map((reason) => {
        const item = document.createElement("li");
        item.textContent = reason;
        return item;
      }),
    );
  }

  function updateFrameInterface() {
    const meta = state.frame.meta;
    const dimensions = frameDimensions();
    const sampleId = cleanText(meta.sample_id, `frame-${state.index + 1}`);
    elements.labelsToggle.disabled = false;
    elements.labelsToggle.closest(".switch-row")?.classList.remove("is-disabled");
    elements.lensToggle.disabled = false;
    elements.lensToggle.closest(".switch-row")?.classList.remove("is-disabled");
    elements.thermalGain.disabled = state.autoFusion;
    elements.labelsTitle.textContent = "Labels";
    elements.labelsDetail.textContent = "Show declared annotations";
    elements.lensDetail.textContent = "Inspect a registered source view";
    elements.evidenceInstructions.textContent = "The lens preserves one scene window and reveals the complementary registered view at the same coordinates. Thermal imagery is normalized and color-mapped.";
    elements.sampleReadout.textContent = sampleId.toUpperCase();
    elements.dimensionReadout.textContent = `${dimensions.width.toLocaleString()} × ${dimensions.height.toLocaleString()}`;
    elements.sceneDescription.textContent = `${cleanText(meta.dataset, state.datasetId)} / ${cleanText(meta.split, state.splitId)} / ${sampleId}`;
    elements.machineChannels.textContent = displayMachineChannels(meta.machine_channels);
    elements.machineChannels.title = elements.machineChannels.textContent;
    updateMediaAvailability();
    updateRegistrationStatus();
    updatePlaybackStatus();
    updateProvenance();
    updateAIAdvisor();
    updateTerrainLegend();
    updateNavigation();
    updatePresetReadout();
    updateAccessibilitySummary();
  }

  function atlasLayerLabel() {
    return {
      composite: "RGB mosaic + normalized thermal display overlay",
      thermal: "normalized thermal mosaic",
      support: "projection-support mosaic",
    }[state.atlasLayer] || "tactical mosaic";
  }

  function updateAtlasInterface() {
    elements.atlasLayerPanel.hidden = state.preset !== "atlas";
    elements.terrainLegend.hidden = true;
    elements.sourceModeBadge.textContent = "ATLAS";
    elements.sourceModeBadge.classList.remove("is-synthetic", "is-unverified");
    elements.sourceModeBadge.classList.add("is-atlas");
    elements.provenanceCard.classList.remove("is-ground-truth", "is-model");
    elements.provenanceCard.classList.add("is-atlas");
    elements.labelsToggle.disabled = true;
    elements.labelsToggle.closest(".switch-row")?.classList.add("is-disabled");
    elements.labelsTitle.textContent = "Object tracks";
    elements.labelsDetail.textContent = "Separate track layer required";
    elements.lensDetail.textContent = "Inspect a complementary atlas layer";
    elements.evidenceInstructions.textContent = "The lens stays coordinate-locked while revealing thermal or RGB atlas evidence. Dynamic people and vehicles belong in a separate time-varying track layer.";

    const available = state.atlasStatus === "available" && state.atlas?.available;
    elements.atlasLayerButtons.forEach((button) => { button.disabled = !available; });
    elements.thermalGain.disabled = !available || state.atlasLayer !== "composite";
    updateAIAdvisor();
    elements.lensToggle.disabled = !available;
    elements.lensToggle.closest(".switch-row")?.classList.toggle("is-disabled", !available);
    const channels = [
      { id: "RGB", detail: "north-up mosaic", available: Boolean(available && state.atlasMedia.rgb) },
      { id: "TIR", detail: "normalized mosaic", available: Boolean(available && state.atlasMedia.thermal) },
      { id: "SUP", detail: "projection support", available: Boolean(available && state.atlasMedia.support) },
    ];
    elements.sensorGrid.replaceChildren(
      ...channels.map((channel) => {
        const chip = document.createElement("div");
        chip.className = `sensor-chip${channel.available ? "" : " is-offline"}`;
        const name = document.createElement("span");
        const detail = document.createElement("small");
        name.textContent = channel.id;
        detail.textContent = channel.available ? channel.detail : "unavailable";
        chip.append(name, detail);
        return chip;
      }),
    );
    const availableCount = channels.filter((channel) => channel.available).length;
    elements.healthSummary.textContent = availableCount ? `${availableCount}/3 available` : "No bundle";
    elements.healthSummary.classList.toggle("is-ready", availableCount === 3);

    elements.registrationMeter.classList.add("is-declared");
    elements.registrationMeter.classList.remove("is-estimated");
    elements.registrationMeter.setAttribute("role", "img");
    elements.registrationMeter.removeAttribute("aria-valuenow");
    elements.registrationMeter.removeAttribute("aria-valuemin");
    elements.registrationMeter.removeAttribute("aria-valuemax");
    elements.registrationMeter.removeAttribute("aria-valuetext");
    elements.registrationBar.style.width = "0%";

    if (!available) {
      elements.sampleReadout.textContent = "NO EXPORTED ATLAS";
      elements.dimensionReadout.textContent = "— × —";
      elements.sceneDescription.textContent = "Atlas / no georeferenced mission bundle";
      elements.overlayDescription.textContent = "No map is fabricated from dataset replay frames.";
      elements.machineChannels.textContent = "—";
      elements.machineChannels.removeAttribute("title");
      elements.registrationValue.textContent = "No georeferenced evidence";
      elements.registrationValue.removeAttribute("title");
      elements.registrationMeter.setAttribute("aria-label", "No georeferenced atlas evidence available");
      elements.captureTimeStatus.textContent = "Awaiting an exported mission bundle";
      elements.captureTimeStatus.removeAttribute("title");
      elements.provenanceTitle.textContent = "NO MAP PRODUCT";
      elements.provenanceDetail.textContent = "OpenPRISM will show only a validated atlas bundle. Replay imagery is never assigned invented coordinates.";
      updatePresetReadout();
      updateAccessibilitySummary();
      updateNavigation();
      return;
    }

    const meta = state.atlas.meta;
    const dimensions = frameDimensions();
    const mission = cleanText(meta.mission_id, "unnamed mission");
    const originStatus = cleanText(meta.origin_status, "unverified");
    const synthetic = originStatus === "synthetic_demo";
    const captured = originStatus === "captured_evidence";
    const dynamicExcluded = meta.dynamic_objects_excluded_from_static_atlas === true;
    const objectCount = state.atlas.objects.length;
    const trackCount = Math.max(0, Math.trunc(finiteNumber(meta.track_count, objectCount)));
    const freshnessStatus = atlasFreshness(meta).status;
    const freshnessLabel = freshnessStatus === "fresh"
      ? "LIVE TTL VERIFIED"
      : freshnessStatus === "stale"
        ? "STALE SNAPSHOT"
        : freshnessStatus === "future"
          ? "CLOCK ERROR · FUTURE SNAPSHOT"
        : freshnessStatus === "snapshot"
          ? "MISSION SNAPSHOT"
          : "FRESHNESS UNVERIFIED";
    elements.labelsToggle.disabled = objectCount === 0;
    elements.labelsToggle.closest(".switch-row")?.classList.toggle("is-disabled", objectCount === 0);
    elements.labelsDetail.textContent = dynamicExcluded
      ? `${trackCount} track${trackCount === 1 ? "" : "s"} · ${objectCount} observation${objectCount === 1 ? "" : "s"} · excluded from terrain`
      : "Exclusion not declared";
    const accepted = Math.max(0, Math.trunc(finiteNumber(meta.accepted_capture_count)));
    const rejected = Math.max(0, Math.trunc(finiteNumber(meta.rejected_capture_count)));
    const resolution = Number(meta.grid?.resolution_m);
    const resolutionLabel = Number.isFinite(resolution) ? `${resolution.toFixed(resolution < 1 ? 2 : 1)} m/cell` : "resolution not supplied";
    elements.sampleReadout.textContent = mission.toUpperCase();
    elements.dimensionReadout.textContent = `${dimensions.width.toLocaleString()} × ${dimensions.height.toLocaleString()}`;
    elements.sceneDescription.textContent = `${synthetic ? "SYNTHETIC DEMO" : captured ? "Atlas" : "UNVERIFIED ATLAS"} / ${mission} / ${atlasLayerLabel()}`;
    elements.overlayDescription.textContent = synthetic
      ? `SYNTHETIC · no real sensor or GPS data · ${resolutionLabel} · not survey-grade.`
      : captured
        ? `North-up tactical 2.5D · ${resolutionLabel} · not survey-grade.`
        : `UNVERIFIED DATA ORIGIN · ${resolutionLabel} · do not use operationally.`;
    elements.machineChannels.textContent = meta.layers.length ? meta.layers.join(" · ") : "RGB · thermal · support";
    elements.machineChannels.title = elements.machineChannels.textContent;
    elements.registrationValue.textContent = synthetic
      ? "Analytic demo geometry"
      : captured
        ? "Pose-projected · not survey-grade"
        : "Origin unverified";
    elements.registrationValue.title = synthetic
      ? "Synthetic analytic camera poses; not recorded Pixhawk telemetry"
      : captured
        ? "Camera calibration and Pixhawk pose projection; no bundle-adjusted survey accuracy claim"
        : "The bundle does not explicitly confirm both real sensor and real navigation data";
    elements.registrationMeter.setAttribute("aria-label", elements.registrationValue.title);
    elements.captureTimeStatus.textContent = `${freshnessLabel} · ${accepted} ${synthetic ? "synthetic" : captured ? "accepted" : "unverified"} capture${accepted === 1 ? "" : "s"} · ${rejected} rejected · ${resolutionLabel}`;
    elements.captureTimeStatus.title = elements.captureTimeStatus.textContent;
    if (synthetic) {
      elements.sourceModeBadge.textContent = "SYNTHETIC";
      elements.sourceModeBadge.classList.remove("is-atlas");
      elements.sourceModeBadge.classList.add("is-synthetic");
      elements.provenanceTitle.textContent = "SYNTHETIC DEMO · NOT OBSERVED TERRAIN";
      elements.provenanceDetail.textContent = "Analytically generated imagery, thermal values, timestamps, and coordinates. No real sensor data, no real GPS data, and no operational or survey use.";
    } else if (captured) {
      elements.provenanceTitle.textContent = "CAPTURED EVIDENCE · TACTICAL ATLAS";
      elements.provenanceDetail.textContent = `Weighted projection of explicitly declared real sensor and navigation captures. ${freshnessLabel}. ${dynamicExcluded ? `${trackCount} transient object track${trackCount === 1 ? "" : "s"} from ${objectCount} observation${objectCount === 1 ? "" : "s"}; none baked into static terrain.` : "Dynamic-object exclusion is not declared."} No model-generated pixels and no survey-grade claim.`;
    } else {
      elements.sourceModeBadge.textContent = "UNVERIFIED";
      elements.sourceModeBadge.classList.remove("is-atlas");
      elements.sourceModeBadge.classList.add("is-unverified");
      elements.provenanceTitle.textContent = "UNVERIFIED DATA ORIGIN";
      elements.provenanceDetail.textContent = "This bundle does not explicitly declare both real sensor and real navigation inputs. It is displayed for inspection only and must not be treated as captured evidence.";
    }
    updatePresetReadout();
    updateAccessibilitySummary();
    updateNavigation();
  }

  function updatePresetReadout(baseKey = chooseBaseImage()?.key || "") {
    let readout = PRESETS[state.preset].readout;
    const viewNames = {
      fused: "FUSED",
      visible: "VISIBLE",
      thermal: "NORMALIZED THERMAL",
      semantic: "SEMANTIC",
      support: "SUPPORT",
      atlas_composite: "RGB + TIR MOSAIC",
      atlas_thermal: "THERMAL MOSAIC",
      atlas_support: "PROJECTION SUPPORT",
    };
    if (state.preset === "terrain") {
      readout = baseKey === "semantic"
        ? "SEMANTIC TERRAIN"
        : baseKey
          ? `TERRAIN · ${viewNames[baseKey] || "MEDIA"} FALLBACK`
          : "TERRAIN VIEW · UNAVAILABLE";
    }
    if (state.preset === "integrity") {
      readout = baseKey === "support"
        ? "FUSION SUPPORT"
        : baseKey
          ? `INTEGRITY · ${viewNames[baseKey] || "MEDIA"} FALLBACK`
          : "INTEGRITY VIEW · UNAVAILABLE";
    }
    if (state.preset === "atlas") {
      readout = state.atlasStatus === "available" && baseKey
        ? `TACTICAL ATLAS · ${viewNames[baseKey] || "MOSAIC"}`
        : state.atlasStatus === "loading"
          ? "TACTICAL ATLAS · LOADING"
          : "TACTICAL ATLAS · UNAVAILABLE";
    }
    elements.presetReadout.textContent = readout;
  }

  function updateAccessibilitySummary() {
    if (state.preset === "atlas") {
      if (state.atlasStatus !== "available" || !state.atlas?.available) {
        const status = state.atlasStatus === "loading"
          ? "The latest exported mission atlas is loading."
          : "No validated exported mission atlas is available; no map is being fabricated.";
        elements.canvas.setAttribute("aria-label", status);
        elements.sceneA11ySummary.textContent = status;
        return;
      }
      const dimensions = frameDimensions();
      const mission = cleanText(state.atlas.meta.mission_id, "unnamed mission");
      const originStatus = cleanText(state.atlas.meta.origin_status, "unverified");
      const originWarning = originStatus === "synthetic_demo"
        ? "Synthetic demonstration only; no real sensor or GPS data."
        : originStatus === "captured_evidence"
          ? "Explicitly declared real sensor and navigation evidence."
          : "Unverified data origin; do not treat as captured evidence.";
      const dynamicStatement = state.atlas.meta.dynamic_objects_excluded_from_static_atlas === true
        ? "Dynamic objects are not baked into terrain."
        : "Dynamic-object exclusion is not declared.";
      const description = `Atlas view of mission ${mission}, ${dimensions.width} by ${dimensions.height} cells. ${originWarning} ${atlasLayerLabel()}. North-up tactical 2.5-D pose projection; not survey-grade. ${dynamicStatement} No model-generated pixels are displayed.`;
      elements.canvas.setAttribute("aria-label", description);
      elements.sceneA11ySummary.textContent = description;
      return;
    }
    if (!state.frame) {
      elements.canvas.setAttribute("aria-label", "Multisensor scene has not loaded.");
      return;
    }
    const dimensions = frameDimensions();
    const sample = cleanText(state.frame.meta.sample_id, `frame ${state.index + 1}`);
    const count = state.frame.detections.length;
    const overlayState = state.labelsVisible ? "shown" : "hidden";
    const lensState = state.lensEnabled ? "enabled" : "disabled";
    const terrainClasses = state.frame.terrainClasses || [];
    const terrainSource = sourceKind(annotationSource()) === "ground-truth" ? "ground-truth " : "";
    const terrainDetails = state.preset === "terrain" && terrainClasses.length > 0
      ? ` ${terrainSource}terrain legend: ${terrainClasses.map((item) => `${item.label} ${terrainCoverageLabel(item.coverage)}`).join(", ")}.`
      : "";
    const registration = elements.registrationValue.textContent;
    const captureTime = elements.captureTimeStatus.textContent;
    const lensPrecision = state.lensEnabled ? " Thermal evidence is normalized and color-mapped." : "";
    const description = `${PRESETS[state.preset].name} view of ${sample}, ${dimensions.width} by ${dimensions.height} pixels. Dataset replay; ${captureTime.toLowerCase()}. Registration: ${registration}. ${count} supplied annotation${count === 1 ? "" : "s"} ${overlayState}. Evidence lens ${lensState}.${lensPrecision}${terrainDetails}`;
    elements.canvas.setAttribute("aria-label", description);
    elements.sceneA11ySummary.textContent = description;
  }

  function chooseBaseImage() {
    if (state.preset === "atlas") {
      if (state.atlasStatus !== "available") return null;
      const key = {
        composite: "rgb",
        thermal: "thermal",
        support: "support",
      }[state.atlasLayer];
      const image = state.atlasMedia[key];
      return image ? { key: `atlas_${state.atlasLayer}`, image } : null;
    }
    const media = state.media;
    const preferences = {
      navigate: ["fused", "visible", "thermal"],
      search: ["fused", "thermal", "visible"],
      terrain: ["semantic", "fused", "visible", "thermal"],
      integrity: ["support", "fused", "visible", "thermal"],
    }[state.preset];
    const key = preferences.find((candidate) => media[candidate]);
    return key ? { key, image: media[key] } : null;
  }

  function fitImage(sourceWidth, sourceHeight, targetWidth, targetHeight) {
    const scale = Math.min(targetWidth / sourceWidth, targetHeight / sourceHeight);
    const width = sourceWidth * scale;
    const height = sourceHeight * scale;
    return {
      x: (targetWidth - width) / 2,
      y: (targetHeight - height) / 2,
      width,
      height,
      scale,
    };
  }

  function scheduleRender() {
    if (state.renderQueued) return;
    state.renderQueued = true;
    requestAnimationFrame(() => {
      state.renderQueued = false;
      renderScene();
    });
  }

  function canvasSize() {
    const bounds = elements.canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(bounds.width));
    const height = Math.max(1, Math.round(bounds.height));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const pixelWidth = Math.round(width * dpr);
    const pixelHeight = Math.round(height * dpr);
    if (elements.canvas.width !== pixelWidth || elements.canvas.height !== pixelHeight) {
      elements.canvas.width = pixelWidth;
      elements.canvas.height = pixelHeight;
    }
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { width, height };
  }

  function drawEmptyScene(width, height) {
    context.fillStyle = "#030605";
    context.fillRect(0, 0, width, height);
    context.strokeStyle = "rgba(201, 245, 109, 0.08)";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(width / 2 - 18, height / 2);
    context.lineTo(width / 2 + 18, height / 2);
    context.moveTo(width / 2, height / 2 - 18);
    context.lineTo(width / 2, height / 2 + 18);
    context.stroke();
  }

  function normalizeDetectionBox(detection, sceneWidth, sceneHeight) {
    let x = finiteNumber(detection.x, NaN);
    let y = finiteNumber(detection.y, NaN);
    let width = finiteNumber(detection.w, NaN);
    let height = finiteNumber(detection.h, NaN);
    if (![x, y, width, height].every(Number.isFinite) || width <= 0 || height <= 0) return null;

    const normalized = Math.max(Math.abs(x), Math.abs(y), Math.abs(width), Math.abs(height)) <= 1.5;
    if (normalized) {
      x *= sceneWidth;
      y *= sceneHeight;
      width *= sceneWidth;
      height *= sceneHeight;
    }

    const left = clamp(x, 0, sceneWidth);
    const top = clamp(y, 0, sceneHeight);
    const right = clamp(x + width, 0, sceneWidth);
    const bottom = clamp(y + height, 0, sceneHeight);
    if (right <= left || bottom <= top) return null;
    return { x: left, y: top, width: right - left, height: bottom - top };
  }

  function annotationStyle(source) {
    const kind = sourceKind(source);
    if (kind === "ground-truth") return { kind, color: "#ffad73", fill: "rgba(47, 22, 15, 0.9)", tag: "GROUND TRUTH" };
    if (kind === "model") return { kind, color: "#65dbe5", fill: "rgba(8, 31, 34, 0.9)", tag: "MODEL OUTPUT" };
    return { kind, color: "#ffd36a", fill: "rgba(41, 33, 11, 0.9)", tag: "DECLARED" };
  }

  function drawCornerAccents(x, y, width, height, color, emphasized) {
    const corner = clamp(Math.min(width, height) * 0.22, 8, 19);
    context.strokeStyle = color;
    context.lineWidth = emphasized ? 3 : 2;
    context.beginPath();
    context.moveTo(x, y + corner); context.lineTo(x, y); context.lineTo(x + corner, y);
    context.moveTo(x + width - corner, y); context.lineTo(x + width, y); context.lineTo(x + width, y + corner);
    context.moveTo(x + width, y + height - corner); context.lineTo(x + width, y + height); context.lineTo(x + width - corner, y + height);
    context.moveTo(x + corner, y + height); context.lineTo(x, y + height); context.lineTo(x, y + height - corner);
    context.stroke();
  }

  function drawDetections(fit, sceneWidth, sceneHeight) {
    if (!state.labelsVisible) return;
    const detections = state.frame?.detections || [];
    const defaultSource = annotationSource();

    context.save();
    context.beginPath();
    context.rect(fit.x, fit.y, fit.width, fit.height);
    context.clip();

    detections.forEach((detection) => {
      const box = normalizeDetectionBox(detection, sceneWidth, sceneHeight);
      if (!box) return;
      const x = fit.x + (box.x / sceneWidth) * fit.width;
      const y = fit.y + (box.y / sceneHeight) * fit.height;
      const width = (box.width / sceneWidth) * fit.width;
      const height = (box.height / sceneHeight) * fit.height;
      const source = cleanText(detection.source, defaultSource);
      const style = annotationStyle(source);
      const emphasized = state.preset === "search";

      context.strokeStyle = style.color;
      context.lineWidth = emphasized ? 1.2 : 0.8;
      context.globalAlpha = emphasized ? 0.56 : 0.38;
      context.strokeRect(x + 0.5, y + 0.5, width - 1, height - 1);
      context.globalAlpha = 1;
      drawCornerAccents(x, y, width, height, style.color, emphasized);

      const label = cleanText(detection.label, "object");
      const confidence = formatConfidence(detection.confidence);
      const text = `${style.tag} · ${label}${confidence ? ` ${confidence}` : ""}`;
      context.font = `700 ${emphasized ? 10 : 9}px "Cascadia Mono", Consolas, monospace`;
      const textWidth = Math.ceil(context.measureText(text).width);
      const labelHeight = emphasized ? 22 : 20;
      const labelWidth = Math.min(textWidth + 13, Math.max(width, 78));
      const labelY = y >= labelHeight + fit.y ? y - labelHeight : y;
      context.fillStyle = style.fill;
      context.fillRect(x, labelY, labelWidth, labelHeight);
      context.fillStyle = style.color;
      context.textBaseline = "middle";
      context.fillText(text, x + 6, labelY + labelHeight / 2, Math.max(20, labelWidth - 10));
    });
    context.restore();
  }

  function lensImage(baseKey) {
    if (state.preset === "atlas") {
      if (baseKey === "atlas_composite") {
        return state.atlasMedia.thermal ? { key: "atlas_thermal", image: state.atlasMedia.thermal } : null;
      }
      return state.atlasMedia.rgb ? { key: "atlas_rgb", image: state.atlasMedia.rgb } : null;
    }
    if (state.preset === "integrity" || baseKey === "thermal") {
      return state.media.visible ? { key: "visible", image: state.media.visible } : null;
    }
    return state.media.thermal
      ? { key: "thermal", image: state.media.thermal }
      : state.media.visible
        ? { key: "visible", image: state.media.visible }
        : null;
  }

  function drawEvidenceLens(fit, baseKey, canvasWidth, canvasHeight) {
    if (!state.lensEnabled || !state.pointer.inside) return;
    const lens = lensImage(baseKey);
    if (!lens) return;
    const { x, y } = state.pointer;
    if (x < fit.x || y < fit.y || x > fit.x + fit.width || y > fit.y + fit.height) return;

    const radius = clamp(Math.min(canvasWidth, canvasHeight) * 0.13, 62, 108);
    const color = lens.key === "thermal" || lens.key === "atlas_thermal" ? "#ffad73" : "#65dbe5";
    context.save();
    context.beginPath();
    context.arc(x, y, radius, 0, Math.PI * 2);
    context.clip();
    context.fillStyle = "#020403";
    context.fillRect(x - radius, y - radius, radius * 2, radius * 2);
    context.drawImage(lens.image, fit.x, fit.y, fit.width, fit.height);

    context.globalAlpha = 0.22;
    context.strokeStyle = color;
    context.lineWidth = 0.8;
    const spacing = 14;
    for (let offset = -radius; offset <= radius; offset += spacing) {
      context.beginPath();
      context.moveTo(x - radius, y + offset);
      context.lineTo(x + radius, y + offset);
      context.stroke();
    }
    context.restore();

    context.save();
    context.strokeStyle = "rgba(0, 0, 0, 0.82)";
    context.lineWidth = 6;
    context.beginPath();
    context.arc(x, y, radius + 1, 0, Math.PI * 2);
    context.stroke();
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.beginPath();
    context.arc(x, y, radius, 0, Math.PI * 2);
    context.stroke();

    context.beginPath();
    context.moveTo(x - 11, y); context.lineTo(x + 11, y);
    context.moveTo(x, y - 11); context.lineTo(x, y + 11);
    context.stroke();
    context.fillStyle = color;
    context.beginPath();
    context.arc(x, y, 2.4, 0, Math.PI * 2);
    context.fill();

    const label = lens.key === "thermal"
      ? "NORMALIZED THERMAL · COORDINATE LOCK"
      : lens.key === "atlas_thermal"
        ? "NORMALIZED THERMAL ATLAS · COORDINATE LOCK"
        : lens.key === "atlas_rgb"
          ? "RGB ATLAS · COORDINATE LOCK"
          : "VISIBLE IMAGE · COORDINATE LOCK";
    context.font = '700 9px "Cascadia Mono", Consolas, monospace';
    const labelWidth = context.measureText(label).width + 14;
    const labelX = clamp(x - labelWidth / 2, 8, canvasWidth - labelWidth - 8);
    const preferredY = y - radius - 27;
    const labelY = preferredY >= 8 ? preferredY : y + radius + 8;
    context.fillStyle = "rgba(3, 7, 5, 0.9)";
    context.fillRect(labelX, labelY, labelWidth, 19);
    context.fillStyle = color;
    context.textBaseline = "middle";
    context.fillText(label, labelX + 7, labelY + 9.5);
    context.restore();
  }

  function drawAtlasObjects(fit) {
    if (state.preset !== "atlas" || !state.labelsVisible) return;
    const objects = Array.isArray(state.atlas?.objects) ? state.atlas.objects : [];
    const grid = state.atlas?.meta?.grid || {};
    const eastMin = Number(grid.east_min_m);
    const eastMax = Number(grid.east_max_m);
    const northMin = Number(grid.north_min_m);
    const northMax = Number(grid.north_max_m);
    const eastSpan = eastMax - eastMin;
    const northSpan = northMax - northMin;
    if (!objects.length || !(eastSpan > 0) || !(northSpan > 0)) return;

    // TAI nanoseconds exceed Number's exact range. Sort with BigInt and retain
    // input order only as a deterministic tie-breaker or timestamp fallback.
    const visibleObjects = objects
      .map((item, inputIndex) => ({ item, inputIndex }))
      .sort((first, second) => (
        compareAtlasObservationTime(first.item, second.item)
        || first.inputIndex - second.inputIndex
      ))
      .slice(-500)
      .map(({ item }) => item);
    const pointFor = (item) => ({
      x: fit.x + ((item.east_m - eastMin) / eastSpan) * fit.width,
      y: fit.y + ((northMax - item.north_m) / northSpan) * fit.height,
    });
    const observationsByTrack = new Map();
    visibleObjects.forEach((item) => {
      if (!observationsByTrack.has(item.object_id)) observationsByTrack.set(item.object_id, []);
      observationsByTrack.get(item.object_id).push(item);
    });
    const latestByTrack = new Map();
    observationsByTrack.forEach((track, objectId) => {
      track.sort(compareAtlasObservationTime);
      latestByTrack.set(objectId, track[track.length - 1]);
    });

    context.save();
    context.beginPath();
    context.rect(fit.x, fit.y, fit.width, fit.height);
    context.clip();
    const trailColor = state.atlas?.meta?.origin_status === "synthetic_demo" ? "#ffd36a" : "#ff6d8a";
    observationsByTrack.forEach((track) => {
      if (track.length < 2) return;
      context.beginPath();
      let previous = null;
      let connected = false;
      track.forEach((item, index) => {
        const point = pointFor(item);
        if (
          index === 0
          || previous === null
          || !atlasTrackPointsAreContinuous(previous, item)
        ) {
          context.moveTo(point.x, point.y);
        } else {
          context.lineTo(point.x, point.y);
          connected = true;
        }
        previous = item;
      });
      if (!connected) return;
      context.strokeStyle = trailColor;
      context.lineWidth = 1.4;
      context.globalAlpha = 0.42;
      context.stroke();
    });
    context.globalAlpha = 1;
    visibleObjects.forEach((item) => {
      const { x, y } = pointFor(item);
      if (x < fit.x || x > fit.x + fit.width || y < fit.y || y > fit.y + fit.height) return;
      const uncertainty = Number(item.horizontal_uncertainty_m);
      const radius = Number.isFinite(uncertainty)
        ? clamp(uncertainty * 0.5 * (fit.width / eastSpan + fit.height / northSpan), 3, 48)
        : 5;
      const color = trailColor;
      context.strokeStyle = color;
      context.fillStyle = "rgba(2, 6, 5, 0.72)";
      context.lineWidth = 1;
      context.globalAlpha = 0.48;
      context.beginPath();
      context.arc(x, y, radius, 0, Math.PI * 2);
      context.stroke();
      context.globalAlpha = 1;
      context.beginPath();
      context.moveTo(x, y - 6);
      context.lineTo(x + 6, y);
      context.lineTo(x, y + 6);
      context.lineTo(x - 6, y);
      context.closePath();
      context.fill();
      context.stroke();
      context.fillStyle = color;
      context.beginPath();
      context.arc(x, y, 1.8, 0, Math.PI * 2);
      context.fill();
      if (latestByTrack.get(item.object_id) === item) {
        const confidence = Number.isFinite(item.confidence) ? ` ${Math.round(item.confidence * 100)}%` : "";
        const label = `${cleanText(item.label, "object").toUpperCase()}${confidence}`;
        context.font = '700 9px "Cascadia Mono", Consolas, monospace';
        const labelWidth = Math.min(150, context.measureText(label).width + 10);
        context.fillStyle = "rgba(2, 6, 5, 0.86)";
        context.fillRect(x + 8, y - 9, labelWidth, 17);
        context.fillStyle = color;
        context.textBaseline = "middle";
        context.fillText(label, x + 13, y - 0.5, labelWidth - 8);
      }
    });
    context.restore();
  }

  function drawAtlasOriginWarning(fit) {
    if (state.preset !== "atlas") return;
    const originStatus = cleanText(state.atlas?.meta?.origin_status, "unverified");
    const synthetic = originStatus === "synthetic_demo";
    const freshness = atlasFreshness(state.atlas?.meta).status;
    const stale = freshness === "stale";
    const future = freshness === "future";
    const snapshot = freshness === "snapshot" || freshness === "unverified";
    if (originStatus === "captured_evidence" && freshness === "fresh") return;
    const label = synthetic
      ? "SYNTHETIC DEMO · NO REAL SENSOR OR GPS DATA"
      : originStatus !== "captured_evidence"
        ? "UNVERIFIED DATA ORIGIN · INSPECTION ONLY"
        : future
          ? "FUTURE-DATED ATLAS · CLOCKS INVALID"
        : stale
          ? "STALE ATLAS · NOT CURRENT SENSOR STATE"
          : snapshot
            ? "MISSION SNAPSHOT · NOT A LIVE FEED"
            : "ATLAS FRESHNESS UNVERIFIED";
    const color = synthetic ? "#ffd36a" : stale || future ? "#ff6d8a" : "#ffad73";

    context.save();
    context.beginPath();
    context.rect(fit.x, fit.y, fit.width, fit.height);
    context.clip();
    context.translate(fit.x + fit.width / 2, fit.y + fit.height / 2);
    context.rotate(-Math.PI / 7);
    context.font = '800 18px "Cascadia Mono", Consolas, monospace';
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillStyle = synthetic ? "rgba(255, 211, 106, 0.19)" : stale || future ? "rgba(255, 78, 112, 0.2)" : "rgba(255, 173, 115, 0.2)";
    const rowStep = 92;
    for (let y = -fit.height; y <= fit.height; y += rowStep) {
      context.fillText(label, 0, y);
    }
    context.restore();

    context.save();
    const bannerHeight = 29;
    context.fillStyle = "rgba(4, 8, 6, 0.92)";
    context.fillRect(fit.x, fit.y + 8, fit.width, bannerHeight);
    context.strokeStyle = color;
    context.lineWidth = 1;
    context.strokeRect(fit.x + 0.5, fit.y + 8.5, fit.width - 1, bannerHeight - 1);
    context.fillStyle = color;
    context.font = '800 10px "Cascadia Mono", Consolas, monospace';
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(label, fit.x + fit.width / 2, fit.y + 8 + bannerHeight / 2, Math.max(20, fit.width - 16));
    context.restore();
  }

  function renderScene() {
    const { width: canvasWidth, height: canvasHeight } = canvasSize();
    drawEmptyScene(canvasWidth, canvasHeight);
    const base = chooseBaseImage();
    const activeEvidence = state.preset === "atlas"
      ? state.atlasStatus === "available" && state.atlas?.available
      : Boolean(state.frame);
    if (!base || !activeEvidence) {
      state.lastFit = null;
      updatePresetReadout("");
      return;
    }

    const dimensions = frameDimensions();
    const fit = fitImage(dimensions.width, dimensions.height, canvasWidth, canvasHeight);
    state.lastFit = fit;
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(base.image, fit.x, fit.y, fit.width, fit.height);

    if (state.preset === "atlas" && state.atlasLayer === "composite" && state.atlasMedia.thermal) {
      context.save();
      context.globalCompositeOperation = "screen";
      context.globalAlpha = (state.thermalGain / 100) * 0.52;
      context.drawImage(state.atlasMedia.thermal, fit.x, fit.y, fit.width, fit.height);
      context.restore();
    }

    if (state.preset === "search") {
      context.fillStyle = "rgba(4, 7, 5, 0.08)";
      context.fillRect(fit.x, fit.y, fit.width, fit.height);
    }

    if (state.preset !== "atlas") drawDetections(fit, dimensions.width, dimensions.height);
    else drawAtlasObjects(fit);
    drawAtlasOriginWarning(fit);
    drawEvidenceLens(fit, base.key, canvasWidth, canvasHeight);
    updatePresetReadout(base.key);
  }

  function setPreset(preset) {
    if (!PRESETS[preset]) return;
    state.preset = preset;
    elements.presetButtons.forEach((button) => {
      const active = button.dataset.preset === preset;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    if (preset === "atlas") {
      updateAtlasInterface();
      if (state.atlasStatus === "idle") loadAtlas();
      else pollAtlasStatus();
    } else {
      elements.atlasLayerPanel.hidden = true;
      if (state.frame) updateFrameInterface();
      else {
        updatePresetReadout();
        updateTerrainLegend();
        updateAccessibilitySummary();
      }
    }
    syncSceneState();
    updateNavigation();
    scheduleRender();
  }

  function navigateTo(index) {
    if (state.preset === "atlas" || state.loading || state.count <= 0) return;
    const nextIndex = clamp(Math.trunc(index), 0, state.count - 1);
    if (nextIndex === state.index) {
      updateNavigation();
      return;
    }
    state.index = nextIndex;
    updateNavigation();
    loadFrame();
  }

  function updateThermalGain(value, fetchFrame = false) {
    state.thermalGain = clamp(Math.round(finiteNumber(value, state.thermalGain)), 0, 100);
    elements.thermalGain.value = String(state.thermalGain);
    elements.thermalOutput.value = `${state.thermalGain}%`;
    elements.thermalOutput.textContent = `${state.thermalGain}%`;
    window.clearTimeout(state.thermalTimer);
    if (state.preset === "atlas") {
      if (state.atlasStatus === "available") {
        elements.sceneDescription.textContent = `Atlas / ${cleanText(state.atlas?.meta?.mission_id, "unnamed mission")} / ${atlasLayerLabel()}`;
      }
      scheduleRender();
      return;
    }
    if (!state.frame) return;
    if (fetchFrame) {
      loadFrame();
    } else {
      state.thermalTimer = window.setTimeout(loadFrame, 220);
    }
  }

  function setAutomaticControl(enabled, reload = true) {
    state.autoFusion = Boolean(enabled);
    elements.autoFusionToggle.checked = state.autoFusion;
    elements.thermalGain.disabled = state.preset !== "atlas" && state.autoFusion;
    updateAIAdvisor();
    if (reload && state.preset !== "atlas" && state.datasetId && state.splitId) {
      loadFrame();
    }
  }

  function setToggle(input, value) {
    input.checked = Boolean(value);
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function interactiveTarget(target) {
    return target instanceof HTMLElement && Boolean(target.closest("input, select, button, summary, a, textarea"));
  }

  function handleKeyboard(event) {
    if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey || interactiveTarget(event.target)) return;
    const key = event.key.toLowerCase();
    const presetKeys = { "1": "navigate", "2": "search", "3": "terrain", "4": "integrity", "5": "atlas" };

    if (event.key === "ArrowLeft") {
      event.preventDefault();
      navigateTo(state.index - 1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      navigateTo(state.index + 1);
    } else if (event.key === "Home") {
      event.preventDefault();
      navigateTo(0);
    } else if (event.key === "End") {
      event.preventDefault();
      navigateTo(state.count - 1);
    } else if (presetKeys[key]) {
      event.preventDefault();
      setPreset(presetKeys[key]);
    } else if (key === "l") {
      event.preventDefault();
      setToggle(elements.lensToggle, !state.lensEnabled);
    } else if (key === "g") {
      event.preventDefault();
      setToggle(elements.labelsToggle, !state.labelsVisible);
    } else if (event.key === "[") {
      event.preventDefault();
      updateThermalGain(state.thermalGain - 5);
    } else if (event.key === "]") {
      event.preventDefault();
      updateThermalGain(state.thermalGain + 5);
    }
  }

  function pointerCoordinates(event) {
    const bounds = elements.canvas.getBoundingClientRect();
    return {
      x: clamp(event.clientX - bounds.left, 0, bounds.width),
      y: clamp(event.clientY - bounds.top, 0, bounds.height),
    };
  }

  function updateCoordinateReadout() {
    const fit = state.lastFit;
    if (!fit || !state.pointer.inside) {
      elements.coordinateReadout.textContent = "X —   Y —";
      return;
    }
    const dimensions = frameDimensions();
    const inImage = state.pointer.x >= fit.x && state.pointer.x <= fit.x + fit.width
      && state.pointer.y >= fit.y && state.pointer.y <= fit.y + fit.height;
    if (!inImage) {
      elements.coordinateReadout.textContent = "OUTSIDE FRAME";
      return;
    }
    const x = Math.round(((state.pointer.x - fit.x) / fit.width) * (dimensions.width - 1));
    const y = Math.round(((state.pointer.y - fit.y) / fit.height) * (dimensions.height - 1));
    if (state.preset === "atlas") {
      const grid = state.atlas?.meta?.grid || {};
      const eastMin = Number(grid.east_min_m);
      const northMax = Number(grid.north_max_m);
      const resolution = Number(grid.resolution_m);
      if ([eastMin, northMax, resolution].every(Number.isFinite) && resolution > 0) {
        const east = eastMin + (x + 0.5) * resolution;
        const north = northMax - (y + 0.5) * resolution;
        elements.coordinateReadout.textContent = `E ${east.toFixed(1)} m   N ${north.toFixed(1)} m`;
      } else {
        elements.coordinateReadout.textContent = `COL ${x.toString().padStart(4, "0")}   ROW ${y.toString().padStart(4, "0")}`;
      }
      return;
    }
    elements.coordinateReadout.textContent = `X ${x.toString().padStart(4, "0")}   Y ${y.toString().padStart(4, "0")}`;
  }

  function handlePointerMove(event) {
    state.pointer = { inside: true, ...pointerCoordinates(event) };
    elements.canvas.classList.toggle("is-lens-active", state.lensEnabled);
    updateCoordinateReadout();
    scheduleRender();
  }

  elements.datasetSelect.addEventListener("change", () => {
    state.datasetId = elements.datasetSelect.value;
    const dataset = selectedDataset();
    state.splitId = dataset?.splits[0]?.id || "";
    state.index = 0;
    state.count = dataset?.splits[0]?.count || 0;
    populateSplitSelect();
    updateNavigation();
    loadFrame();
  });

  elements.splitSelect.addEventListener("change", () => {
    state.splitId = elements.splitSelect.value;
    state.index = 0;
    state.count = selectedSplit()?.count || 0;
    updateNavigation();
    loadFrame();
  });

  elements.previousFrame.addEventListener("click", () => navigateTo(state.index - 1));
  elements.nextFrame.addEventListener("click", () => navigateTo(state.index + 1));
  elements.frameIndex.addEventListener("change", () => navigateTo(finiteNumber(elements.frameIndex.value, 1) - 1));
  elements.frameIndex.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      elements.frameIndex.blur();
    }
  });

  elements.presetButtons.forEach((button) => {
    button.addEventListener("click", () => {
      if (state.autoFusion && button.dataset.preset !== "atlas") {
        setAutomaticControl(false, false);
      }
      setPreset(button.dataset.preset);
    });
  });

  elements.atlasLayerButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const layer = button.dataset.atlasLayer;
      if (!["composite", "thermal", "support"].includes(layer)) return;
      state.atlasLayer = layer;
      elements.atlasLayerButtons.forEach((candidate) => {
        const active = candidate.dataset.atlasLayer === layer;
        candidate.classList.toggle("is-active", active);
        candidate.setAttribute("aria-pressed", String(active));
      });
      if (state.preset === "atlas") updateAtlasInterface();
      scheduleRender();
    });
  });

  elements.thermalGain.addEventListener("input", () => updateThermalGain(elements.thermalGain.value));
  elements.thermalGain.addEventListener("change", () => updateThermalGain(elements.thermalGain.value, true));
  elements.autoFusionToggle.addEventListener("change", () => {
    setAutomaticControl(elements.autoFusionToggle.checked);
  });

  elements.labelsToggle.addEventListener("change", () => {
    state.labelsVisible = elements.labelsToggle.checked;
    updateAccessibilitySummary();
    scheduleRender();
  });

  elements.lensToggle.addEventListener("change", () => {
    state.lensEnabled = elements.lensToggle.checked;
    elements.canvas.classList.toggle("is-lens-active", state.lensEnabled && state.pointer.inside);
    updateAccessibilitySummary();
    scheduleRender();
  });

  elements.retryButton.addEventListener("click", () => {
    if (state.catalogFailed) loadCatalog();
    else loadFrame();
  });

  elements.atlasRetryButton.addEventListener("click", () => loadAtlas(true));

  elements.canvas.addEventListener("pointerenter", handlePointerMove);
  elements.canvas.addEventListener("pointermove", handlePointerMove);
  elements.canvas.addEventListener("pointerdown", (event) => {
    if (event.pointerType !== "mouse") elements.canvas.setPointerCapture?.(event.pointerId);
    handlePointerMove(event);
  });
  elements.canvas.addEventListener("pointerleave", () => {
    state.pointer.inside = false;
    elements.canvas.classList.remove("is-lens-active");
    updateCoordinateReadout();
    scheduleRender();
  });
  elements.canvas.addEventListener("pointerup", (event) => {
    if (event.pointerType === "mouse") return;
    state.pointer.inside = false;
    elements.canvas.classList.remove("is-lens-active");
    updateCoordinateReadout();
    scheduleRender();
  });

  document.addEventListener("keydown", handleKeyboard);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && state.preset === "atlas") {
      pollAtlasStatus();
    }
  });
  window.addEventListener("resize", scheduleRender, { passive: true });
  new ResizeObserver(scheduleRender).observe(elements.sceneStage);

  window.setInterval(() => {
    if (state.preset === "atlas") pollAtlasStatus();
  }, ATLAS_STATUS_POLL_MS);
  window.setInterval(() => {
    if (state.preset !== "atlas" || state.atlasStatus !== "available") return;
    updateAtlasInterface();
    scheduleRender();
  }, ATLAS_FRESHNESS_TICK_MS);

  updateThermalGain(state.thermalGain);
  updateAIAdvisor();
  updateNavigation();
  loadCatalog();
})();
