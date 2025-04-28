// Полный изменённый клиентский скрипт для работы с графиками и данными от backend

// Глобальные константы и переменные
const ALLOWED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT"];
let mainChart, volumeChart, candleSeries, predictedCandleSeries;
let livePriceLine = null, forecastPriceLine = null, predictPriceLine = null, lastForecastTime = null;
let currentSymbol = "BTCUSDT", currentInterval = "1h", currentExchange = "binance";
let latestCandles = [];
let liveCandle = null, lastUpdatedTime = 0, previousClose = null;
let maSeries, bollingerMiddleSeries, bollingerUpperSeries, bollingerLowerSeries;
let volumeSeries, volumeMASeries, rsiSeries, macdSeries, macdSignalSeries, atrSeries, stochasticKSeries, stochasticDSeries;
let drawingStartPoint = null;
let drawingTempRect = null, drawingTempSeries = null, drawingTempPriceLine = null, drawingTempVerticalLine = null;
let drawnTrendlines = [], drawnFibLines = [], drawnVerticalLines = [], drawnRectangles = [];
let currentDrawingTool = "none"; // Возможные значения: "trendline", "fibonacci", "horizontal", "vertical", "rectangle", "none"

// Функция преобразования интервала в секунды
function getIntervalSeconds(interval) {
  if (interval.endsWith("m")) return parseInt(interval) * 60;
  if (interval.endsWith("h")) return parseInt(interval) * 3600;
  if (interval.endsWith("d")) return parseInt(interval) * 86400;
  return 3600;
}

// Функция для расчёта "видимого" диапазона цены на основе latestCandles
function getVisiblePriceRange() {
  if (!latestCandles || latestCandles.length === 0) {
    return { min: 0, max: 0 };
  }
  const lows = latestCandles.map(c => c.low);
  const highs = latestCandles.map(c => c.high);
  return {
    min: Math.min(...lows),
    max: Math.max(...highs)
  };
}

// Вспомогательная функция для преобразования координаты Y в цену
function coordinateToPrice(y) {
  const range = getVisiblePriceRange();
  const chartHeight = document.getElementById("main-chart").clientHeight;
  // Верхняя граница графика соответствует range.max, нижняя – range.min
  return range.max - (range.max - range.min) * (y / chartHeight);
}

// Инициализация графиков с помощью LightweightCharts
function initCharts() {
  mainChart = LightweightCharts.createChart(document.getElementById("main-chart"), {
    layout: { background: { color: "#0e1117" }, textColor: "#c8cdd2" },
    grid: {
      vertLines: { color: "#1f2937", visible: document.getElementById("toggleGrid").checked },
      horzLines: { color: "#1f2937", visible: document.getElementById("toggleGrid").checked }
    },
    timeScale: { borderColor: "#2B2B43", timeVisible: true },
    crosshair: {
      mode: document.getElementById("toggleCrosshair").checked
        ? LightweightCharts.CrosshairMode.Normal
        : LightweightCharts.CrosshairMode.Disabled
    },
    rightPriceScale: { borderColor: "#2B2B43" }
  });

  volumeChart = LightweightCharts.createChart(document.getElementById("volume-chart"), {
    layout: { background: { color: "#0e1117" }, textColor: "#c8cdd2" },
    width: document.getElementById("volume-chart").clientWidth,
    height: 150,
    timeScale: { borderColor: "#2B2B43", timeVisible: true },
    rightPriceScale: { visible: true, borderColor: "#2B2B43" },
    grid: {
      vertLines: { color: "#1f2937", visible: document.getElementById("toggleGrid").checked },
      horzLines: { visible: false }
    },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal }
  });

  candleSeries = mainChart.addCandlestickSeries({
    upColor: "#4caf50", downColor: "#e53935",
    borderVisible: false, wickUpColor: "#4caf50", wickDownColor: "#e53935"
  });
  predictedCandleSeries = mainChart.addCandlestickSeries({
    upColor: "rgba(76,175,80,0.4)", downColor: "rgba(229,57,53,0.4)",
    borderVisible: true, wickUpColor: "rgba(76,175,80,0.4)", wickDownColor: "rgba(229,57,53,0.4)"
  });

  volumeSeries = volumeChart.addHistogramSeries({
    color: "#26a69a", priceFormat: { type: "volume" },
    priceScaleId: "", scaleMargins: { top: 0.1, bottom: 0 }
  });
  volumeMASeries = volumeChart.addLineSeries({ color: "#ffab00", lineWidth: 2 });

  rsiSeries = volumeChart.addLineSeries({ color: "#ff4081", lineWidth: 2 });
  macdSeries = volumeChart.addLineSeries({ color: "#00bcd4", lineWidth: 2 });
  macdSignalSeries = volumeChart.addLineSeries({ color: "#ff9800", lineWidth: 2 });
  atrSeries = volumeChart.addLineSeries({ color: "#8e44ad", lineWidth: 2 });
  stochasticKSeries = volumeChart.addLineSeries({ color: "#3498db", lineWidth: 2 });
  stochasticDSeries = volumeChart.addLineSeries({ color: "#e74c3c", lineWidth: 2 });

  maSeries = mainChart.addLineSeries({ color: "#ffab00", lineWidth: 2 });
  bollingerMiddleSeries = mainChart.addLineSeries({ color: "#90caf9", lineWidth: 1 });
  bollingerUpperSeries = mainChart.addLineSeries({ color: "#90caf9", lineWidth: 1 });
  bollingerLowerSeries = mainChart.addLineSeries({ color: "#90caf9", lineWidth: 1 });

  mainChart.subscribeCrosshairMove(param => {
    const tooltip = document.getElementById("chart-tooltip");
    if (!param || !param.point || !param.seriesPrices || !param.seriesPrices.size) {
      tooltip.style.display = "none";
      return;
    }
    const price = param.seriesPrices.get(candleSeries);
    if (!price) { tooltip.style.display = "none"; return; }
    const dateStr = new Date(param.time * 1000).toLocaleString();
    tooltip.style.display = "block";
    tooltip.innerHTML = `<strong>${dateStr}</strong><br>Цена: ${price.toFixed(2)}`;
    tooltip.style.left = (param.point.x + 20) + "px";
    tooltip.style.top = param.point.y + "px";
  });
}

// Инициализация инструментов рисования – регистрация событий на оверлее
function initDrawingTools() {
  const overlay = document.getElementById("drawing-overlay");
  overlay.addEventListener("mousedown", onOverlayMouseDown);
  overlay.addEventListener("mousemove", onOverlayMouseMove);
  overlay.addEventListener("mouseup", onOverlayMouseUp);
}

// Обработчики событий для рисования
function onOverlayMouseDown(event) {
  if (currentDrawingTool === "none") return;
  const rect = event.currentTarget.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const time = mainChart.timeScale().coordinateToTime(x);
  const price = coordinateToPrice(y);
  if (time == null || price == null) return;
  drawingStartPoint = { x, y, time, price };

  if (currentDrawingTool === "rectangle") {
    drawingTempRect = document.createElement("div");
    drawingTempRect.style.cssText = `
      position:absolute; border:2px dashed #fff;
      background:rgba(255,255,255,0.1); left:${x}px; top:${y}px;
      width:0; height:0;
    `;
    document.getElementById("drawing-overlay").appendChild(drawingTempRect);
  }
}

function onOverlayMouseMove(event) {
  if (!drawingStartPoint || currentDrawingTool === "none") return;
  const rect = event.currentTarget.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const currentTime = mainChart.timeScale().coordinateToTime(x);
  const currentPrice = coordinateToPrice(y);
  if (currentTime == null || currentPrice == null) return;
  switch (currentDrawingTool) {
    case "trendline":
    case "fibonacci":
      if (!drawingTempSeries) {
        drawingTempSeries = mainChart.addLineSeries({ color: "#fff", lineWidth: 2 });
      }
      drawingTempSeries.setData([
        { time: drawingStartPoint.time, value: drawingStartPoint.price },
        { time: currentTime, value: currentPrice }
      ]);
      break;
    case "horizontal":
      if (drawingTempPriceLine) mainChart.removePriceLine(drawingTempPriceLine);
      drawingTempPriceLine = candleSeries.createPriceLine({
        price: currentPrice,
        color: "#fff",
        lineWidth: 2,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        axisLabelVisible: false
      });
      break;
    case "vertical":
      if (!drawingTempVerticalLine) {
        drawingTempVerticalLine = document.createElement("div");
        drawingTempVerticalLine.style.cssText = `
          position:absolute; width:2px; height:100%; top:0; background:#fff;
        `;
        document.getElementById("drawing-overlay").appendChild(drawingTempVerticalLine);
      }
      drawingTempVerticalLine.style.left = x + "px";
      break;
    case "rectangle":
      if (drawingTempRect) {
        const w = x - drawingStartPoint.x;
        const h = y - drawingStartPoint.y;
        drawingTempRect.style.left = (w < 0 ? x : drawingStartPoint.x) + "px";
        drawingTempRect.style.top = (h < 0 ? y : drawingStartPoint.y) + "px";
        drawingTempRect.style.width = Math.abs(w) + "px";
        drawingTempRect.style.height = Math.abs(h) + "px";
      }
      break;
  }
}

function onOverlayMouseUp(event) {
  if (!drawingStartPoint || currentDrawingTool === "none") return;
  const rect = event.currentTarget.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const endTime = mainChart.timeScale().coordinateToTime(x);
  const endPrice = coordinateToPrice(y);
  if (endTime == null || endPrice == null) return;
  const drawingEndPoint = { x, y, time: endTime, price: endPrice };

  switch (currentDrawingTool) {
    case "trendline":
      drawTrendline(drawingStartPoint, drawingEndPoint);
      break;
    case "fibonacci":
      drawFibRetracement(drawingStartPoint, drawingEndPoint);
      break;
    case "horizontal":
      finalizeHorizontalLine(endPrice);
      break;
    case "vertical":
      finalizeVerticalLine(x);
      break;
    case "rectangle":
      finalizeRectangle();
      break;
  }
  if (drawingTempSeries) { mainChart.removeSeries(drawingTempSeries); drawingTempSeries = null; }
  if (drawingTempPriceLine) { candleSeries.removePriceLine(drawingTempPriceLine); drawingTempPriceLine = null; }
  if (drawingTempVerticalLine) { drawingTempVerticalLine.remove(); drawingTempVerticalLine = null; }
  drawingStartPoint = null;
}

// Функции завершения рисования
function drawTrendline(start, end) {
  const trendlineSeries = mainChart.addLineSeries({ color: "#ffffff", lineWidth: 2 });
  trendlineSeries.setData([
    { time: start.time, value: start.price },
    { time: end.time, value: end.price }
  ]);
  drawnTrendlines.push(trendlineSeries);
}

function drawFibRetracement(start, end) {
  const fibLevels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
  const fibLines = [];
  const priceDiff = end.price - start.price;
  fibLevels.forEach(level => {
    const levelPrice = start.price + priceDiff * level;
    const fibLine = candleSeries.createPriceLine({
      price: levelPrice,
      color: "rgba(255, 255, 255, 0.5)",
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dotted,
      axisLabelVisible: true,
      title: `Fib ${Math.round(level * 100)}%`
    });
    fibLines.push(fibLine);
  });
  drawnFibLines.push(fibLines);
}

function finalizeHorizontalLine(price) {
  candleSeries.createPriceLine({
    price: price,
    color: "#ffffff",
    lineWidth: 2,
    lineStyle: LightweightCharts.LineStyle.Solid,
    axisLabelVisible: true,
    title: "H-Line"
  });
}

function finalizeVerticalLine(x) {
  const overlay = document.getElementById("drawing-overlay");
  const vLine = document.createElement("div");
  vLine.style.position = "absolute";
  vLine.style.width = "2px";
  vLine.style.height = "100%";
  vLine.style.top = "0";
  vLine.style.left = x + "px";
  vLine.style.background = "#ffffff";
  overlay.appendChild(vLine);
  drawnVerticalLines.push(vLine);
}

function finalizeRectangle() {
  if (drawingTempRect) {
    drawnRectangles.push(drawingTempRect);
    drawingTempRect = null;
  }
}

// Обновление линии с текущей ценой
function updateLivePriceLine(price, open) {
  if (livePriceLine) {
    candleSeries.removePriceLine(livePriceLine);
    livePriceLine = null;
  }
  const color = price >= open ? "#4caf50" : "#e53935";
  livePriceLine = candleSeries.createPriceLine({
    price: price,
    color: color,
    lineWidth: 2,
    lineStyle: LightweightCharts.LineStyle.Solid,
    axisLabelVisible: true,
    title: "L"
  });
}

// Обновление информации о рынке (цена, открытие, закрытие, объем)
function updateMarketInfo(candle) {
  const priceEl = document.getElementById("price-info");
  const openEl = document.getElementById("open-info");
  const closeEl = document.getElementById("close-info");
  const volumeEl = document.getElementById("volume-info");
  if (previousClose !== null) {
    priceEl.style.color = candle.close > previousClose ? "#4caf50" : (candle.close < previousClose ? "#e53935" : "#e0e0e0");
  }
  previousClose = candle.close;
  priceEl.innerText = `Цена: ${candle.close.toFixed(2)}`;
  openEl.innerText = `Открытие: ${candle.open.toFixed(2)}`;
  closeEl.innerText = `Закрытие: ${candle.close.toFixed(2)}`;
  volumeEl.innerText = `Объем: ${candle.volume.toFixed(2)}`;
}

// Обновление наложенных индикаторов (MA, Bollinger Bands)
function updateOverlays(candles) {
  const period = 10;
  if (candles.length < period) return;
  let maData = [], bollMiddle = [], bollUpper = [], bollLower = [];
  for (let i = period - 1; i < candles.length; i++) {
    let sum = 0;
    for (let j = i - period + 1; j <= i; j++) { sum += candles[j].close; }
    const ma = sum / period;
    let sumSq = 0;
    for (let j = i - period + 1; j <= i; j++) { sumSq += Math.pow(candles[j].close - ma, 2); }
    const std = Math.sqrt(sumSq / period);
    maData.push({ time: candles[i].time, value: ma });
    bollMiddle.push({ time: candles[i].time, value: ma });
    bollUpper.push({ time: candles[i].time, value: ma + 2 * std });
    bollLower.push({ time: candles[i].time, value: ma - 2 * std });
  }
  if (document.getElementById("toggleMA").checked) {
    maSeries.setData(maData);
  } else {
    maSeries.setData([]);
  }
  if (document.getElementById("toggleBB").checked) {
    bollingerMiddleSeries.setData(bollMiddle);
    bollingerUpperSeries.setData(bollUpper);
    bollingerLowerSeries.setData(bollLower);
  } else {
    bollingerMiddleSeries.setData([]);
    bollingerUpperSeries.setData([]);
    bollingerLowerSeries.setData([]);
  }
}

// Функция расчета RSI
function calculateRSI(candles, period = 14) {
  if (candles.length < period + 1) return [];
  let rsi = [], gains = 0, losses = 0;
  for (let i = 1; i <= period; i++) {
    const change = candles[i].close - candles[i - 1].close;
    change > 0 ? gains += change : losses -= change;
  }
  let avgGain = gains / period, avgLoss = losses / period;
  let rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
  let rsiValue = 100 - (100 / (1 + rs));
  rsi.push({ time: candles[period].time, value: rsiValue });
  for (let i = period + 1; i < candles.length; i++) {
    const change = candles[i].close - candles[i - 1].close;
    const gain = change > 0 ? change : 0;
    const loss = change < 0 ? -change : 0;
    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;
    rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    rsiValue = 100 - (100 / (1 + rs));
    rsi.push({ time: candles[i].time, value: rsiValue });
  }
  return rsi;
}

// Функция расчета EMA
function calculateEMA(values, period) {
  const k = 2 / (period + 1);
  let ema = [];
  let sum = 0;
  for (let i = 0; i < period; i++) { sum += values[i]; }
  let prevEma = sum / period;
  ema[period - 1] = prevEma;
  for (let i = period; i < values.length; i++) {
    const currentEma = values[i] * k + prevEma * (1 - k);
    ema[i] = currentEma;
    prevEma = currentEma;
  }
  return ema;
}

// Функция расчета MACD
function calculateMACD(candles) {
  const closes = candles.map(c => c.close);
  const ema12 = calculateEMA(closes, 12);
  const ema26 = calculateEMA(closes, 26);
  let macdArray = [];
  const startIndex = 26 - 1;
  for (let i = startIndex; i < closes.length; i++) {
    if (ema12[i] !== undefined && ema26[i] !== undefined) {
      macdArray.push(ema12[i] - ema26[i]);
    }
  }
  const signalArray = calculateEMA(macdArray, 9);
  let macdData = [], signalData = [];
  for (let i = startIndex; i < closes.length; i++) {
    const idx = i - startIndex;
    if (signalArray[idx] !== undefined) {
      macdData.push({ time: candles[i].time, value: macdArray[idx] });
      signalData.push({ time: candles[i].time, value: signalArray[idx] });
    }
  }
  return { macdData, signalData };
}

// Функция расчета ATR
function calculateATR(candles, period = 14) {
  if (candles.length < period + 1) return [];
  let atr = [];
  for (let i = 1; i < candles.length; i++) {
    const current = candles[i], prev = candles[i - 1];
    const tr = Math.max(
      current.high - current.low,
      Math.abs(current.high - prev.close),
      Math.abs(current.low - prev.close)
    );
    atr.push(tr);
  }
  let atrSMA = [];
  for (let i = period - 1; i < atr.length; i++) {
    let sum = 0;
    for (let j = i - period + 1; j <= i; j++) { sum += atr[j]; }
    atrSMA.push({ time: candles[i+1].time, value: sum / period });
  }
  return atrSMA;
}

// Функция расчета стохастических осцилляторов
function calculateStochastic(candles, kPeriod = 14, dPeriod = 3) {
  if (candles.length < kPeriod) return { stochasticK: [], stochasticD: [] };
  let stochasticK = [];
  for (let i = kPeriod - 1; i < candles.length; i++) {
    let periodSlice = candles.slice(i - kPeriod + 1, i + 1);
    let highMax = Math.max(...periodSlice.map(c => c.high));
    let lowMin = Math.min(...periodSlice.map(c => c.low));
    let kValue = ((candles[i].close - lowMin) / (highMax - lowMin)) * 100;
    stochasticK.push({ time: candles[i].time, value: kValue });
  }
  let stochasticD = [];
  for (let i = dPeriod - 1; i < stochasticK.length; i++) {
    let sum = 0;
    for (let j = i - dPeriod + 1; j <= i; j++) { sum += stochasticK[j].value; }
    stochasticD.push({ time: stochasticK[i].time, value: sum / dPeriod });
  }
  return { stochasticK, stochasticD };
}

// Обновление объёмных индикаторов
function calculateVolumeMA(candles, period = 10) {
  if (candles.length < period) return [];
  let volMA = [];
  for (let i = period - 1; i < candles.length; i++) {
    let sum = 0;
    for (let j = i - period + 1; j <= i; j++) { sum += candles[j].volume; }
    volMA.push({ time: candles[i].time, value: sum / period });
  }
  return volMA;
}

function updateVolumeIndicators(candles) {
  const volMAData = calculateVolumeMA(candles, 10);
  volumeMASeries.setData(volMAData);
}

// Получение данных свечей через backend (универсальный endpoint для любой биржи)
async function fetchCandlesAndUpdateCharts() {
  try {
    const url = `/candles?symbol=${currentSymbol}&interval=${currentInterval}&exchange=${currentExchange}&limit=61`;
    const res = await fetch(url);
    const candles = await res.json();
    if (!candles || candles.length === 0) return;
    latestCandles = candles;
    const liveCandle = candles[candles.length - 1];
    lastUpdatedTime = liveCandle.time;
    candleSeries.setData(candles);
    updateLivePriceLine(liveCandle.close, liveCandle.open);

    // Обновление объёмного графика
    const volData = candles.map(c => ({
      time: c.time,
      value: c.volume,
      color: c.close >= c.open ? "rgba(76,175,80,0.5)" : "rgba(229,57,53,0.5)"
    }));
    volumeSeries.setData(volData);
    updateVolumeIndicators(candles);
    updateMarketInfo(liveCandle);
    updateOverlays(latestCandles);
    updateTechnicalIndicators(latestCandles);

    // Получение последних прогнозов через backend
    const predictionData = await fetchLatestPrediction();
    if (predictionData && predictionData.forecast && predictionData.predict) {
      const forecast = predictionData.forecast;
      const predict = predictionData.predict;
      const intervalSec = getIntervalSeconds(currentInterval);
      if (predict.time <= liveCandle.time) predict.time = liveCandle.time + intervalSec;
      if (forecast.time <= liveCandle.time) forecast.time = liveCandle.time + intervalSec;
      if (forecastPriceLine) {
        candleSeries.removePriceLine(forecastPriceLine);
        forecastPriceLine = null;
      }
      forecastPriceLine = candleSeries.createPriceLine({
        price: forecast.close,
        color: "rgba(228,228,225,0.5)",
        lineWidth: 2,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        axisLabelVisible: true,
        title: "F"
      });
      lastForecastTime = forecast.time;
      if (predictPriceLine) {
        candleSeries.removePriceLine(predictPriceLine);
        predictPriceLine = null;
      }
      const futureColor = predictionData.signal === "buy" ? "rgba(76,175,80,0.4)" : "rgba(229,57,53,0.4)";
      predictPriceLine = candleSeries.createPriceLine({
        price: predict.close,
        color: futureColor,
        lineWidth: 2,
        lineStyle: LightweightCharts.LineStyle.Solid,
        axisLabelVisible: true,
        title: "P"
      });
      predictedCandleSeries.setData([{
        time: predict.time,
        open: predict.open,
        high: predict.high,
        low: predict.low,
        close: predict.close
      }]);
      if (predictionData.signal !== "waiting" && predictionData.signal !== "error") {
        showNotification(predictionData.signal, predictionData.confidence, predict);
      }
    }
  } catch (e) {
    console.error("Ошибка fetchCandlesAndUpdateCharts:", e);
  }
}

// Обновление последней свечи
async function updateLiveCandle() {
  try {
    const url = `/candles?symbol=${currentSymbol}&interval=${currentInterval}&exchange=${currentExchange}&limit=2`;
    const res = await fetch(url);
    const data = await res.json();
    if (!data || data.length < 2) return;
    const k = data[data.length - 1];
    const newCandleTime = k.time;
    const newCandle = {
      time: newCandleTime,
      open: +k.open,
      high: +k.high,
      low: +k.low,
      close: +k.close,
      volume: +k.volume,
      timestamp: k.timestamp
    };
    if (newCandleTime >= lastUpdatedTime) {
      candleSeries.update(newCandle);
      liveCandle = newCandle;
      lastUpdatedTime = newCandleTime;
      updateMarketInfo(newCandle);
    }
    await fetchCandlesAndUpdateCharts();
  } catch (e) {
    console.error("Ошибка updateLiveCandle:", e);
  }
}

// Получение последнего предсказания через backend
async function fetchLatestPrediction() {
  try {
    const url = `/latest_prediction?interval=${currentInterval}&exchange=${currentExchange}`;
    const res = await fetch(url);
    const data = await res.json();
    return data[currentSymbol];
  } catch (e) {
    console.error("Ошибка получения последнего предсказания:", e);
    return null;
  }
}




// Инициализация выпадающего списка выбора пары
function initPairDropdown() {
  const pairInput = document.getElementById("pair-input");
  const pairOptions = document.getElementById("pair-options");
  pairInput.addEventListener("focus", () => { pairOptions.style.display = "block"; });
  document.addEventListener("click", function(e) {
    if (!document.getElementById("pair-dropdown").contains(e.target)) {
      pairOptions.style.display = "none";
    }
  });
  pairInput.addEventListener("input", () => {
    const filter = pairInput.value.toUpperCase();
    const options = pairOptions.getElementsByClassName("option");
    Array.from(options).forEach(option => {
      const text = option.textContent || option.innerText;
      option.style.display = text.toUpperCase().indexOf(filter) > -1 ? "" : "none";
    });
  });
  const options = pairOptions.getElementsByClassName("option");
  Array.from(options).forEach(option => {
    option.addEventListener("click", () => {
      const value = option.getAttribute("data-value");
      pairInput.value = option.textContent.trim();
      currentSymbol = value;
      localStorage.setItem("selectedPair", currentSymbol);
      pairOptions.style.display = "none";
      fetchCandlesAndUpdateCharts();
      updateForecastComparisonsTable();
    });
  });
}

// Обновление опций графика (сетка, кроссхейр)
function updateChartOptions() {
  const options = {
    grid: {
      vertLines: { visible: document.getElementById("toggleGrid").checked, color: "#1f2937" },
      horzLines: { visible: document.getElementById("toggleGrid").checked, color: "#1f2937" }
    },
    crosshair: {
      mode: document.getElementById("toggleCrosshair").checked
        ? LightweightCharts.CrosshairMode.Normal
        : LightweightCharts.CrosshairMode.Disabled
    }
  };
  mainChart.applyOptions(options);
  volumeChart.applyOptions({
    grid: {
      vertLines: { visible: document.getElementById("toggleGrid").checked, color: "#1f2937" },
      horzLines: { visible: false }
    },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal }
  });
}

// Показ уведомления о сигнале
// Функция получения сохранённого сигнала с эндпоинта /api/signals и отображения уведомления
async function showLatestNotification() {
  try {
    // Запрашиваем данные с эндпоинта /api/signals
    const response = await fetch('/api/signals');
    if (!response.ok) {
      throw new Error(`Ошибка HTTP: ${response.status}`);
    }
    const signals = await response.json();

    // Проверяем, что массив сигналов не пуст
    if (!Array.isArray(signals) || signals.length === 0) {
      console.error("Нет доступных сигналов");
      return;
    }

    // Берём последний сигнал из массива
    const latestSignal = signals[signals.length - 1];
    const { signal, confidence, predicted } = latestSignal;

    // Обновляем элемент уведомления
    const notificationEl = document.getElementById("notification");
    notificationEl.className = "notification " + signal.toLowerCase();
    const forecastTimeStr = new Date(predicted.time * 1000).toLocaleString();
    notificationEl.innerHTML = `
      <strong>${signal.toUpperCase()}</strong><br>
      Уверенность: ${confidence.toFixed(2)}<br>
      Прогноз Close: ${predicted.close.toFixed(2)}<br>
      Future Time: ${forecastTimeStr}
    `;
    notificationEl.style.display = "block";

    // Скрываем уведомление через 5 секунд
    setTimeout(() => {
      notificationEl.style.display = "none";
    }, 5000);
  } catch (error) {
    console.error("Ошибка при получении сигнала:", error);
  }
}

// Функция для периодического опроса сервера
function startNotificationPolling(intervalMs = 10000) {
  // Немедленно вызываем функцию для первого отображения уведомления
  showLatestNotification();
  // Затем вызываем её каждые intervalMs миллисекунд
  setInterval(showLatestNotification, intervalMs);
}

// Запускаем периодический опрос
startNotificationPolling();







// Обновление технических индикаторов (RSI, MACD, ATR, Стохастик)
function updateTechnicalIndicators(candles) {
  if (document.getElementById("toggleRSI").checked) {
    const rsiData = calculateRSI(candles, 14);
    rsiSeries.setData(rsiData);
  } else {
    rsiSeries.setData([]);
  }
  if (document.getElementById("toggleMACD").checked) {
    const macdResults = calculateMACD(candles);
    macdSeries.setData(macdResults.macdData);
    macdSignalSeries.setData(macdResults.signalData);
  } else {
    macdSeries.setData([]);
    macdSignalSeries.setData([]);
  }
  if (document.getElementById("toggleATR") && document.getElementById("toggleATR").checked) {
    const atrData = calculateATR(candles, 14);
    atrSeries.setData(atrData);
  } else {
    atrSeries.setData([]);
  }
  if (document.getElementById("toggleStochastic") && document.getElementById("toggleStochastic").checked) {
    const stochasticData = calculateStochastic(candles, 14, 3);
    stochasticKSeries.setData(stochasticData.stochasticK);
    stochasticDSeries.setData(stochasticData.stochasticD);
  } else {
    stochasticKSeries.setData([]);
    stochasticDSeries.setData([]);
  }
}

// Инициализация панели инструментов рисования
function initDrawingToolsPanel() {
  const buttons = document.querySelectorAll("#drawing-tools-panel button");
  buttons.forEach(btn => {
    btn.addEventListener("click", () => {
      buttons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentDrawingTool = btn.getAttribute("data-tool");
      const overlay = document.getElementById("drawing-overlay");
      overlay.style.pointerEvents = currentDrawingTool === "none" ? "none" : "auto";
      // Очистка временных объектов
      if (drawingTempSeries) {
        mainChart.removeSeries(drawingTempSeries);
        drawingTempSeries = null;
      }
      if (drawingTempPriceLine) {
        candleSeries.removePriceLine(drawingTempPriceLine);
        drawingTempPriceLine = null;
      }
      if (drawingTempVerticalLine) {
        drawingTempVerticalLine.remove();
        drawingTempVerticalLine = null;
      }
      if (drawingTempRect) {
        drawingTempRect.remove();
        drawingTempRect = null;
      }
    });
  });
}

// Инициализация страницы и установка обработчиков событий
window.addEventListener("load", () => {
  const savedPair = localStorage.getItem("selectedPair");
  if (savedPair) {
    currentSymbol = savedPair;
    document.getElementById("pair-input").value = savedPair;
  }
  const savedInterval = localStorage.getItem("selectedInterval");
  currentInterval = savedInterval ? savedInterval : document.getElementById("interval").value;
  if (savedInterval) document.getElementById("interval").value = savedInterval;
  const savedExchange = localStorage.getItem("selectedExchange");
  currentExchange = savedExchange ? savedExchange : document.getElementById("exchange").value;
  if (savedExchange) document.getElementById("exchange").value = savedExchange;

  // Обработчик смены биржи
  document.getElementById("exchange").addEventListener("change", function(e) {
    currentExchange = e.target.value;
    localStorage.setItem("selectedExchange", currentExchange);
    fetchCandlesAndUpdateCharts();
  });

  document.getElementById("current-year").innerText = new Date().getFullYear();

  initCharts();
  initDrawingTools();
  initPairDropdown();
  initDrawingToolsPanel();
  fetchCandlesAndUpdateCharts();
  updateForecastComparisonsTable();
  setInterval(updateLiveCandle, 10000);
  setInterval(updateForecastComparisonsTable, 60000);

  document.getElementById("toggleGrid").addEventListener("change", updateChartOptions);
  document.getElementById("toggleCrosshair").addEventListener("change", updateChartOptions);
  document.getElementById("toggleMA").addEventListener("change", () => updateOverlays(latestCandles));
  document.getElementById("toggleBB").addEventListener("change", () => updateOverlays(latestCandles));
  document.getElementById("toggleRSI").addEventListener("change", () => updateTechnicalIndicators(latestCandles));
  document.getElementById("toggleMACD").addEventListener("change", () => updateTechnicalIndicators(latestCandles));
  if (document.getElementById("toggleATR"))
    document.getElementById("toggleATR").addEventListener("change", () => updateTechnicalIndicators(latestCandles));
  if (document.getElementById("toggleStochastic"))
    document.getElementById("toggleStochastic").addEventListener("change", () => updateTechnicalIndicators(latestCandles));
});

window.addEventListener("resize", () => {
  if (mainChart && volumeChart) {
    mainChart.resize(document.getElementById("main-chart").clientWidth, 600);
    volumeChart.resize(document.getElementById("volume-chart").clientWidth, 150);
  }
});
