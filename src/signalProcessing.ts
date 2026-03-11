// signalProcessing.ts
// Pure TypeScript signal processing for rPPG: detrending, filtering, FFT, HR estimation.

/**
 * Remove linear trend from a signal using least-squares regression.
 */
export function detrend(signal: Float32Array): Float32Array {
  const n = signal.length;
  if (n < 2) return new Float32Array(signal);

  // Compute linear regression: y = a + b*x
  let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
  for (let i = 0; i < n; i++) {
    sumX += i;
    sumY += signal[i];
    sumXY += i * signal[i];
    sumXX += i * i;
  }
  const denom = n * sumXX - sumX * sumX;
  const b = (n * sumXY - sumX * sumY) / denom;
  const a = (sumY - b * sumX) / n;

  const result = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    result[i] = signal[i] - (a + b * i);
  }
  return result;
}

/** Pre-computed Hann window for the 150-sample BVP buffer (saves 150 Math.cos ops per call). */
const HANN_150: Float32Array = (() => {
  const w = new Float32Array(150);
  for (let i = 0; i < 150; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / 149));
  }
  return w;
})();

/**
 * Apply a Hann window to reduce spectral leakage before FFT.
 */
export function hannWindow(signal: Float32Array): Float32Array {
  const n = signal.length;
  const result = new Float32Array(n);
  if (n === 150) {
    // Fast path: use the pre-computed 150-element window.
    for (let i = 0; i < 150; i++) result[i] = signal[i] * HANN_150[i];
  } else {
    for (let i = 0; i < n; i++) {
      result[i] = signal[i] * (0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1))));
    }
  }
  return result;
}

/**
 * Normalize a signal to zero mean and unit variance.
 */
export function normalize(signal: Float32Array): Float32Array {
  const n = signal.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += signal[i];
  mean /= n;

  let variance = 0;
  for (let i = 0; i < n; i++) variance += (signal[i] - mean) ** 2;
  variance /= n;

  const std = Math.sqrt(variance) || 1;
  const result = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    result[i] = (signal[i] - mean) / std;
  }
  return result;
}

/**
 * Apply a 2nd-order Butterworth bandpass filter.
 * Implemented as cascaded biquad (second-order sections).
 *
 * @param signal Input signal
 * @param fs Sample rate in Hz
 * @param lowCut Low cutoff frequency in Hz
 * @param highCut High cutoff frequency in Hz
 * @returns Filtered signal
 */
export function bandpassFilter(
  signal: Float32Array,
  fs: number,
  lowCut: number = 0.67,   // 40 BPM
  highCut: number = 4    // 240 BPM
): Float32Array {
  // Design 2nd-order Butterworth bandpass via bilinear transform
  // Using cascaded high-pass + low-pass biquad sections

  const result = new Float32Array(signal.length);
  result.set(signal);

  // High-pass filter (removes below lowCut Hz)
  applyBiquadHP(result, fs, lowCut);
  // Low-pass filter (removes above highCut Hz)
  applyBiquadLP(result, fs, highCut);

  return result;
}

/** 2nd-order Butterworth high-pass biquad (in-place) */
function applyBiquadHP(data: Float32Array, fs: number, fc: number): void {
  const w0 = (2 * Math.PI * fc) / fs;
  const alpha = Math.sin(w0) / (2 * Math.SQRT2); // Q = sqrt(2)/2 for Butterworth
  const cosW0 = Math.cos(w0);

  const b0 = (1 + cosW0) / 2;
  const b1 = -(1 + cosW0);
  const b2 = (1 + cosW0) / 2;
  const a0 = 1 + alpha;
  const a1 = -2 * cosW0;
  const a2 = 1 - alpha;

  applyBiquad(data, b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0);
}

/** 2nd-order Butterworth low-pass biquad (in-place) */
function applyBiquadLP(data: Float32Array, fs: number, fc: number): void {
  const w0 = (2 * Math.PI * fc) / fs;
  const alpha = Math.sin(w0) / (2 * Math.SQRT2);
  const cosW0 = Math.cos(w0);

  const b0 = (1 - cosW0) / 2;
  const b1 = 1 - cosW0;
  const b2 = (1 - cosW0) / 2;
  const a0 = 1 + alpha;
  const a1 = -2 * cosW0;
  const a2 = 1 - alpha;

  applyBiquad(data, b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0);
}

/** Apply a biquad filter section to data in-place (Direct Form I) */
function applyBiquad(
  data: Float32Array,
  b0: number, b1: number, b2: number,
  a1: number, a2: number
): void {
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;

  // Forward pass
  for (let i = 0; i < data.length; i++) {
    const x0 = data[i];
    const y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    x2 = x1; x1 = x0;
    y2 = y1; y1 = y0;
    data[i] = y0;
  }

  // Backward pass (zero-phase filtering like scipy.signal.filtfilt)
  x1 = 0; x2 = 0; y1 = 0; y2 = 0;
  for (let i = data.length - 1; i >= 0; i--) {
    const x0 = data[i];
    const y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    x2 = x1; x1 = x0;
    y2 = y1; y1 = y0;
    data[i] = y0;
  }
}

/**
 * Compute FFT magnitude spectrum using radix-2 Cooley-Tukey algorithm.
 * Input length is zero-padded to next power of 2.
 *
 * @returns Object with magnitudes and corresponding frequencies
 */
export function fft(signal: Float32Array, sampleRate: number): {
  magnitudes: Float32Array;
  frequencies: Float32Array;
} {
  // Zero-pad to next power of 2
  const n = nextPow2(signal.length);
  const real = new Float32Array(n);
  const imag = new Float32Array(n);
  real.set(signal);

  // Bit-reversal permutation
  const bits = Math.log2(n);
  for (let i = 0; i < n; i++) {
    const j = bitReverse(i, bits);
    if (j > i) {
      let tmp = real[i]; real[i] = real[j]; real[j] = tmp;
      tmp = imag[i]; imag[i] = imag[j]; imag[j] = tmp;
    }
  }

  // Cooley-Tukey butterfly with per-stage twiddle pre-computation
  // (avoids calling Math.cos/sin O(n log n) times in the inner loop).
  for (let size = 2; size <= n; size *= 2) {
    const halfSize = size / 2;
    const angleStep = (-2 * Math.PI) / size;

    const twr = new Float32Array(halfSize);
    const twi = new Float32Array(halfSize);
    for (let j = 0; j < halfSize; j++) {
      twr[j] = Math.cos(angleStep * j);
      twi[j] = Math.sin(angleStep * j);
    }

    for (let i = 0; i < n; i += size) {
      for (let j = 0; j < halfSize; j++) {
        const evenIdx = i + j;
        const oddIdx = i + j + halfSize;

        const tReal = twr[j] * real[oddIdx] - twi[j] * imag[oddIdx];
        const tImag = twi[j] * real[oddIdx] + twr[j] * imag[oddIdx];

        real[oddIdx] = real[evenIdx] - tReal;
        imag[oddIdx] = imag[evenIdx] - tImag;
        real[evenIdx] += tReal;
        imag[evenIdx] += tImag;
      }
    }
  }

  // Compute magnitudes (only first half — positive frequencies)
  const halfN = n / 2;
  const magnitudes = new Float32Array(halfN);
  const frequencies = new Float32Array(halfN);
  const freqStep = sampleRate / n;

  for (let i = 0; i < halfN; i++) {
    magnitudes[i] = Math.hypot(real[i], imag[i]);
    frequencies[i] = i * freqStep;
  }

  return { magnitudes, frequencies };
}

function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

function bitReverse(x: number, bits: number): number {
  let result = 0;
  for (let i = 0; i < bits; i++) {
    result = (result << 1) | (x & 1);
    x >>= 1;
  }
  return result;
}

export interface HREstimate {
  /** Heart rate in BPM */
  hr: number;
  /** Confidence score 0-1 (SNR-based) */
  confidence: number;
}

/**
 * Estimate heart rate from a BVP (blood volume pulse) signal.
 *
 * Pipeline: detrend → normalize → Hann window → bandpass → FFT → peak detection
 *
 * @param bvpSignal Raw BVP signal from the rPPG model
 * @param sampleRate Actual frames per second
 * @returns Heart rate in BPM and confidence score
 */
export function estimateHeartRate(bvpSignal: Float32Array, sampleRate: number): HREstimate {
  if (bvpSignal.length < 32 || sampleRate < 5) {
    return { hr: 0, confidence: 0 };
  }

  // Signal conditioning pipeline
  let signal = detrend(bvpSignal);
  signal = normalize(signal);
  signal = hannWindow(signal);
  signal = bandpassFilter(signal, sampleRate, 0.67, 4);

  // FFT
  const { magnitudes, frequencies } = fft(signal, sampleRate);

  // Find dominant peak in physiological HR range (40-240 BPM → 0.67-4.0 Hz)
  const minFreq = 0.67;
  const maxFreq = 4;
  let peakMag = -Infinity;
  let peakFreq = 0;
  let totalPower = 0;
  let peakCount = 0;

  for (let i = 0; i < frequencies.length; i++) {
    if (frequencies[i] >= minFreq && frequencies[i] <= maxFreq) {
      const mag = magnitudes[i];
      totalPower += mag;
      peakCount++;
      if (mag > peakMag) {
        peakMag = mag;
        peakFreq = frequencies[i];
      }
    }
  }

  const hr = peakFreq * 60; // Convert Hz to BPM
  const avgPower = peakCount > 0 ? totalPower / peakCount : 1;
  // SNR-based confidence: how much the peak stands out from average
  const snr = avgPower > 0 ? peakMag / avgPower : 0;
  // Map SNR to 0-1 range (SNR of 3+ = high confidence)
  const confidence = Math.min(1, Math.max(0, (snr - 1) / 4));

  return { hr, confidence };
}

/**
 * HR smoother using exponential moving average with outlier rejection.
 */
export class HRSmoother {
  private history: number[] = [];
  private readonly maxHistory: number;
  private readonly maxJump: number;
  private lastValidHR = 0;

  constructor(maxHistory = 8, maxJump = 20) {
    this.maxHistory = maxHistory;
    this.maxJump = maxJump;
  }

  /**
   * Add a raw HR estimate and return the smoothed value.
   * Rejects outliers that jump more than maxJump BPM from the running average.
   */
  update(hr: number, confidence: number): number {
    if (hr <= 0 || confidence < 0.05) return this.lastValidHR;

    // Outlier rejection: if we have history, reject huge jumps
    if (this.history.length > 2) {
      const avg = this.history.reduce((a, b) => a + b, 0) / this.history.length;
      if (Math.abs(hr - avg) > this.maxJump) {
        // Reject but allow slow convergence
        hr = avg + Math.sign(hr - avg) * this.maxJump * 0.3;
      }
    }

    this.history.push(hr);
    if (this.history.length > this.maxHistory) {
      this.history.shift();
    }

    // Exponential moving average (more recent values weighted higher)
    let ema = this.history[0];
    const alpha = 0.3;
    for (let i = 1; i < this.history.length; i++) {
      ema = alpha * this.history[i] + (1 - alpha) * ema;
    }

    this.lastValidHR = Math.round(ema);
    return this.lastValidHR;
  }

  reset(): void {
    this.history = [];
    this.lastValidHR = 0;
  }

  getHistory(): number[] {
    return [...this.history];
  }
}
