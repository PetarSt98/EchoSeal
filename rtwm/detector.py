"""
Offline watermark detector: ingest a recording, return frames and a verdict.

Detection pipeline (per hop band):

    resample to 48 kHz  →  band-pass  →  preamble correlation (sync)  →
    coherent BPSK demod (channel phase estimated from the preamble)  →
    header majority vote (frame counter)  →  PN despread  →  LLRs  →
    CA-SCL polar decode  →  AEAD decrypt + magic/counter verification

Every frame is self-contained, so detection works on any excerpt of a longer
recording — nothing depends on absolute time or on the talk's beginning being
present.  Polarity inversion of the recording is absorbed by the phase
estimate, and the frame counter is recovered from the header rather than from
the frame's position.

Verdict layer (:meth:`WatermarkDetector.analyze`): decoded frames are
cryptographically authentic by construction, so tampering is judged on their
*consistency* — plus a waveform-integrity check inside each decoded frame:

    * all frames must carry the same session nonce      (splice detection)
    * no frame counter may repeat                       (replay detection)
    * counter deltas must match position deltas         (cut / insert detection)
    * holes in an otherwise healthy timeline            (same-length replacement)
    * decoded frames must match their reconstructed
      TX waveform window-by-window                      (sub-frame edits)
    * a single undecodable frame between healthy
      neighbours gets erasure-assisted re-decoding      (small edits, benign loss)

The waveform check exploits that a decoded frame is fully known: its exact
transmitted waveform can be regenerated and correlated against the recording
in ~21 ms windows, localising edits far smaller than a frame.  The recovery
step distinguishes a benignly lost frame (weak everywhere) from an edited one
(dead zone inside an otherwise coherent frame) and often re-authenticates the
untouched remainder of the frame via erasure decoding.  Frames that are
merely *missing* on a uniformly degraded channel (aggressive codec, heavy
noise) never count as tampering on their own.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import correlate, lfilter

from rtwm import frame
from rtwm.crypto import SecureChannel
from rtwm.polar_fast import decode as polar_dec, encode as polar_enc
from rtwm.utils import BAND_PLAN, butter_bandpass, choose_band, resample_to

EPS = 1e-12

MIN_NCC = 0.18              # absolute floor for preamble correlation peaks
NCC_MAD_FACTOR = 6.0        # adaptive threshold: median + k * MAD
MAX_PEAKS_PER_BAND = 24     # sync candidates examined per band
HEADER_MIN_COHERENCE = 0.55  # within-group sign agreement gate (wrong key ~0.35)
MAX_CTR_EPOCHS = 2          # counter candidates: lo16 + epoch * 2^16 (~14.6 h each)
LLR_CLIP = 25.0

# Timeline consistency: tolerate benign channel effects (clock drift, codec
# delay jitter, smeared sync on heavily attenuated bands) but catch any cut or
# insert of ~a phoneme or more.  1024 samples = 21 ms at 48 kHz.
TIMING_JITTER_SAMPLES = 1024
TIMING_DRIFT_TOLERANCE = 0.01

# Same-length replacements (e.g. an AI-dubbed passage) do not disturb the
# timeline; their fingerprint is a contiguous run of undecodable frames inside
# an otherwise healthy timeline.  Runs of >= MIN_GAP_FRAMES missing frames are
# flagged, but only when the rest of the timeline decodes well — a uniformly
# harsh channel (aggressive codec, heavy noise) must not be mistaken for
# tampering.  (Spans are still reported for information either way.)
# A *single* missing frame is empirically common on benign channels (band-edge
# codecs), so it is never flagged directly: it triggers erasure-assisted
# recovery instead (see _recover_single_gaps).
MIN_GAP_FRAMES = 2
HEALTHY_COVERAGE = 0.8

# Erasure-assisted recovery of single missing frames.  The frame's expected
# position and counter are known from its neighbours; dead symbol runs are
# located with a non-data-aided BPSK coherence metric (|sum z^2| / sum |z|^2,
# which needs no knowledge of the data bits), padded, erased (LLR = 0) and the
# polar decoder is retried.  A successful authenticated decode is
# cryptographic proof that the frame is genuine *except* the erased span ->
# flagged as a local edit with zero false-positive risk.  If decoding still
# fails but the dead run is unambiguous (hard-dead inside an otherwise
# coherent frame), it is flagged on signal evidence.  A frame that is weak
# *everywhere* (codec band-kill, noise) has no localised dead run and is left
# as benign degradation.
RECOVERY_MAX_PER_SCAN = 3
COHERENCE_WINDOW = 32        # symbols per coherence window (~21 ms)
COHERENCE_PAD = 16           # symbols of padding around a dead run
COHERENCE_DEAD_REL = 0.5     # dead: below this fraction of the frame median
COHERENCE_HARD_DEAD_REL = 0.3  # signal-evidence threshold when decode fails
COHERENCE_MIN_MEDIAN = 0.5   # below this, the whole frame is unjudgeable
ERASURE_MAX_SYMBOLS = 512    # beyond this the rate-0.44 code cannot recover
RECOVERY_SILENCE_RMS = 1e-3  # dead zones this quiet may be the silence gate

# Waveform integrity inside decoded frames: correlate the recording against
# the reconstructed TX waveform in half-overlapping windows.  Two metrics per
# window — matched-filter gain (robust to loud host audio) and NCC (robust to
# quiet-host watermark level dips) — and a window is anomalous only when BOTH
# collapse relative to the frame's own medians, for >= WAVEFORM_MIN_RUN
# consecutive windows.  Frames whose overall match is too poor are skipped.
WAVEFORM_WINDOW_SYMBOLS = 32          # 32 symbols = 1024 samples ~ 21 ms
WAVEFORM_REL_THRESHOLD = 0.4
WAVEFORM_MIN_MEDIAN = 0.4
WAVEFORM_MIN_RUN = 2                  # >= 2 bad windows ~ >= 32 ms edit


@dataclass(frozen=True)
class FrameHit:
    """One successfully decoded and decrypted watermark frame."""

    start: int                # sample index of the frame in the 48 kHz signal
    band: tuple[int, int]
    ctr: int                  # frame counter, verified against the sealed payload
    nonce: bytes              # 8-byte session nonce from the decrypted payload


@dataclass(frozen=True)
class Report:
    """Outcome of :meth:`WatermarkDetector.analyze`."""

    hits: list[FrameHit]
    issues: list[str] = field(default_factory=list)
    unverified_spans: list[tuple[float, float]] = field(default_factory=list)
    """(t0, t1) seconds carrying no verifiable watermark inside an otherwise
    healthy timeline — possible replacement (e.g. AI-generated insert)."""

    @property
    def verdict(self) -> str:
        """"authentic" | "tampered" | "no-watermark"."""
        if not self.hits:
            return "no-watermark"
        return "tampered" if self.issues else "authentic"

    @property
    def coverage(self) -> float:
        """Fraction of expected frames found between first and last hit."""
        if not self.hits:
            return 0.0
        ctrs = {h.ctr for h in self.hits}
        span = max(ctrs) - min(ctrs) + 1
        return len(ctrs) / span


class WatermarkDetector:
    """Recover EchoSeal watermark frames from a recording (offline)."""

    def __init__(
        self,
        key32: bytes,
        *,
        fs_target: int = 48_000,
        sps: int = frame.SPS,
        list_size: int = 8,
        max_decodes_per_band: int = 8,
    ) -> None:
        self.sec = SecureChannel(key32)
        self.fs = fs_target
        self.sps = sps
        self.list_size = list_size
        self.max_decodes_per_band = max_decodes_per_band
        self._templates: dict[tuple[int, int], np.ndarray] = {}

    # ------------------------------------------------------------------ API
    def verify(self, audio: np.ndarray, fs_in: int) -> bool:
        """True iff the recording carries a watermark and shows no tampering."""
        return self.analyze(audio, fs_in).verdict == "authentic"

    def analyze(self, audio: np.ndarray, fs_in: int) -> Report:
        """Scan the recording and judge the consistency of the decoded frames."""
        hits, wf_spans, wf_issues = self._scan_full(audio, fs_in)
        issues = self._consistency_issues(hits)
        gap_spans, gap_issues = self._unverified_spans(hits)
        return Report(
            hits=hits,
            issues=issues + gap_issues + wf_issues,
            unverified_spans=sorted(gap_spans + wf_spans),
        )

    def scan(self, audio: np.ndarray, fs_in: int) -> list[FrameHit]:
        """Return all decodable frames, sorted by position (no dedup: a
        repeated counter is evidence, not noise)."""
        return self._scan_full(audio, fs_in)[0]

    def _scan_full(
        self, audio: np.ndarray, fs_in: int
    ) -> tuple[list[FrameHit], list[tuple[float, float]], list[str]]:
        x = np.asarray(audio, dtype=np.float64)
        if x.ndim == 2:  # downmix stereo recordings
            x = x.mean(axis=1)
        signal, _ = resample_to(self.fs, x, int(fs_in))

        hits: list[FrameHit] = []
        spans: list[tuple[float, float]] = []
        issues: list[str] = []
        for band in BAND_PLAN:
            band_hits, band_spans, band_issues = self._scan_band(signal, band)
            hits.extend(band_hits)
            spans.extend(band_spans)
            issues.extend(band_issues)
        hits.sort(key=lambda h: h.start)

        # Second pass: single-frame holes between healthy neighbours.
        n_frame = frame.frame_samples(self.sps)
        attempts = 0
        for a, b in list(zip(hits, hits[1:])):
            if attempts >= RECOVERY_MAX_PER_SCAN:
                break
            if b.ctr - a.ctr != 2 or a.nonce != b.nonce:
                continue
            if abs((b.start - a.start) - 2 * n_frame) > TIMING_JITTER_SAMPLES:
                continue  # timeline broken here: the cut/insert check owns it
            attempts += 1
            hit, r_spans, r_issues = self._recover_single_gap(signal, a)
            if hit is not None:
                hits.append(hit)
            spans.extend(r_spans)
            issues.extend(r_issues)

        return sorted(hits, key=lambda h: h.start), spans, issues

    # ------------------------------------------------------- tamper analysis
    def _consistency_issues(self, hits: list[FrameHit]) -> list[str]:
        issues: list[str] = []
        if len(hits) < 2:
            return issues
        n_frame = frame.frame_samples(self.sps)

        nonces = {h.nonce for h in hits}
        if len(nonces) > 1:
            issues.append(
                f"{len(nonces)} different session nonces present "
                "(splice of separate recordings suspected)"
            )

        seen: dict[int, int] = {}
        for hit in hits:
            if hit.ctr in seen:
                issues.append(
                    f"frame counter {hit.ctr} appears twice "
                    f"(t={seen[hit.ctr] / self.fs:.2f}s and t={hit.start / self.fs:.2f}s; "
                    "copied/replayed audio suspected)"
                )
            seen.setdefault(hit.ctr, hit.start)

        for a, b in zip(hits, hits[1:]):
            if b.ctr <= a.ctr:
                continue  # repeats already reported; ordering is by position
            expected = (b.ctr - a.ctr) * n_frame
            tolerance = max(
                TIMING_JITTER_SAMPLES, int(TIMING_DRIFT_TOLERANCE * expected)
            )
            offset = (b.start - a.start) - expected
            if abs(offset) > tolerance:
                what = "missing" if offset < 0 else "inserted"
                issues.append(
                    f"~{abs(offset) / self.fs:.2f}s of audio {what} between "
                    f"frames {a.ctr} and {b.ctr} "
                    f"(around t={a.start / self.fs:.2f}s; cut/insert suspected)"
                )
        return issues

    def _unverified_spans(
        self, hits: list[FrameHit]
    ) -> tuple[list[tuple[float, float]], list[str]]:
        """Contiguous runs of undecodable frames inside a healthy timeline.

        A same-length replacement (AI dub, local wipe) leaves the timeline
        intact but produces a hole spanning *all* bands.  Spans are always
        reported; they count as tampering (issues) only when the remaining
        timeline decodes well, so uniformly degraded channels are not
        misreported.
        """
        if len(hits) < 2:
            return [], []

        ctrs = sorted({h.ctr for h in hits})
        gaps = [
            (a, b, b - a - 1)
            for a, b in zip(ctrs, ctrs[1:])
            if b - a - 1 >= MIN_GAP_FRAMES
        ]
        if not gaps:
            return [], []

        # Coverage of the timeline *outside* the candidate holes.  On a poor
        # channel holes prove nothing, so they stay informational only.
        span = ctrs[-1] - ctrs[0] + 1
        missing_in_gaps = sum(g for _, _, g in gaps)
        healthy = len(ctrs) / max(1, span - missing_in_gaps) >= HEALTHY_COVERAGE

        n_frame = frame.frame_samples(self.sps)
        by_ctr = {h.ctr: h for h in hits}
        spans: list[tuple[float, float]] = []
        issues: list[str] = []
        for a, b, missing in gaps:
            t0 = (by_ctr[a].start + n_frame) / self.fs
            t1 = by_ctr[b].start / self.fs
            spans.append((t0, t1))
            if healthy:
                issues.append(
                    f"no verifiable watermark between t={t0:.2f}s and t={t1:.2f}s "
                    f"({missing} frame(s) silent while surrounding audio verifies; "
                    "replaced or AI-generated segment suspected)"
                )
        return spans, issues

    # ------------------------------------------------------------ band scan
    def _template(self, band: tuple[int, int]) -> np.ndarray:
        """Unit-norm preamble template as seen after TX *and* RX filtering."""
        tpl = self._templates.get(band)
        if tpl is None:
            b, a = butter_bandpass(*band, self.fs, order=4)
            tpl = lfilter(b, a, frame.preamble_waveform(band, self.fs, self.sps))
            tpl = tpl / (np.linalg.norm(tpl) + EPS)
            self._templates[band] = tpl
        return tpl

    def _scan_band(
        self, signal: np.ndarray, band: tuple[int, int]
    ) -> tuple[list[FrameHit], list[tuple[float, float]], list[str]]:
        n_frame = frame.frame_samples(self.sps)
        if signal.size < n_frame:
            return [], [], []

        b, a = butter_bandpass(*band, self.fs, order=4)
        y = lfilter(b, a, signal)

        peaks = self._preamble_peaks(y, band, n_frame)

        hits: list[FrameHit] = []
        spans: list[tuple[float, float]] = []
        issues: list[str] = []
        failures = 0  # only unsuccessful decodes count against the budget
        for peak in peaks:
            aligned = self._demod_frame(y, peak, band)
            if aligned is None:
                continue
            start, symbols = aligned

            ok, ctr_lo16 = self._header_counter(symbols)
            if not ok:
                continue

            for epoch in range(MAX_CTR_EPOCHS):
                ctr = (epoch << 16) | ctr_lo16
                if choose_band(self.sec.band_key, ctr) != band:
                    continue
                payload = polar_dec(
                    self._payload_llr(symbols, ctr),
                    list_size=self.list_size,
                    validator=self._validator(ctr),
                )
                if payload is not None:
                    plain = self.sec.open(payload)  # re-open: cheap, µs-scale
                    hits.append(FrameHit(start, band, ctr, plain[8:16]))
                    wf_spans, wf_issues = self._waveform_anomalies(
                        y, start, band, ctr, payload
                    )
                    spans.extend(wf_spans)
                    issues.extend(wf_issues)
                    break
                failures += 1
                if failures >= self.max_decodes_per_band:
                    return hits, spans, issues
        return hits, spans, issues

    def _preamble_peaks(
        self, y: np.ndarray, band: tuple[int, int], n_frame: int
    ) -> list[int]:
        """Candidate frame starts: peaks of |NCC(y, preamble template)|."""
        tpl = self._template(band)
        L = tpl.size
        if y.size < L:
            return []

        corr = correlate(y, tpl, mode="valid", method="fft")
        csum = np.concatenate(([0.0], np.cumsum(y * y)))
        energy = np.sqrt(csum[L:] - csum[:-L]) + EPS
        ncc = np.abs(corr / energy)

        med = float(np.median(ncc))
        mad = float(np.median(np.abs(ncc - med)))
        thr = max(MIN_NCC, med + NCC_MAD_FACTOR * 1.4826 * mad)

        candidates = np.flatnonzero(ncc >= thr)
        if candidates.size == 0:
            return []

        # Greedy non-maximum suppression, strongest first.
        order = candidates[np.argsort(ncc[candidates])[::-1]]
        peaks: list[int] = []
        for idx in order:
            if all(abs(idx - p) >= n_frame // 2 for p in peaks):
                peaks.append(int(idx))
                if len(peaks) >= MAX_PEAKS_PER_BAND:
                    break
        return peaks

    # ---------------------------------------------------------------- demod
    def _mixer(self, band: tuple[int, int], num: int) -> np.ndarray:
        fc = frame.carrier_freq(band, self.fs, self.sps)
        return np.exp(-2j * np.pi * fc * np.arange(num) / self.fs)

    def _demod_frame(
        self, y: np.ndarray, peak: int, band: tuple[int, int]
    ) -> tuple[int, np.ndarray] | None:
        """Integrate-and-dump demod with timing refinement.

        The sync peak marks the frame *start*, but the double-filtered symbol
        energy lags it by the TX+RX filter group delay (up to ~1.5 symbols for
        the 2 kHz bands), so the search runs from -SPS/2 to +2 SPS and picks
        the offset that maximises preamble coherence.

        Returns (refined start, per-symbol soft values) or None if the frame
        does not fit inside the recording.
        """
        n_frame = frame.frame_samples(self.sps)
        pre_samples = frame.PRE_LEN * self.sps
        mixer_pre = self._mixer(band, pre_samples)

        best_score, best_start = -1.0, None
        for delta in range(-self.sps // 2, 2 * self.sps + 1):
            start = peak + delta
            if start < 0 or start + n_frame > y.size:
                continue
            z_pre = (y[start : start + pre_samples] * mixer_pre)
            z_pre = z_pre.reshape(frame.PRE_LEN, self.sps).sum(axis=1)
            score = float(np.abs(np.dot(z_pre, frame.PRE_SYMBOLS)))
            if score > best_score:
                best_score, best_start = score, start
        if best_start is None:
            return None

        z = (y[best_start : best_start + n_frame] * self._mixer(band, n_frame))
        z = z.reshape(frame.SYMBOLS_PER_FRAME, self.sps).sum(axis=1)

        # One residual channel phase, estimated from the known preamble.
        # (A polarity-inverted recording simply shows up as an extra pi.)
        theta = np.angle(np.dot(z[: frame.PRE_LEN], frame.PRE_SYMBOLS))
        return best_start, np.real(z * np.exp(-1j * theta))

    # --------------------------------------------------------------- header
    def _header_counter(self, symbols: np.ndarray) -> tuple[bool, int]:
        """Majority-vote the 16 counter bits; gate on within-group coherence."""
        seg = symbols[frame.PRE_LEN : frame.PRE_LEN + frame.HDR_LEN]
        hdr_pn = 2.0 * self.sec.pn_bits(0, frame.HDR_LEN).astype(np.float64) - 1.0
        groups = (seg * hdr_pn).reshape(frame.HDR_BITS, frame.HDR_REPEAT)

        sums = groups.sum(axis=1)
        coherence = np.abs(sums) / (np.abs(groups).sum(axis=1) + EPS)
        if float(np.mean(coherence)) < HEADER_MIN_COHERENCE:
            return False, 0

        bits = (sums > 0.0).astype(np.uint8)
        value = int(np.packbits(bits).view(">u2")[0])
        return True, value

    # ---------------------------------------------------------------- LLRs
    def _payload_llr(
        self, symbols: np.ndarray, ctr: int, dead: np.ndarray | None = None
    ) -> np.ndarray:
        """Despread payload symbols with the frame PN and scale to LLRs.

        `dead` (optional bool mask over the 1024 payload symbols) marks
        erasures: those LLRs are zeroed and excluded from the scaling stats.
        """
        pn = self.sec.pn_bits(ctr, frame.SYMBOLS_PER_FRAME)
        pn_payload = pn[frame.PRE_LEN + frame.HDR_LEN :]
        despread = symbols[frame.PRE_LEN + frame.HDR_LEN :] * (
            2.0 * pn_payload.astype(np.float64) - 1.0
        )

        # Robust Gaussian LLR scaling: amplitude from the median magnitude,
        # noise sigma from the MAD of the magnitude residuals.
        alive = despread if dead is None else despread[~dead]
        amp = float(np.median(np.abs(alive)))
        sigma = 1.4826 * float(np.median(np.abs(np.abs(alive) - amp))) + EPS
        llr = np.clip(2.0 * amp * despread / (sigma * sigma), -LLR_CLIP, LLR_CLIP)
        if dead is not None:
            llr[dead] = 0.0
        return llr

    # ------------------------------------------------- single-gap recovery
    def _recover_single_gap(
        self, signal: np.ndarray, prev_hit: FrameHit
    ) -> tuple[FrameHit | None, list[tuple[float, float]], list[str]]:
        """Diagnose the one missing frame that follows `prev_hit`.

        Counter, band and position are inherited from the healthy neighbours
        (frame spacing is exact, so no preamble sync is needed — robust even
        when the edit destroyed the preamble).  Dead symbol runs are located
        with the data-free BPSK coherence |sum z^2| / sum |z|^2, erased
        (LLR = 0) and the polar decoder is retried with both phase signs.
        """
        n_frame = frame.frame_samples(self.sps)
        ctr = prev_hit.ctr + 1
        band = choose_band(self.sec.band_key, ctr)
        start = prev_hit.start + n_frame
        if start < 0 or start + n_frame > signal.size:
            return None, [], []

        # Local band-pass with warm-up padding for the IIR state.
        pad = min(start, 2048)
        y = lfilter(
            *butter_bandpass(*band, self.fs, order=4),
            signal[start - pad : start + n_frame],
        )[pad:]

        z = (y * self._mixer(band, n_frame)).reshape(
            frame.SYMBOLS_PER_FRAME, self.sps
        ).sum(axis=1)

        # Data-free coherence per 32-symbol window: ~1 for clean BPSK of any
        # phase/polarity, ~1/sqrt(W) for noise or foreign audio.
        n_win = frame.SYMBOLS_PER_FRAME // COHERENCE_WINDOW
        zw = z[: n_win * COHERENCE_WINDOW].reshape(n_win, COHERENCE_WINDOW)
        coh = np.abs((zw**2).sum(axis=1)) / ((np.abs(zw) ** 2).sum(axis=1) + EPS)

        med = float(np.median(coh))
        if med < COHERENCE_MIN_MEDIAN:
            return None, [], []  # weak everywhere: benign degradation

        # Dead runs: consecutive dead windows, padded, merged on overlap.
        # Each run carries whether any of its windows is hard-dead (strong
        # signal evidence usable even without a successful erasure decode).
        dead = coh < COHERENCE_DEAD_REL * med
        hard = coh < COHERENCE_HARD_DEAD_REL * med
        runs: list[list[int | bool]] = []  # [sym0, sym1, hard_dead]
        for w in np.flatnonzero(dead):
            s0 = max(0, int(w) * COHERENCE_WINDOW - COHERENCE_PAD)
            s1 = min(
                frame.SYMBOLS_PER_FRAME, (int(w) + 1) * COHERENCE_WINDOW + COHERENCE_PAD
            )
            if runs and s0 <= runs[-1][1]:
                runs[-1][1] = s1
                runs[-1][2] = runs[-1][2] or bool(hard[w])
            else:
                runs.append([s0, s1, bool(hard[w])])

        # Erasure decode: phase from the coherent part (z^2 halves the angle,
        # with an unavoidable pi ambiguity -> try both LLR signs).
        sym_dead = np.zeros(frame.SYMBOLS_PER_FRAME, dtype=bool)
        for s0, s1, _ in runs:
            sym_dead[s0:s1] = True
        payload_dead = sym_dead[frame.PRE_LEN + frame.HDR_LEN :]

        hit = None
        if int(payload_dead.sum()) <= ERASURE_MAX_SYMBOLS:
            theta = 0.5 * np.angle(np.sum(z[~sym_dead] ** 2))
            symbols = np.real(z * np.exp(-1j * theta))
            llr = self._payload_llr(symbols, ctr, dead=payload_dead)
            for sign in (1.0, -1.0):
                payload = polar_dec(
                    sign * llr, list_size=self.list_size, validator=self._validator(ctr)
                )
                if payload is not None:
                    plain = self.sec.open(payload)
                    hit = FrameHit(start, band, ctr, plain[8:16])
                    break

        spans: list[tuple[float, float]] = []
        issues: list[str] = []
        for s0, s1, hard_dead in runs:
            if hit is None and not hard_dead:
                continue  # ambiguous: neither crypto proof nor strong evidence
            zone = signal[start + s0 * self.sps : start + s1 * self.sps]
            if float(np.sqrt(np.mean(zone**2))) < RECOVERY_SILENCE_RMS:
                continue  # silence-gated by the embedder: legitimately unmarked
            t0 = (start + s0 * self.sps) / self.fs
            t1 = (start + s1 * self.sps) / self.fs
            spans.append((t0, t1))
            confirmation = (
                "rest of frame authenticated" if hit is not None
                else "frame otherwise coherent"
            )
            issues.append(
                f"watermark absent between t={t0:.2f}s and t={t1:.2f}s inside "
                f"frame {ctr} ({confirmation}; local edit suspected)"
            )
        return hit, spans, issues

    # ------------------------------------------------------ waveform check
    def _waveform_anomalies(
        self,
        y: np.ndarray,
        start: int,
        band: tuple[int, int],
        ctr: int,
        payload: bytes,
    ) -> tuple[list[tuple[float, float]], list[str]]:
        """Sub-frame integrity check on a *decoded* frame.

        Every bit of a decoded frame is known, so its exact transmitted
        waveform (in-phase and quadrature, making the check channel-phase
        invariant) is reconstructed, aligned to the recording, and compared in
        half-overlapping ~21 ms windows using two metrics:

        * matched-filter gain — insensitive to loud host audio on top of the
          watermark, but dips where the embedder mixed at a low level;
        * normalised correlation — insensitive to the mixing level, but dips
          where the host is locally loud.

        Only windows where BOTH collapse (for >= WAVEFORM_MIN_RUN consecutive
        windows) are flagged: that happens when the watermark is genuinely
        absent, i.e. the audio was locally replaced.
        """
        pn = self.sec.pn_bits(ctr, frame.SYMBOLS_PER_FRAME)
        symbols = frame.assemble_symbols(
            ctr,
            polar_enc(payload),
            self.sec.pn_bits(0, frame.HDR_LEN),
            pn[frame.PRE_LEN + frame.HDR_LEN :],
        )
        b, a = butter_bandpass(*band, self.fs, order=4)
        ref_i = lfilter(b, a, frame.modulate(symbols, band, self.fs, self.sps))
        ref_q = lfilter(
            b, a,
            frame.modulate(symbols, band, self.fs, self.sps, phase=-np.pi / 2),
        )
        n = ref_i.size

        # Global alignment: the demod start absorbs the filter group delay,
        # while the reconstruction starts from zero state.
        best_score, best_d = -1.0, None
        for d in range(-2 * self.sps, 2 * self.sps + 1):
            s0 = start + d
            if s0 < 0 or s0 + n > y.size:
                continue
            seg = y[s0 : s0 + n]
            score = np.hypot(float(np.dot(ref_i, seg)), float(np.dot(ref_q, seg)))
            if score > best_score:
                best_score, best_d = score, d
        if best_d is None:
            return [], []
        seg = y[start + best_d : start + best_d + n]

        win = WAVEFORM_WINDOW_SYMBOLS * self.sps
        hop = win // 2
        gain, ncc = [], []
        for i in range(0, n - win + 1, hop):
            s = seg[i : i + win]
            ri, rq = ref_i[i : i + win], ref_q[i : i + win]
            ei = np.linalg.norm(ri) + EPS
            eq = np.linalg.norm(rq) + EPS
            es = np.linalg.norm(s) + EPS
            m = np.hypot(float(np.dot(ri, s)) / ei, float(np.dot(rq, s)) / eq)
            gain.append(m / ((ei + eq) / 2.0))  # watermark amplitude estimate
            ncc.append(m / es)
        gain_arr, ncc_arr = np.asarray(gain), np.asarray(ncc)

        med_gain = float(np.median(gain_arr))
        med_ncc = float(np.median(ncc_arr))
        if med_ncc < WAVEFORM_MIN_MEDIAN or med_gain < EPS:
            return [], []  # channel too poor to judge sub-frame integrity

        bad = (gain_arr < WAVEFORM_REL_THRESHOLD * med_gain) & (
            ncc_arr < WAVEFORM_REL_THRESHOLD * med_ncc
        )
        if not bad.any():
            return [], []

        # Merge consecutive bad windows into spans (in seconds).
        spans: list[tuple[float, float]] = []
        issues: list[str] = []
        idx = np.flatnonzero(bad)
        run_start = prev = idx[0]
        for i in list(idx[1:]) + [None]:  # sentinel flushes the last run
            if i is not None and i == prev + 1:
                prev = i
                continue
            if prev - run_start + 1 >= WAVEFORM_MIN_RUN:
                t0 = (start + best_d + run_start * hop) / self.fs
                t1 = (start + best_d + prev * hop + win) / self.fs
                spans.append((t0, t1))
                issues.append(
                    f"watermark waveform breaks inside frame {ctr} between "
                    f"t={t0:.2f}s and t={t1:.2f}s (local edit suspected)"
                )
            if i is not None:
                run_start = prev = i
        return spans, issues

    # ------------------------------------------------------------ validator
    def _validator(self, ctr: int):
        """AEAD open + magic + counter match: the final decode arbiter."""

        def validate(payload: bytes) -> bool:
            try:
                plain = self.sec.open(payload)
            except Exception:
                return False
            return plain[:4] == b"ESAL" and int.from_bytes(plain[4:8], "big") == ctr

        return validate
