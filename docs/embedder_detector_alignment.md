# Embedder/Detector Alignment Review

This note double-checks the transmit (embedder) and receive (detector) stages after
recent fixes so we can confirm that filtering, spreading, channel coding, and
crypto handling remain synchronized.

## Shared framing and spreading
- The embedder caches the preamble MLS, header PN mask, and per-frame payload PN
  so every frame is assembled as `63 + 128 + 1024 = 1215` chips before filtering
  and raises if any slice drifts from that length. 【F:rtwm/embedder.py†L29-L127】
- The detector initializes with the same preamble MLS and header PN sequence,
  validating the cached length, and slices the payload PN from the same
  frame-length generator when evaluating the nominal despreading variant. 【F:rtwm/detector.py†L27-L342】

## Filtering model
- Transmit filtering starts from a zero state over the preamble and then carries
  the IIR state into the header and payload, matching how the signal will appear
  on the wire. 【F:rtwm/embedder.py†L136-L151】
- The detector mirrors this by building matched-filter taps from the cascaded
  (TX ∘ RX) impulse response, prefilling each alignment window with the
  preamble/header tail so the filter history matches what the embedder emitted.
  【F:rtwm/detector.py†L55-L515】

## Polar coding and CRC
- The embedder calls `polar_fast.encode` with `N=1024`, `K=448`, so the payload
  (55 bytes = 440 bits) is extended with the CRC-8 that `PolarCode` adds before
  mapping to BPSK symbols. 【F:rtwm/embedder.py†L95-L123】【F:rtwm/polar_fast.py†L1-L73】
- The detector feeds the resulting LLRs through the list decoder with the same
  `(N, K)` parameters and lets the CRC gate the candidate that survives the
  validator hook. 【F:rtwm/detector.py†L154-L207】【F:rtwm/polar_fast.py†L39-L73】

## Crypto and session checks
- Frames embed a 27-byte plaintext (`ESAL‖ctr‖session_nonce‖padding`) that is
  sealed by `SecureChannel.seal`, guaranteeing per-frame AEAD coverage. 【F:rtwm/embedder.py†L153-L168】
- On receive we attempt to open the blob with the same channel, retry both nonce
  placements for legacy layouts, and fall back to legacy plaintext frames before
  validating the session nonce monotonicity. 【F:rtwm/detector.py†L177-L233】

## Conclusion
With the cached sequences, cascaded matched-filter taps, and shared polar/AEAD
parameters in place, the embedder and detector now agree on filtering, spreading,
CRC-aided polar coding, and encryption. The new regression tests guard the shared
PN sequences and verify that the embedder's frame filtering matches the documented
specification, so future drift in those critical assumptions will surface quickly.
【F:tests/test_embedder_detector_alignment.py†L1-L70】

## Revalidation checklist (after alignment fixes)
- **Filtering continuity:** Verified that transmit filtering keeps a continuous
  IIR state from the preamble through the payload and that detection aligns with
  the cascaded response before matched filtering, matching the expectations laid
  out after the earlier regression fixes. 【F:rtwm/embedder.py†L136-L151】【F:rtwm/detector.py†L42-L515】
- **Polar + CRC parity:** Confirmed the embedder still emits a CRC-aided
  `N=1024` polar codeword per frame and the detector hands the same length of LLRs
  into the list decoder with the validator hook that enforces the CRC and frame
  counter checks. 【F:rtwm/embedder.py†L95-L123】【F:rtwm/detector.py†L154-L207】
- **Crypto/session flow:** Re-checked that frames seal the `ESAL‖ctr‖nonce` body
  with `SecureChannel` and that detection validates both nonce layouts before
  asserting the session-level replay guard. 【F:rtwm/embedder.py†L153-L168】【F:rtwm/detector.py†L177-L233】
- **Shared PN sequences:** Ensured cached PN slices match across modules and the
  regression test exercises representative counters to prevent reintroduction of
  the drift caught in earlier PRs. 【F:rtwm/embedder.py†L29-L134】【F:rtwm/detector.py†L27-L342】【F:tests/test_embedder_detector_alignment.py†L1-L70】

## Regression test environment note
- The alignment regression tests rely on NumPy and SciPy to build and compare
  the filtered chip streams. They now call `pytest.importorskip` for those
  dependencies so continuous-integration runs without the scientific stack
  report a skipped test instead of a hard error. This keeps the suite green in
  minimal environments while still exercising the checks whenever the optional
  packages are available.【F:tests/test_embedder_detector_alignment.py†L1-L70】

## Follow-up PR timeline
The repeated reviews resulted in a series of narrowly scoped patches. The list
below documents why each PR existed, what changed, and how those adjustments
surface in the current code base.

1. **Cold-start matched filter fix.** The detector now preserves the IIR tail
   by prefilling the payload matched-filter window with the preamble/header
   tail and widening the shift search so despreading can recover the proper
   chip phase.【F:rtwm/detector.py†L300-L374】
2. **Crypto fallback and embedder/detector audit.** A cached AEAD handle lets
   the detector transparently retry both nonce layouts before giving up, while
   the embedder continues to seal the same `ESAL‖ctr‖nonce` payload that the
   detector authenticates.【F:rtwm/embedder.py†L153-L168】【F:rtwm/detector.py†L177-L233】【F:rtwm/detector.py†L389-L431】
3. **Cached PN sequences and frame guards.** Both modules precompute the
   preamble/header PN symbols, slice the payload PN from the same generator, and
   assert the 63 + 128 + 1024 chip structure each frame so drift is caught at
   runtime.【F:rtwm/embedder.py†L26-L121】【F:rtwm/detector.py†L26-L119】
4. **Cascaded filtering alignment.** Correlation templates and matched-filter
   taps are now derived from the transmit∘receive cascade, matching what the
   embedder actually emits before despreading.【F:rtwm/detector.py†L48-L76】【F:rtwm/detector.py†L242-L313】
5. **Regression coverage.** The new regression test confirms the embedder and
   detector reuse the exact PN sequences and filtering pipeline documented
   above.【F:tests/test_embedder_detector_alignment.py†L1-L70】
6. **Test guards for optional dependencies.** Import guards ensure the
   alignment tests skip cleanly when NumPy/SciPy are absent so minimal CI stays
   green without masking regressions when those packages are installed.【F:tests/test_embedder_detector_alignment.py†L1-L70】
7. **Documentation clarity.** This note now records the shared assumptions and
   validates them against the code so reviewers can verify future fixes without
   reopening the same questions.【F:docs/embedder_detector_alignment.md†L1-L118】

With these stages complete, the current implementation keeps the embedder and
detector synchronized across filtering, spreading, polar coding, and crypto. No
further code changes are required for the alignment review; this PR exists only
to capture the rationale behind the earlier sequence of fixes.
