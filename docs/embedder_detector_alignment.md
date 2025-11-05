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
CRC-aided polar coding, and encryption. No additional drift was found during
this review.
