# SpiralOS AI Genius

## Spiral Barcode v1

SpiralOS now has a native authority-barcode transport designed to pair with GOLF GUFF L15 authority receipts.

A barcode carries only public proof metadata:

- canonical `guff:authority:sha256:<digest>` receipt ID
- authority-envelope SHA-256
- authority purpose
- subject SHA-256
- CRC32 transport checksum

`render_barcode_modules()` converts the token into deterministic one-dimensional bar/space modules with start/end sentinels and per-byte parity. `scan_barcode_modules()` validates framing/parity and then decodes the authority token.

### Security law

**The barcode is not authority.** CRC32 and parity detect transport corruption only. SpiralOS must resolve the referenced GOLF GUFF authority receipt and cryptographically verify its signer, digest, purpose and subject before authorizing an operation.

```text
USER / SIGNER
    |
    v
GOLF GUFF AUTHORITY RECEIPT
    |
 receipt id + public digests
    v
SPIRAL BARCODE v1
    |
 scan / parity / CRC
    v
RECEIPT RESOLUTION
    |
 signer + scope verification
    v
ALLOW / REFUSE
```

This makes the barcode useful as a physical/UI token, badge, cartridge label, recovery receipt, QR-adjacent visual object or cross-device handoff without embedding private signing material or turning an image scan into implicit execution authority.
