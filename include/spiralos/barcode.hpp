#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace spiralos {

enum class BarcodePurpose : std::uint8_t {
    Recovery,
    DestructiveExecution,
    PersistentSymbiosis,
    CapabilityGrant
};

enum class BarcodeDecodeStatus : std::uint8_t {
    Ok,
    InvalidPrefix,
    InvalidFieldCount,
    InvalidEncoding,
    ChecksumMismatch,
    InvalidReceiptId,
    InvalidDigest,
    ModuleFramingError,
    ModuleParityError
};

struct AuthorityBarcodePayload {
    std::uint32_t schema_version{1U};
    std::string receipt_id;
    std::string envelope_sha256;
    BarcodePurpose purpose{BarcodePurpose::CapabilityGrant};
    std::string subject_sha256;
};

struct BarcodeDecodeResult {
    BarcodeDecodeStatus status{BarcodeDecodeStatus::InvalidEncoding};
    std::optional<AuthorityBarcodePayload> payload;
    std::string error;

    [[nodiscard]] bool ok() const noexcept;
};

// Spiral Barcode v1 is a transport encoding, not an authentication primitive.
// Callers MUST resolve and cryptographically verify the referenced authority receipt.
[[nodiscard]] std::string encode_authority_barcode(const AuthorityBarcodePayload& payload);
[[nodiscard]] BarcodeDecodeResult decode_authority_barcode(std::string_view token);

// Deterministic one-dimensional module representation for rendering/scanning.
// false = space, true = bar. Uses start/end sentinels and per-byte parity.
[[nodiscard]] std::vector<bool> render_barcode_modules(std::string_view token);
[[nodiscard]] BarcodeDecodeResult scan_barcode_modules(const std::vector<bool>& modules);

[[nodiscard]] std::string_view to_string(BarcodePurpose purpose) noexcept;
[[nodiscard]] std::string_view to_string(BarcodeDecodeStatus status) noexcept;

} // namespace spiralos
