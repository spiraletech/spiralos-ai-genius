#include "spiralos/barcode.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <iomanip>
#include <sstream>

namespace spiralos {
namespace {

constexpr std::string_view kPrefix = "SPBC1";
constexpr std::array<bool, 8> kStart{{true, false, true, true, false, true, false, true}};
constexpr std::array<bool, 8> kEnd{{true, true, false, true, false, false, true, true}};
constexpr std::string_view kReceiptPrefix = "guff:authority:sha256:";

bool is_hex64(std::string_view value) noexcept {
    if (value.size() != 64U) return false;
    return std::all_of(value.begin(), value.end(), [](unsigned char ch) {
        return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') ||
               (ch >= 'A' && ch <= 'F');
    });
}

bool valid_receipt_id(std::string_view value) noexcept {
    return value.starts_with(kReceiptPrefix) && is_hex64(value.substr(kReceiptPrefix.size()));
}

std::uint32_t crc32(std::string_view data) noexcept {
    std::uint32_t crc = 0xffffffffU;
    for (const auto ch : data) {
        crc ^= static_cast<std::uint8_t>(ch);
        for (int i = 0; i < 8; ++i) {
            const auto mask = static_cast<std::uint32_t>(-(static_cast<int>(crc & 1U)));
            crc = (crc >> 1U) ^ (0xedb88320U & mask);
        }
    }
    return ~crc;
}

std::string crc_hex(std::string_view data) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(8) << crc32(data);
    return out.str();
}

std::vector<std::string_view> split(std::string_view text, char delimiter) {
    std::vector<std::string_view> fields;
    std::size_t start = 0U;
    while (start <= text.size()) {
        const auto next = text.find(delimiter, start);
        if (next == std::string_view::npos) {
            fields.push_back(text.substr(start));
            break;
        }
        fields.push_back(text.substr(start, next - start));
        start = next + 1U;
    }
    return fields;
}

bool parse_u32(std::string_view text, std::uint32_t* out) noexcept {
    if (!out || text.empty()) return false;
    std::uint32_t value{};
    const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || ptr != text.data() + text.size()) return false;
    *out = value;
    return true;
}

bool even_parity(std::uint8_t value) noexcept {
    unsigned ones = 0U;
    for (int bit = 0; bit < 8; ++bit) ones += (value >> bit) & 1U;
    return (ones % 2U) == 0U;
}

BarcodeDecodeResult fail(BarcodeDecodeStatus status, std::string error) {
    return {status, std::nullopt, std::move(error)};
}

} // namespace

bool BarcodeDecodeResult::ok() const noexcept {
    return status == BarcodeDecodeStatus::Ok && payload.has_value();
}

std::string encode_authority_barcode(const AuthorityBarcodePayload& payload) {
    if (payload.schema_version != 1U || !valid_receipt_id(payload.receipt_id) ||
        !is_hex64(payload.envelope_sha256) || !is_hex64(payload.subject_sha256)) {
        return {};
    }

    std::ostringstream body;
    body << kPrefix << '|'
         << payload.schema_version << '|'
         << static_cast<unsigned>(payload.purpose) << '|'
         << payload.receipt_id << '|'
         << payload.envelope_sha256 << '|'
         << payload.subject_sha256;
    const auto text = body.str();
    return text + '|' + crc_hex(text);
}

BarcodeDecodeResult decode_authority_barcode(std::string_view token) {
    const auto fields = split(token, '|');
    if (fields.empty() || fields.front() != kPrefix)
        return fail(BarcodeDecodeStatus::InvalidPrefix, "missing SPBC1 prefix");
    if (fields.size() != 7U)
        return fail(BarcodeDecodeStatus::InvalidFieldCount, "SPBC1 requires seven fields");

    const auto checksum_offset = token.rfind('|');
    if (checksum_offset == std::string_view::npos)
        return fail(BarcodeDecodeStatus::InvalidEncoding, "checksum separator missing");
    const auto body = token.substr(0U, checksum_offset);
    if (fields[6] != crc_hex(body))
        return fail(BarcodeDecodeStatus::ChecksumMismatch, "CRC32 mismatch");

    AuthorityBarcodePayload payload;
    if (!parse_u32(fields[1], &payload.schema_version) || payload.schema_version != 1U)
        return fail(BarcodeDecodeStatus::InvalidEncoding, "unsupported barcode schema");

    std::uint32_t purpose{};
    if (!parse_u32(fields[2], &purpose) || purpose > static_cast<unsigned>(BarcodePurpose::CapabilityGrant))
        return fail(BarcodeDecodeStatus::InvalidEncoding, "invalid barcode purpose");
    payload.purpose = static_cast<BarcodePurpose>(purpose);
    payload.receipt_id = std::string(fields[3]);
    payload.envelope_sha256 = std::string(fields[4]);
    payload.subject_sha256 = std::string(fields[5]);

    if (!valid_receipt_id(payload.receipt_id))
        return fail(BarcodeDecodeStatus::InvalidReceiptId, "receipt id is not canonical");
    if (!is_hex64(payload.envelope_sha256) || !is_hex64(payload.subject_sha256))
        return fail(BarcodeDecodeStatus::InvalidDigest, "barcode digest field is invalid");

    return {BarcodeDecodeStatus::Ok, std::move(payload), {}};
}

std::vector<bool> render_barcode_modules(std::string_view token) {
    if (token.empty() || token.size() > 4096U) return {};
    std::vector<bool> modules;
    modules.reserve(kStart.size() + token.size() * 9U + kEnd.size());
    modules.insert(modules.end(), kStart.begin(), kStart.end());
    for (const auto ch : token) {
        const auto byte = static_cast<std::uint8_t>(ch);
        for (int bit = 7; bit >= 0; --bit) modules.push_back(((byte >> bit) & 1U) != 0U);
        modules.push_back(!even_parity(byte));
    }
    modules.insert(modules.end(), kEnd.begin(), kEnd.end());
    return modules;
}

BarcodeDecodeResult scan_barcode_modules(const std::vector<bool>& modules) {
    if (modules.size() < kStart.size() + kEnd.size() + 9U)
        return fail(BarcodeDecodeStatus::ModuleFramingError, "module stream too short");
    if (!std::equal(kStart.begin(), kStart.end(), modules.begin()))
        return fail(BarcodeDecodeStatus::ModuleFramingError, "start sentinel mismatch");
    if (!std::equal(kEnd.rbegin(), kEnd.rend(), modules.rbegin()))
        return fail(BarcodeDecodeStatus::ModuleFramingError, "end sentinel mismatch");

    const auto payload_modules = modules.size() - kStart.size() - kEnd.size();
    if ((payload_modules % 9U) != 0U)
        return fail(BarcodeDecodeStatus::ModuleFramingError, "module payload is not byte aligned");

    std::string token;
    token.reserve(payload_modules / 9U);
    std::size_t offset = kStart.size();
    while (offset < modules.size() - kEnd.size()) {
        std::uint8_t value = 0U;
        for (int bit = 0; bit < 8; ++bit) {
            value = static_cast<std::uint8_t>((value << 1U) | (modules[offset++] ? 1U : 0U));
        }
        const bool parity = modules[offset++];
        if (parity != !even_parity(value))
            return fail(BarcodeDecodeStatus::ModuleParityError, "per-byte parity mismatch");
        token.push_back(static_cast<char>(value));
    }
    return decode_authority_barcode(token);
}

std::string_view to_string(BarcodePurpose purpose) noexcept {
    switch (purpose) {
    case BarcodePurpose::Recovery: return "RECOVERY";
    case BarcodePurpose::DestructiveExecution: return "DESTRUCTIVE_EXECUTION";
    case BarcodePurpose::PersistentSymbiosis: return "PERSISTENT_SYMBIOSIS";
    case BarcodePurpose::CapabilityGrant: return "CAPABILITY_GRANT";
    }
    return "CAPABILITY_GRANT";
}

std::string_view to_string(BarcodeDecodeStatus status) noexcept {
    switch (status) {
    case BarcodeDecodeStatus::Ok: return "OK";
    case BarcodeDecodeStatus::InvalidPrefix: return "INVALID_PREFIX";
    case BarcodeDecodeStatus::InvalidFieldCount: return "INVALID_FIELD_COUNT";
    case BarcodeDecodeStatus::InvalidEncoding: return "INVALID_ENCODING";
    case BarcodeDecodeStatus::ChecksumMismatch: return "CHECKSUM_MISMATCH";
    case BarcodeDecodeStatus::InvalidReceiptId: return "INVALID_RECEIPT_ID";
    case BarcodeDecodeStatus::InvalidDigest: return "INVALID_DIGEST";
    case BarcodeDecodeStatus::ModuleFramingError: return "MODULE_FRAMING_ERROR";
    case BarcodeDecodeStatus::ModuleParityError: return "MODULE_PARITY_ERROR";
    }
    return "INVALID_ENCODING";
}

} // namespace spiralos
