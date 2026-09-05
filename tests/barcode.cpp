#include "spiralos/barcode.hpp"

#include <iostream>

#define CHECK(expression) do { if (!(expression)) { std::cerr << "CHECK failed: " #expression << " @ " << __FILE__ << ':' << __LINE__ << '\n'; return 1; } } while (false)

int main() {
    spiralos::AuthorityBarcodePayload payload;
    payload.receipt_id = "guff:authority:sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    payload.envelope_sha256 = "1111111111111111111111111111111111111111111111111111111111111111";
    payload.purpose = spiralos::BarcodePurpose::Recovery;
    payload.subject_sha256 = "2222222222222222222222222222222222222222222222222222222222222222";

    const auto token = spiralos::encode_authority_barcode(payload);
    CHECK(!token.empty());
    CHECK(token.starts_with("SPBC1|"));

    const auto decoded = spiralos::decode_authority_barcode(token);
    CHECK(decoded.ok());
    CHECK(decoded.payload->receipt_id == payload.receipt_id);
    CHECK(decoded.payload->envelope_sha256 == payload.envelope_sha256);
    CHECK(decoded.payload->purpose == payload.purpose);
    CHECK(decoded.payload->subject_sha256 == payload.subject_sha256);

    const auto modules = spiralos::render_barcode_modules(token);
    CHECK(!modules.empty());
    const auto scanned = spiralos::scan_barcode_modules(modules);
    CHECK(scanned.ok());
    CHECK(scanned.payload->receipt_id == payload.receipt_id);

    auto corrupted_modules = modules;
    corrupted_modules[12] = !corrupted_modules[12];
    CHECK(!spiralos::scan_barcode_modules(corrupted_modules).ok());

    auto corrupted_token = token;
    corrupted_token[10] = corrupted_token[10] == '0' ? '1' : '0';
    CHECK(spiralos::decode_authority_barcode(corrupted_token).status ==
          spiralos::BarcodeDecodeStatus::ChecksumMismatch);

    auto invalid_receipt = payload;
    invalid_receipt.receipt_id = "not-authority";
    CHECK(spiralos::encode_authority_barcode(invalid_receipt).empty());

    return 0;
}
