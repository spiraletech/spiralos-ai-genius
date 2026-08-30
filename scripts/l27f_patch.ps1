$ErrorActionPreference = 'Stop'

# Start from the proven L27E/HF3 runtime and UI fixes.
& "$PSScriptRoot/l27e_patch.ps1"

# L27F: auto-discover the bundled GGUF beside the executable and prefer it over cloud fallback.
$p = 'src/ether_ai.cpp'
$s = Get-Content $p -Raw

$includeAnchor = '#include <utility>'
$includePatch = @'
#include <utility>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
'@
if ($s.Contains($includeAnchor) -and -not $s.Contains('#include <windows.h>')) {
    $s = $s.Replace($includeAnchor, $includePatch.Trim())
}

$helperAnchor = @'
std::string default_organic_path(const HostDescriptor& host, std::string requested) {
    if (!requested.empty()) return requested;
    if (host.kind == HostKind::StandaloneWindows || host.kind == HostKind::XenonOS) return "SpiralEtherAI.organic";
    return {};
}
'@
$helperPatch = @'
std::string default_organic_path(const HostDescriptor& host, std::string requested) {
    if (!requested.empty()) return requested;
    if (host.kind == HostKind::StandaloneWindows || host.kind == HostKind::XenonOS) return "SpiralEtherAI.organic";
    return {};
}

std::filesystem::path runtime_directory() {
#ifdef _WIN32
    std::vector<char> buffer(32768, '\0');
    const DWORD size = GetModuleFileNameA(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (size > 0 && size < buffer.size()) return std::filesystem::path(std::string(buffer.data(), size)).parent_path();
#endif
    std::error_code ec;
    const auto cwd = std::filesystem::current_path(ec);
    return ec ? std::filesystem::path{} : cwd;
}

std::filesystem::path bundled_cortex_path() {
    const auto root = runtime_directory();
    if (root.empty()) return {};
    const std::filesystem::path direct = root / "SmolLM2-135M-Instruct-Q4_K_M.gguf";
    if (std::filesystem::exists(direct)) return direct;
    const std::filesystem::path named = root / "SpiralCortex.gguf";
    if (std::filesystem::exists(named)) return named;
    return {};
}
'@
if (-not $s.Contains($helperAnchor.Trim())) { throw 'L27F runtime directory anchor not found' }
$s = $s.Replace($helperAnchor.Trim(), $helperPatch.Trim())

$ctorOld = @'
    const char* configured_model = std::getenv("SPIRAL_MODEL_PATH");
    if (configured_model != nullptr && *configured_model != '\0') {
        std::string ignored;
        if (local_cortex_.configure_gguf(configured_model, {}, &ignored)) shell_.set_gpt_backend(genius::GptBackend::SpiralLocal);
    } else {
        const auto startup = shell_.status();
        if (startup.openai_platform_supported && startup.openai_key_present) shell_.set_gpt_backend(genius::GptBackend::OpenAI);
    }
'@
$ctorNew = @'
    std::string startup_model;
    const char* configured_model = std::getenv("SPIRAL_MODEL_PATH");
    if (configured_model != nullptr && *configured_model != '\0') {
        startup_model = configured_model;
    } else {
        const auto bundled = bundled_cortex_path();
        if (!bundled.empty()) startup_model = bundled.string();
    }

    if (!startup_model.empty()) {
        std::string error;
        if (local_cortex_.configure_gguf(startup_model, {}, &error)) {
            shell_.set_gpt_backend(genius::GptBackend::SpiralLocal);
        }
    }

    if (shell_.gpt_backend() != genius::GptBackend::SpiralLocal) {
        const auto startup = shell_.status();
        if (startup.openai_platform_supported && startup.openai_key_present) shell_.set_gpt_backend(genius::GptBackend::OpenAI);
    }
'@
if (-not $s.Contains($ctorOld.Trim())) { throw 'L27F constructor model bootstrap target not found' }
$s = $s.Replace($ctorOld.Trim(), $ctorNew.Trim())

$limitedOld = 'return "LANGUAGE CORTEX: OFFLINE / LIMITED MODE. ORGANIC memory, Liratel state, XENON tools, and native runtime grounding are online, but no trained GGUF language cortex is loaded. Drop a compatible instruct .gguf onto this window (with llama-cli configured) for real generative conversation. I will not disguise the handcrafted fallback as GPT-quality output.";'
$limitedNew = 'return "CORTEX OFFLINE. ORGANIC is still online, but the bundled language model could not be loaded.";'
if (-not $s.Contains($limitedOld)) { throw 'L27F concise offline reply target not found' }
$s = $s.Replace($limitedOld, $limitedNew)
Set-Content $p $s -NoNewline -Encoding utf8

# SmolLM2 uses a ChatML-style instruct template. Make that explicit when detected by filename.
$p = 'src/xenon_os.cpp'
$s = Get-Content $p -Raw
$templateAnchor = 'if (name.find("qwen") != std::string::npos) return "chatml";'
$templatePatch = @'
if (name.find("qwen") != std::string::npos) return "chatml";
    if (name.find("smollm") != std::string::npos) return "chatml";
'@
if (-not $s.Contains($templateAnchor)) { throw 'L27F chat template target not found' }
$s = $s.Replace($templateAnchor, $templatePatch.Trim())
Set-Content $p $s -NoNewline -Encoding utf8

# L27F identity: this package has both runtime and model weights.
$p = 'apps/ether_ai/main.cpp'
$s = Get-Content $p -Raw
$s = $s.Replace('L"Spiral Ether AI — L27E Cortex",', 'L"Spiral Ether AI — L27F Self-Contained Cortex",')
$s = $s.Replace('L"Spiral Ether AI L27E-20260830 online. Bundled llama.cpp cortex runtime ready; drag an instruct GGUF to activate generative intelligence."', 'L"Spiral Ether AI L27F-20260830 online. Local language cortex model and llama.cpp runtime are bundled and auto-loaded."')
$s = $s.Replace('L"L27E language cortex loaded: "+dropped', 'L"L27F language cortex loaded: "+dropped')
Set-Content $p $s -NoNewline -Encoding utf8
