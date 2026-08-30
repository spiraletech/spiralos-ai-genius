$ErrorActionPreference = 'Stop'

# HF3 conversational routing + robust native clock routing.
$p = 'src/ether_ai.cpp'
$s = Get-Content $p -Raw
$dateOld = @'
bool asks_for_date_or_day(std::string_view text) {
    const std::string value = lower(std::string(text));
    return value == "what day is it" || value == "what day is it?" || value == "what date is it" || value == "what date is it?" ||
           value.find("what's the date") != std::string::npos || value.find("whats the date") != std::string::npos;
}
'@
$dateNew = @'
bool asks_for_date_or_day(std::string_view text) {
    std::string value = lower(std::string(text));
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return false;
    value.erase(0, first);
    const auto last = value.find_last_not_of(" \t\r\n");
    if (last != std::string::npos) value.erase(last + 1);
    while (!value.empty() && (value.back() == '?' || value.back() == '!' || value.back() == '.')) value.pop_back();
    return value.find("what day is it") != std::string::npos ||
           value.find("what date is it") != std::string::npos ||
           value.find("what's the date") != std::string::npos ||
           value.find("whats the date") != std::string::npos ||
           value.find("today's date") != std::string::npos ||
           value.find("todays date") != std::string::npos;
}
'@
if (-not $s.Contains($dateOld.Trim())) { throw 'date router target not found' }
$s = $s.Replace($dateOld.Trim(), $dateNew.Trim())

$fallbackOld = @'
    } else {
        (void)shell_.organic_mind_mutable().respond(visible, context_for(host_));
        reply = limited_mode_reply();
        shell_.organic_mind_mutable().adopt_reply(reply);
        std::string ignored;
        (void)shell_.save_organic_state(&ignored);
    }
'@
$fallbackNew = @'
    } else {
        const auto organic_response = shell_.organic_mind_mutable().respond(visible, context_for(host_));
        const std::string& organic_reply = organic_response.text;
        const bool needs_cortex = organic_reply.empty() ||
            organic_reply.find("LANGUAGE CORTEX OFFLINE / LIMITED MODE") != std::string::npos ||
            organic_reply.find("not enough learned knowledge") != std::string::npos ||
            organic_reply.find("need the trained Spiral language cortex") != std::string::npos;
        reply = needs_cortex ? limited_mode_reply() : organic_reply;
        shell_.organic_mind_mutable().adopt_reply(reply);
        std::string ignored;
        (void)shell_.save_organic_state(&ignored);
    }
'@
if (-not $s.Contains($fallbackOld.Trim())) { throw 'fallback target not found' }
$s = $s.Replace($fallbackOld.Trim(), $fallbackNew.Trim())
Set-Content $p $s -NoNewline -Encoding utf8

# Keep local small-talk useful without pretending it is the trained cortex.
$p = 'src/organic_ai.cpp'
$s = Get-Content $p -Raw
$legacy = 'I don''t have enough native learned knowledge to answer that reliably yet. '
if ($s.Contains($legacy)) {
    $s = $s.Replace($legacy, 'LANGUAGE CORTEX OFFLINE / LIMITED MODE. This request needs either native grounding, a connected XENON tool, or a trained language cortex. ')
}
$greetOld = 'if (contains_word(tokens, "hello") || contains_word(tokens, "hey") || contains_word(tokens, "hi") || tokens.front() == "yo") {'
$greetNew = 'if (contains_word(tokens, "hello") || contains_word(tokens, "hey") || contains_word(tokens, "hi") || contains_word(tokens, "sup") || tokens.front() == "yo" || lower.find("what''s up") != std::string::npos || lower.find("whats up") != std::string::npos) {'
if ($s.Contains($greetOld)) { $s = $s.Replace($greetOld, $greetNew) }
Set-Content $p $s -NoNewline -Encoding utf8

# L27E: resolve the bundled llama-cli beside SpiralEtherAI.exe and tune inference.
$p = 'src/xenon_os.cpp'
$s = Get-Content $p -Raw
$anchor = 'std::filesystem::path prompt_path() {'
$helper = @'
#ifdef _WIN32
std::filesystem::path executable_directory() {
    std::vector<char> buffer(32768, '\0');
    const DWORD size = GetModuleFileNameA(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (size == 0 || size >= buffer.size()) return {};
    return std::filesystem::path(std::string(buffer.data(), size)).parent_path();
}
#endif

std::filesystem::path prompt_path() {
'@
if (-not $s.Contains($anchor)) { throw 'prompt helper target not found' }
$s = $s.Replace($anchor, $helper.Trim())
$runtimeOld = 'if (runtime_path.empty()) runtime_path = "llama-cli.exe";'
$runtimeNew = @'
if (runtime_path.empty()) {
#ifdef _WIN32
            const auto bundled = executable_directory() / "llama-cli.exe";
            runtime_path = std::filesystem::exists(bundled) ? bundled.string() : "llama-cli.exe";
#else
            runtime_path = "llama-cli";
#endif
        }
'@
if (-not $s.Contains($runtimeOld)) { throw 'runtime discovery target not found' }
$s = $s.Replace($runtimeOld, $runtimeNew.Trim())
$samplingOld = '<< " --top-k 40 --top-p 0.90 --min-p 0.05 --repeat-penalty 1.08 --no-display-prompt";'
$samplingNew = '<< " -c 4096 --top-k 40 --top-p 0.90 --min-p 0.05 --repeat-penalty 1.08 --no-display-prompt";'
if (-not $s.Contains($samplingOld)) { throw 'sampling target not found' }
$s = $s.Replace($samplingOld, $samplingNew)
$promptOld = '<< "SYSTEM: For a tool request, emit exactly one line beginning TOOL_CALL followed by host.action and optional key=value arguments. Do not claim success before a ToolResult.\n";'
$promptNew = @'
<< "SYSTEM: For a tool request, emit exactly one line beginning TOOL_CALL followed by host.action and optional key=value arguments. Do not claim success before a ToolResult.\n"
           << "SYSTEM: Tool namespaces available through XENON are hakui.*, etherplay.*, and etherbeat.*. Never invent a tool outside these namespaces.\n"
           << "SYSTEM: For ordinary conversation, answer normally; do not emit TOOL_CALL unless execution or live host state is actually needed.\n";
'@
if (-not $s.Contains($promptOld)) { throw 'prompt policy target not found' }
$s = $s.Replace($promptOld, $promptNew.Trim())
Set-Content $p $s -NoNewline -Encoding utf8

# Regression expectations: greetings remain conversational without a GGUF.
$p = 'tests/ether_ai_tests.cpp'
$s = Get-Content $p -Raw
$testOld = @'
    if (runtime.backend() == genius::GptBackend::Auto) {
        const std::string limited = runtime.send("hello spiral");
        assert(limited.find("LANGUAGE CORTEX: OFFLINE / LIMITED MODE") != std::string::npos);
        assert(runtime.status().shell.organic_turns == 1);
        assert(runtime.status().shell.organic_memories >= 1);
    }
'@
$testNew = @'
    if (runtime.backend() == genius::GptBackend::Auto) {
        const std::string hello = runtime.send("hello spiral");
        assert(hello.find("LANGUAGE CORTEX: OFFLINE / LIMITED MODE") == std::string::npos);
        assert(hello.find("Organic mode is running locally") != std::string::npos);
        assert(runtime.status().shell.organic_turns == 1);
        assert(runtime.status().shell.organic_memories >= 1);
    }
'@
if (-not $s.Contains($testOld.Trim())) { throw 'fallback test target not found' }
$s = $s.Replace($testOld.Trim(), $testNew.Trim())
Set-Content $p $s -NoNewline -Encoding utf8

# GUI identity + prior chat fixes + GGUF drag-load.
$p = 'apps/ether_ai/main.cpp'
$s = Get-Content $p -Raw
$s = $s.Replace('L"Spiral Ether AI",', 'L"Spiral Ether AI — L27E Cortex",')
$s = $s.Replace('L"Spiral Ether AI online. Native organic cognition is active. The same C++ runtime can inhabit Windows, EtherPlay, Hakui, EtherBeat, or another Spiral host."', 'L"Spiral Ether AI L27E-20260830 online. Bundled llama.cpp cortex runtime ready; drag an instruct GGUF to activate generative intelligence."')
$pairs = @(
    @('float measure_text(const std::wstring& text,IDWriteTextFormat* format,float width) const { ComPtr<IDWriteTextLayout> l; if(FAILED(write_factory_->CreateTextLayout(text.c_str(),static_cast<UINT32>(text.size()),format,std::max(1.0F,width),4000.0F,l.GetAddressOf())))return 24.0F; DWRITE_TEXT_METRICS m{}; if(FAILED(l->GetMetrics(&m)))return 24.0F; return m.height; }', 'float measure_text(const std::wstring& text,IDWriteTextFormat* format,float width) const { ComPtr<IDWriteTextLayout> l; if(FAILED(write_factory_->CreateTextLayout(text.c_str(),static_cast<UINT32>(text.size()),format,std::max(1.0F,width),4000.0F,l.GetAddressOf())))return 28.0F; DWRITE_TEXT_METRICS m{}; if(FAILED(l->GetMetrics(&m)))return 28.0F; DWRITE_OVERHANG_METRICS o{}; (void)l->GetOverhangMetrics(&o); return std::ceil(m.height + std::max(0.0F,o.bottom) + 8.0F); }'),
    @('for(const auto& m:messages_){const float bh=measure_text(m.text,body_format_.Get(),tw)+32.0F;heights.push_back(bh);content_height_+=bh+14.0F;}', 'for(const auto& m:messages_){const float bh=measure_text(m.text,body_format_.Get(),tw)+50.0F;heights.push_back(bh);content_height_+=bh+14.0F;}'),
    @('target_->PushAxisAlignedClip(clip,D2D1_ANTIALIAS_MODE_PER_PRIMITIVE); float y=clip.top+12.0F-scroll_offset_;', 'target_->PushAxisAlignedClip(clip,D2D1_ANTIALIAS_MODE_PER_PRIMITIVE); const float inner_content_height=std::max(0.0F,content_height_-12.0F); const float bottom_anchor=std::max(0.0F,chat_view_height_-inner_content_height); float y=clip.top+12.0F+bottom_anchor-scroll_offset_;'),
    @('draw_text(m.text,body_format_.Get(),text_brush_.Get(),D2D1::RectF(bubble.left+15,bubble.top+27,bubble.right-15,bubble.bottom-10));', 'draw_text(m.text,body_format_.Get(),text_brush_.Get(),D2D1::RectF(bubble.left+15,bubble.top+31,bubble.right-15,bubble.bottom-14));'),
    @('if(ext==L".bundle"){std::string error;if(runtime_.load_local_model(wide_to_utf8(dropped),&error))messages_.push_back(UiMessage{false,L"Local Spiral model loaded: "+dropped});', 'if(ext==L".bundle"||ext==L".gguf"){std::string error;if(runtime_.load_local_model(wide_to_utf8(dropped),&error))messages_.push_back(UiMessage{false,L"L27E language cortex loaded: "+dropped});')
)
foreach ($pair in $pairs) { if ($s.Contains($pair[0])) { $s = $s.Replace($pair[0], $pair[1]) } }
$wheelOld = 'if (self != nullptr && msg == WM_KEYDOWN && wparam == VK_ESCAPE) {'
if ($s.Contains($wheelOld) -and -not $s.Contains('msg == WM_MOUSEWHEEL')) {
    $s = $s.Replace($wheelOld, 'if (self != nullptr && msg == WM_MOUSEWHEEL) { SendMessageW(self->hwnd_, WM_MOUSEWHEEL, wparam, lparam); return 0; }' + [Environment]::NewLine + '        ' + $wheelOld)
}
Set-Content $p $s -NoNewline -Encoding utf8
