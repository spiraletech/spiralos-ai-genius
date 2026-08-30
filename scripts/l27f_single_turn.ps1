$ErrorActionPreference = 'Stop'
$p = 'src/xenon_os.cpp'
$s = Get-Content $p -Raw
$old = '<< " -c 4096 --top-k 40 --top-p 0.90 --min-p 0.05 --repeat-penalty 1.08 --no-display-prompt";'
$new = '<< " -c 4096 -st --top-k 40 --top-p 0.90 --min-p 0.05 --repeat-penalty 1.08 --no-display-prompt";'
if (-not $s.Contains($old)) { throw 'L27F single-turn inference target not found' }
$s = $s.Replace($old, $new)
Set-Content $p $s -NoNewline -Encoding utf8
