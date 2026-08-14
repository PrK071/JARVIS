param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("list", "synthesize")]
    [string]$Action,
    [string]$VoiceId,
    [ValidateSet("WinRT", "SAPI")]
    [string]$Interface = "WinRT",
    [string]$RequestJson,
    [Parameter(Mandatory = $true)]
    [string]$ResultJson
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-JsonFile {
    param([object]$Value, [string]$Path)
    $json = $Value | ConvertTo-Json -Depth 8
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $json, $encoding)
}

function Get-SapiLocale {
    param([string]$Language)
    try {
        $lcid = [Convert]::ToInt32($Language, 16)
        return [Globalization.CultureInfo]::GetCultureInfo($lcid).Name
    }
    catch {
        return $Language
    }
}

function Await-WinRtStream {
    param([object]$Operation)
    $method = [System.WindowsRuntimeSystemExtensions].GetMethods() |
        Where-Object {
            $_.Name -eq "AsTask" -and
            $_.IsGenericMethod -and
            $_.GetParameters().Count -eq 1
        } |
        Select-Object -First 1
    $generic = $method.MakeGenericMethod(
        [Windows.Media.SpeechSynthesis.SpeechSynthesisStream]
    )
    $task = $generic.Invoke($null, @($Operation))
    $task.Wait()
    return $task.Result
}

Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.SpeechSynthesis.SpeechSynthesizer,Windows.Media.SpeechSynthesis,ContentType=WindowsRuntime]

if ($Action -eq "list") {
    $sapiError = $null
    $systemSpeechError = $null
    $winrtError = $null
    $sapi = @()
    try {
        $speaker = New-Object -ComObject SAPI.SpVoice
        $tokens = $speaker.GetVoices()
        for ($index = 0; $index -lt $tokens.Count; $index++) {
            $token = $tokens.Item($index)
            $language = $token.GetAttribute("Language")
            $sapi += [pscustomobject]@{
                name = $token.GetDescription()
                id = $token.Id
                locale = Get-SapiLocale $language
                language_attribute = $language
                gender = $token.GetAttribute("Gender")
                age = $token.GetAttribute("Age")
                vendor = $token.GetAttribute("Vendor")
                interface = "SAPI 5"
                offline = $true
                asynchronous = $true
                cancellation = "SVSFPurgeBeforeSpeak"
                output = "WAV/PCM stream"
            }
        }
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject(
            $speaker
        )
    }
    catch {
        $sapiError = $_.Exception.Message
    }

    $systemSpeech = @()
    try {
        Add-Type -AssemblyName System.Speech
        $synthesizer = New-Object System.Speech.Synthesis.SpeechSynthesizer
        foreach ($installed in $synthesizer.GetInstalledVoices()) {
            $info = $installed.VoiceInfo
            $systemSpeech += [pscustomobject]@{
                name = $info.Name
                id = $info.Id
                locale = $info.Culture.Name
                gender = [string]$info.Gender
                age = [string]$info.Age
                description = $info.Description
                enabled = $installed.Enabled
                interface = "System.Speech"
                offline = $true
                asynchronous = $true
                cancellation = "SpeakAsyncCancelAll"
                output = "WAV/PCM stream"
            }
        }
        $synthesizer.Dispose()
    }
    catch {
        $systemSpeechError = $_.Exception.Message
    }

    $winrt = @()
    try {
        foreach (
            $voice in [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices
        ) {
            $winrt += [pscustomobject]@{
                name = [string]$voice.DisplayName
                id = [string]$voice.PSObject.Properties["Id"].Value
                locale = [string]$voice.Language
                gender = [string]$voice.Gender
                description = [string]$voice.Description
                interface = "WinRT"
                offline = $true
                asynchronous = $true
                cancellation = "IAsyncOperation.Cancel"
                output = "SpeechSynthesisStream WAV/PCM"
            }
        }
    }
    catch {
        $winrtError = $_.Exception.Message
    }

    Write-JsonFile @{
        sapi = $sapi
        system_speech = $systemSpeech
        winrt = $winrt
        errors = @{
            sapi = $sapiError
            system_speech = $systemSpeechError
            winrt = $winrtError
        }
    } $ResultJson
    exit 0
}

if (-not $RequestJson) {
    throw "RequestJson obrigatório para síntese"
}
$request = Get-Content -LiteralPath $RequestJson -Raw -Encoding UTF8 |
    ConvertFrom-Json
$outputDirectory = [string]$request.output_directory
[void][System.IO.Directory]::CreateDirectory($outputDirectory)
$metrics = @()
$rate = 0
if ($null -ne $request.PSObject.Properties["rate"]) {
    $rate = [double]$request.rate
}
if ($rate -lt -10 -or $rate -gt 10) {
    throw "rate inválido: $rate; intervalo aceito: -10 a 10"
}
$volume = 100
if ($null -ne $request.PSObject.Properties["volume"]) {
    $volume = [int]$request.volume
}
if ($volume -lt 0 -or $volume -gt 100) {
    throw "volume inválido: $volume; intervalo aceito: 0 a 100"
}

if ($Interface -eq "WinRT") {
    $voice = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices |
        Where-Object { $_.Id -eq $VoiceId } |
        Select-Object -First 1
    if (-not $voice) {
        throw "voz WinRT ausente: $VoiceId"
    }
    $synthesizer = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::new()
    $synthesizer.Voice = $voice
    # SAPI rate steps are logarithmic: one step is the tenth root of 3.
    # Daniel is a OneCore/WinRT-only voice here, so preserve those semantics
    # through WinRT's native pre-synthesis tempo option.
    $sapiRateMultiplier = [Math]::Pow(3.0, $rate / 10.0)
    $appliedSpeakingRate = [Math]::Max(
        0.5,
        [Math]::Min(6.0, $sapiRateMultiplier)
    )
    $synthesizer.Options.SpeakingRate = $appliedSpeakingRate
    $synthesizer.Options.AudioVolume = $volume / 100.0
    foreach ($item in $request.items) {
        $target = Join-Path $outputDirectory ([string]$item.filename)
        $watch = [Diagnostics.Stopwatch]::StartNew()
        $operation = $synthesizer.SynthesizeTextToStreamAsync(
            [string]$item.text
        )
        $stream = Await-WinRtStream $operation
        $inputStream = [System.IO.WindowsRuntimeStreamExtensions]::AsStreamForRead(
            $stream
        )
        $outputStream = [System.IO.File]::Create($target)
        $inputStream.CopyTo($outputStream)
        $outputStream.Dispose()
        $inputStream.Dispose()
        $stream.Dispose()
        $watch.Stop()
        $metrics += [pscustomobject]@{
            filename = [string]$item.filename
            synthesis_seconds = $watch.Elapsed.TotalSeconds
            configured_rate = $rate
            applied_speaking_rate = $appliedSpeakingRate
            volume = $volume
        }
    }
    $synthesizer.Dispose()
}
else {
    if ($rate -ne [Math]::Truncate($rate)) {
        throw "SAPI COM exige rate inteiro; recebido: $rate"
    }
    $speaker = New-Object -ComObject SAPI.SpVoice
    $voice = $speaker.GetVoices() |
        Where-Object { $_.Id -eq $VoiceId } |
        Select-Object -First 1
    if (-not $voice) {
        throw "voz SAPI ausente: $VoiceId"
    }
    $speaker.Voice = $voice
    $speaker.Rate = [int]$rate
    $speaker.Volume = $volume
    foreach ($item in $request.items) {
        $target = Join-Path $outputDirectory ([string]$item.filename)
        $stream = New-Object -ComObject SAPI.SpFileStream
        $stream.Open($target, 3, $false)
        $speaker.AudioOutputStream = $stream
        $watch = [Diagnostics.Stopwatch]::StartNew()
        [void]$speaker.Speak([string]$item.text, 0)
        $watch.Stop()
        $stream.Close()
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject(
            $stream
        )
        $metrics += [pscustomobject]@{
            filename = [string]$item.filename
            synthesis_seconds = $watch.Elapsed.TotalSeconds
            configured_rate = $rate
            applied_speaking_rate = $rate
            volume = $volume
        }
    }
    [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($speaker)
}

Write-JsonFile @{
    voice_id = $VoiceId
    interface = $Interface
    configured_rate = $rate
    volume = $volume
    metrics = $metrics
} $ResultJson
