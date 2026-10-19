param(
    [string]$InputDocx = (Join-Path $PSScriptRoot 'Bao_cao_Cuoi_ky_Nhom7_CRISP_DM.docx'),
    [string]$OutputPdf = (Join-Path $PSScriptRoot 'Bao_cao_Cuoi_ky_Nhom7_CRISP_DM.pdf')
)
$ErrorActionPreference = 'Stop'
$inputFile = (Resolve-Path -LiteralPath $InputDocx).Path
$outputFile = [System.IO.Path]::GetFullPath($OutputPdf)
$word = $null
$document = $null
try {
    # Tạo phiên Word ẩn riêng, không dùng tài liệu Word người dùng đang mở.
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($inputFile, $false, $true)
    $document.Repaginate()
    $document.ExportAsFixedFormat($outputFile, 17)
    Write-Output "PDF exported: $outputFile"
} finally {
    if ($null -ne $document) { try { $document.Close(0) } catch {} }
    if ($null -ne $word) { try { $word.Quit() } catch {} }
}
