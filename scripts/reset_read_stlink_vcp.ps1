param([string]$Port = "COM3", [int]$TimeoutSeconds = 8)
$serial = New-Object System.IO.Ports.SerialPort $Port,115200,'None',8,'One'
$serial.DtrEnable = $false
$serial.RtsEnable = $false
$serial.Open()
Start-Sleep -Milliseconds 100
$serial.DtrEnable = $true
$serial.RtsEnable = $true
$serial.BreakState = $true
Start-Sleep -Milliseconds 100
$serial.BreakState = $false
$serial.DtrEnable = $false
$serial.RtsEnable = $false
Start-Sleep -Milliseconds 800
$serial.DiscardInBuffer()
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$text = ''
while ((Get-Date) -lt $deadline) {
    $text += $serial.ReadExisting()
    if ($text.Contains("END`r`n")) { break }
    Start-Sleep -Milliseconds 50
}
$serial.Close()
$text
