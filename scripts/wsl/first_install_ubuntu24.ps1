#Requires -Version 5.1
<#
  Lan dau: cai Ubuntu 24.04 tu Microsoft (chua co distro nao ten Ubuntu-24.04).
  Chay PowerShell **Administrator**:

    wsl --install -d Ubuntu-24.04

  Sau khi cai xong, mo app Ubuntu 24.04, tao user UNIX, roi chay
  scripts\wsl\install_ubuntu24_on_e.ps1 de chuyen o dia ao xuong E:\

  Neu da co Ubuntu-22.04 va muon **them** 24.04 song song:
    wsl --install -d Ubuntu-24.04
#>
Write-Host "Chay lenh sau trong PowerShell (Administrator):"
Write-Host "  wsl --install -d Ubuntu-24.04"
Write-Host ""
Write-Host "Sau do mo Ubuntu 24.04, dat user/password, roi chay install_ubuntu24_on_e.ps1 de dat tren o E:."
