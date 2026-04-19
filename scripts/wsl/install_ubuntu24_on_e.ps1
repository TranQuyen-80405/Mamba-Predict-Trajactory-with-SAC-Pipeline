#Requires -Version 5.1
<#
.SYNOPSIS
  Dat Ubuntu 24.04 LTS (WSL2) len o E: bang wsl --export / --import.

.DESCRIPTION
  Luu y quan trong:
  - Kernel WSL2 luon la kernel Microsoft (vd: *-microsoft-standard-WSL2), KHONG phai
    5.15.0-119-generic nhu may Ubuntu native hay container tren Linux.
  - Phien ban OS van la Ubuntu 24.04.x LTS (Noble); kiem tra: cat /etc/os-release

  Quy trinh (chay PowerShell **voi quyen Administrator** neu can):
  1) Cai Ubuntu-24.04 lan dau (neu chua co):  wsl --install -d Ubuntu-24.04
     (hoac Microsoft Store -> Ubuntu 24.04)
  2) Mo Ubuntu 24.04 mot lan, tao user, roi dong.
  3) Chay script nay de CHUYEN distro xuong E:\

.PARAMETER DistroName
  Ten distro trong 'wsl -l' (mac dinh: Ubuntu-24.04).

.PARAMETER ERoot
  Thu muc tren o E: chua file .vhdx (mac dinh: E:\WSL\Ubuntu-24.04).

.PARAMETER NewName
  Ten distro sau khi import (mac dinh: Ubuntu-24.04-E).
#>
param(
    [string]$DistroName = "Ubuntu-24.04",
    [string]$ERoot = "E:\WSL\Ubuntu-24.04",
    [string]$NewName = "Ubuntu-24.04-E",
    [string]$TarPath = "E:\WSL\ubuntu-24.04-export.tar"
)

$ErrorActionPreference = "Stop"

Write-Host "==> Shutdown WSL..."
wsl --shutdown
Start-Sleep -Seconds 2

if (-not (Test-Path (Split-Path -Parent $ERoot))) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $ERoot) -Force | Out-Null
}
New-Item -ItemType Directory -Path $ERoot -Force | Out-Null

Write-Host "==> Export $DistroName -> $TarPath"
wsl --export $DistroName $TarPath

Write-Host "==> Unregister $DistroName (ban sao da luu trong tar)"
wsl --unregister $DistroName

Write-Host "==> Import -> $ERoot as $NewName"
wsl --import $NewName $ERoot $TarPath --version 2

Write-Host "==> Xoa file tar (tiet kiem dung luong)? Goi: Remove-Item '$TarPath'"
Write-Host "==> Mac dinh WSL: wsl --set-default $NewName"
wsl --set-default $NewName

Write-Host ""
Write-Host "Xong. Khoi dong: wsl -d $NewName"
Write-Host "Trong WSL, dat hostname giong container: sudo hostnamectl set-hostname c76e567292c9"
