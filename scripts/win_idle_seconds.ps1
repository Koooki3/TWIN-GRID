# Print seconds since the last keyboard/mouse input on the Windows desktop.
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class IdleInfo {
    [StructLayout(LayoutKind.Sequential)]
    public struct LASTINPUTINFO { public uint cbSize; public uint dwTime; }
    [DllImport("user32.dll")]
    public static extern bool GetLastInputInfo(ref LASTINPUTINFO plii);
    public static uint Seconds() {
        LASTINPUTINFO l = new LASTINPUTINFO();
        l.cbSize = 8;
        GetLastInputInfo(ref l);
        return ((uint)Environment.TickCount - l.dwTime) / 1000;
    }
}
"@
[IdleInfo]::Seconds()
