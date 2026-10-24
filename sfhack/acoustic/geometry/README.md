# miniDSP UMA-16 v2

16 Knowles SPH1668LM4H MEMS mics, 4x4 uniform rectangular array, 42 mm pitch (aperture 126 mm),
8-48 kHz, 24 bit, USB Audio Class 2 (Windows: miniDSP ASIO driver; macOS driverless).
Product page: https://www.minidsp.com/products/usb-audio-interface/uma-16-microphone-array

Rule of thumb for this geometry: spatial aliasing above c / (2 d) = 343 / 0.084 ~ 4.1 kHz;
good localization roughly 1-4 kHz; below ~500 Hz the beam is very wide.

## geometry/ (copied from https://github.com/JLrobot/Mic-Challenge, hardware/uma16/)
- `minidsp_uma-16.xml`: mic positions per channel (Acoular 26.8, corrected in x after Acoular 24.03, GitHub issue #63).
  Use this one.
- `minidsp_uma-16_mirrored.xml`: the older, mirrored version (kept for reference).
- `uma16_positions.csv`: same positions as CSV, `channel` = 0-based USB channel index.
- Note: channel order is NOT row by row; always map channels through this file.
- Source: Acoular (https://github.com/acoular/acoular), BSD 3-Clause, see `LICENSE-acoular`, `AUTHORS-acoular.rst`.
