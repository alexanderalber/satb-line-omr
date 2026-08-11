<?xml version="1.0" encoding="UTF-8"?>
<!-- Corpus style for the MuseScore second renderer.

Measured, not guessed (scripts/32_lyric_crop_stats.py, work/32_lyric_crop_stats.json):
with MuseScore defaults only 80 % of the detector crops carry lyric ink against
95 % in the verovio corpus - vertical page justification spreads sparse pages
and autoplace pushes whole syllable rows below the 3-spacing crop padding.
With these three values the share is 92 % and the syllable row sits 1.1-3.4
spacings under the staff (verovio: 2.2-2.4). Everything else stays at the
MuseScore default on purpose: the point of the second renderer is foreign
engraving as it really looks, so only the crop-geometry knobs are touched. -->
<museScore version="4.70">
  <Style>
    <enableVerticalSpread>0</enableVerticalSpread>
    <lyricsPosBelow>1.5</lyricsPosBelow>
    <lyricsMinTopDistance>0.5</lyricsMinTopDistance>
  </Style>
</museScore>
