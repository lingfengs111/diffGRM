# Standalone AR beam-width control

The canonical Video23 OPQ4 standalone AR checkpoint used search width 128 and
returned top-10 items. This evaluation-only sweep raises the actual constrained
search width to 256 and 500 while keeping the checkpoint and returned top-10
fixed. Since one coordinate has only 256 values, the width-500 configuration
uses width 256 at the first step and width 500 thereafter.
