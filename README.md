# Fungoid

A fungal garden, simulated and path-traced on the GPU with WebGPU, on a tiny
planet or in a glass terrarium.

Colonies grow from genomes: hyphae branch, feed, fruit and spread spores, and
crossing two colonies breeds a new one. Both worlds run on one system under a
real sun on a tilted axis. The planet turns under it, so day and night and
the seasons differ from place to place, and storms drift across it. The
terrarium stands at a latitude of its own on such a planet, with that place's
days, seasons and weather. A slime mould roams the ground in wet seasons.

## Running it

Needs [Bun](https://bun.sh) and a browser with WebGPU (a recent Chrome, Edge or
Safari).

```sh
bun install
bun run dev
```

Then open http://127.0.0.1:5244.

## Using it

- Planet / Terrarium picks the world; New planet or New terrarium lays out a
  fresh one.
- Drag to turn the view (right-drag while a ground tool is on), Shift-drag to
  pan, scroll to zoom.
- Click a fungus to edit its growth program. Shift-click a second to cross
  them.
- Water, Dry, Feed and Deplete paint the ground; Slime sets a slime mould down.

In development, `window.fungoid` holds console helpers for inspecting and
recording the garden (`src/dev.ts`).

## License

MIT
