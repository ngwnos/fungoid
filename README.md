# Fungoid

A fungal garden on a tiny planet, simulated and path-traced on the GPU with
WebGPU.

Colonies grow from genomes: hyphae branch, feed, fruit and spread spores, and
crossing two colonies breeds a new one. The planet turns under a real sun on a
tilted axis, so day and night and the seasons differ from place to place, and
storms drift across it. A slime mould roams the ground in wet seasons.

## Running it

Needs [Bun](https://bun.sh) and a browser with WebGPU (a recent Chrome, Edge or
Safari).

```sh
bun install
bun run dev
```

Then open http://127.0.0.1:5244.

## Using it

- Drag to turn the planet (right-drag while a ground tool is on), Shift-drag
  to pan, scroll to zoom.
- Click a fungus to edit its growth program. Shift-click a second to cross
  them.
- Water, Dry, Feed and Deplete paint the ground; Slime sets a slime mould down.

In development, `window.fungoid` holds console helpers for inspecting and
recording the garden (`src/dev.ts`).

## License

MIT
