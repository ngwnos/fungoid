// ---- cellprog.wgsl: organism program lookups (requires `pheno`) ----

// Organism programs (expressed genomes).
fn PG(org: u32, i: u32) -> f32 { return pheno[org * PH_STRIDE + i]; }
fn PQ(org: u32, phase: u32, i: u32) -> f32 { return pheno[org * PH_STRIDE + PHASE_BASE + phase * PHASE_STRIDE + i]; }

// Tissue colour of a cell: flesh -> pigment -> spore colour.
fn cellColor(org: u32, phase: u32) -> vec3f {
  let flesh = vec3f(PG(org, G_FLESH_R), PG(org, G_FLESH_G), PG(org, G_FLESH_B));
  let pig = vec3f(PG(org, G_PIG_R), PG(org, G_PIG_G), PG(org, G_PIG_B));
  let spore = vec3f(PG(org, G_SPORE_R), PG(org, G_SPORE_G), PG(org, G_SPORE_B));
  let k = PQ(org, phase, Q_PIGMENT);
  if (phase == 0u) { return mix(vec3f(0.9, 0.89, 0.85), pig, k * 0.5); }
  if (k < 0.66) { return mix(flesh, pig, k / 0.66); }
  return mix(pig, spore, (k - 0.66) / 0.34);
}

// The colour of its light; tissue's paler, nearer white.
fn glowColour(org: u32) -> vec3f { return vec3f(PG(org, G_GLOW_R), PG(org, G_GLOW_G), PG(org, G_GLOW_B)); }
fn cellGlowColour(org: u32) -> vec3f { return mix(glowColour(org), vec3f(1.0), 0.45); }

// The light a cell gives: its phase's own glow, pale, withering away with
// it; and its flash while it fires, in the colour of its light.
fn cellLight(org: u32, ph: u32, flags: u32, age: f32) -> vec3f {
  return cellGlowColour(org) * PQ(org, ph, Q_GLOW) * (1.0 - wither(org, ph, age))
       + glowColour(org) * PG(org, G_PULSE_GLOW) * 2.0 * f32(exciteOf(flags)) / 255.0;
}

// How far a fruiting cell has withered (0..1), over the last quarter of its
// life (mycelium, and tissue that lives on, never do).
fn wither(org: u32, phase: u32, age: f32) -> f32 {
  let life = PQ(org, phase, Q_LIFESPAN);
  if (phase == 0u || life > 1e8) { return 0.0; }
  return smoothstep(0.75 * life, life, age);
}
