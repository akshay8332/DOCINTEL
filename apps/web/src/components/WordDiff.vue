<script setup>
import { LAYER_LABELS } from "../api";

defineProps({
  tokens: { type: Array, default: () => [] },
});

const tone = {
  unchanged: "bg-slate-100 text-slate-700",
  changed: "bg-amber-100 text-amber-950",
  added: "bg-emerald-100 text-emerald-950",
  removed: "bg-rose-100 text-rose-900 line-through",
};
</script>

<template>
  <div>
    <div class="flex flex-wrap gap-1">
      <span
        v-for="(token, index) in tokens"
        :key="`${token.text}-${index}`"
        class="rounded px-1.5 py-0.5 text-sm"
        :class="tone[token.change] || tone.unchanged"
      >
        {{ token.text }}
        <span v-if="LAYER_LABELS[token.layer]" class="ml-1 text-[10px] uppercase tracking-wide">
          {{ LAYER_LABELS[token.layer] }}
        </span>
      </span>
    </div>
    <p v-if="!tokens.length" class="text-sm text-slate-500">No tokens to compare.</p>
    <div class="mt-3 flex flex-wrap gap-3 text-xs text-slate-500">
      <span>Unchanged</span>
      <span class="text-amber-800">Changed</span>
      <span class="text-emerald-800">Added</span>
      <span class="text-rose-800 line-through">Removed</span>
    </div>
  </div>
</template>
