<script setup>
import { onMounted, ref } from "vue";
import { RouterLink, RouterView } from "vue-router";
import { http } from "./api";

const health = ref(null);
const links = [
  ["/", "Dashboard"],
  ["/upload", "Upload"],
  ["/invoices", "Invoices"],
  ["/evaluate", "Evaluation"],
];

onMounted(async () => {
  try {
    health.value = (await http.get("/api/health")).data;
  } catch {
    health.value = { status: "offline" };
  }
});
</script>

<template>
  <div class="min-h-screen md:grid md:grid-cols-[240px_1fr]">
    <aside class="bg-ink text-white md:min-h-screen">
      <div class="px-5 py-6">
        <p class="text-xs tracking-[0.18em] text-teal-200">INVOICE OPERATIONS</p>
        <h1 class="mt-1 text-2xl font-semibold">DOCILE</h1>
        <p class="mt-1 text-sm text-slate-300">Intelligent Invoice Processing</p>
      </div>
      <nav class="flex gap-2 overflow-x-auto px-3 pb-4 md:block md:px-3">
        <RouterLink
          v-for="[to, label] in links"
          :key="to"
          :to="to"
          class="block whitespace-nowrap rounded-md px-3 py-2 text-sm text-slate-200 hover:bg-white/10"
          exact-active-class="bg-white/15 text-white"
        >
          {{ label }}
        </RouterLink>
      </nav>
    </aside>
    <div>
      <header class="flex items-center justify-between border-b border-slate-200 bg-white px-6 py-3">
        <p class="text-sm text-slate-500">Local workspace</p>
        <p
          v-if="health?.demo"
          class="rounded-full bg-amber-100 px-3 py-1 text-xs font-medium text-amber-900"
        >
          Demo mode — these are not model predictions
        </p>
        <p
          v-else-if="health?.inference_mode === 'unavailable'"
          class="rounded-full bg-rose-100 px-3 py-1 text-xs font-medium text-rose-900"
        >
          Frozen checkpoint is not installed
        </p>
        <p
          v-else-if="health?.inference_mode === 'real'"
          class="rounded-full bg-teal-100 px-3 py-1 text-xs font-medium text-teal-900"
        >
          Frozen checkpoint ready
        </p>
        <p v-else class="text-xs text-slate-400">API unavailable</p>
      </header>
      <main class="px-6 py-6">
        <RouterView />
      </main>
    </div>
  </div>
</template>
