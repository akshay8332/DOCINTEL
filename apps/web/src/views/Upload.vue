<script setup>
import { ref } from "vue";
import { RouterLink } from "vue-router";
import { errorText, http } from "../api";

const dragging = ref(false);
const items = ref([]);
const problem = ref("");
const limit = 20 * 1024 * 1024;

function addFiles(list) {
  problem.value = "";
  for (const file of list) {
    const row = {
      id: `${file.name}-${file.size}-${Math.random()}`,
      file,
      progress: 0,
      status: "queued",
      message: "",
      invoiceId: "",
    };
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      row.status = "error";
      row.message = "Only PDF files are accepted.";
    } else if (file.size > limit) {
      row.status = "error";
      row.message = "This file is larger than the 20 MB limit.";
    }
    items.value.push(row);
  }
  uploadQueued();
}

async function uploadQueued() {
  for (const row of items.value) {
    if (row.status !== "queued") continue;
    row.status = "uploading";
    const body = new FormData();
    body.append("files", row.file);
    try {
      const response = await http.post("/api/invoices/upload", body, {
        onUploadProgress(event) {
          if (event.total) row.progress = Math.round((event.loaded / event.total) * 100);
        },
      });
      const accepted = response.data.accepted?.[0];
      row.progress = 100;
      row.status = "uploaded";
      row.invoiceId = accepted?.id || "";
      row.message = "Uploaded. Open it to run extraction.";
    } catch (error) {
      row.status = "error";
      row.message = errorText(error);
    }
  }
}

function onDrop(event) {
  dragging.value = false;
  addFiles(event.dataTransfer.files);
}
</script>

<template>
  <section>
    <h2 class="text-2xl font-semibold">Upload invoices</h2>
    <p class="mt-1 text-sm text-slate-500">PDF only, up to 20 MB each. Several files can be added together.</p>
    <label
      class="mt-6 flex cursor-pointer flex-col items-center rounded-xl border-2 border-dashed bg-white px-6 py-14 text-center"
      :class="dragging ? 'border-tide bg-teal-50' : 'border-slate-300'"
      @dragover.prevent="dragging = true"
      @dragleave.prevent="dragging = false"
      @drop.prevent="onDrop"
    >
      <span class="font-medium">Drop PDF invoices here</span>
      <span class="mt-1 text-sm text-slate-500">or choose files</span>
      <input class="sr-only" type="file" accept="application/pdf,.pdf" multiple @change="addFiles($event.target.files)" />
    </label>
    <p v-if="problem" class="mt-4 text-sm text-rose-700">{{ problem }}</p>
    <ul class="mt-6 space-y-3">
      <li v-for="row in items" :key="row.id" class="rounded-xl border border-slate-200 bg-white p-4">
        <div class="flex items-center justify-between gap-3">
          <p class="font-medium">{{ row.file.name }}</p>
          <span class="text-xs uppercase tracking-wide text-slate-500">{{ row.status }}</span>
        </div>
        <div class="mt-3 h-2 overflow-hidden rounded-full bg-slate-100">
          <div class="h-full bg-tide" :style="{ width: `${row.progress}%` }"></div>
        </div>
        <p class="mt-2 text-sm text-slate-600">{{ row.message }}</p>
        <RouterLink v-if="row.invoiceId" :to="`/invoices/${row.invoiceId}`" class="mt-2 inline-block text-sm text-tide">
          Open invoice
        </RouterLink>
      </li>
    </ul>
  </section>
</template>
