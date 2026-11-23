<script setup>
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import * as pdfjs from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";

pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;

const props = defineProps({
  fileUrl: { type: String, required: true },
  boxes: { type: Array, default: () => [] },
  pageNumber: { type: Number, default: 1 },
});
const emit = defineEmits(["update:pageNumber"]);

const canvas = ref(null);
const stage = ref(null);
const scale = ref(1.15);
const pageCount = ref(0);
const problem = ref("");
let documentTask = null;
let renderToken = 0;

async function draw() {
  await nextTick();
  if (!canvas.value) return;
  const token = ++renderToken;
  problem.value = "";
  try {
    if (!documentTask) documentTask = pdfjs.getDocument(props.fileUrl).promise;
    const pdf = await documentTask;
    if (token !== renderToken) return;
    pageCount.value = pdf.numPages;
    const page = await pdf.getPage(Math.min(props.pageNumber, pdf.numPages));
    const viewport = page.getViewport({ scale: scale.value });
    const target = canvas.value;
    target.width = viewport.width;
    target.height = viewport.height;
    await page.render({ canvasContext: target.getContext("2d"), viewport }).promise;
  } catch {
    if (token === renderToken) problem.value = "This PDF could not be displayed.";
  }
}

function move(next) {
  const page = Math.min(Math.max(1, next), pageCount.value || 1);
  emit("update:pageNumber", page);
}

let loadedUrl = "";
watch(
  () => props.fileUrl,
  (url) => {
    if (url !== loadedUrl) {
      documentTask = null;
      loadedUrl = url;
    }
  },
);
watch(() => [props.fileUrl, props.pageNumber, scale.value], draw);
onMounted(draw);
onBeforeUnmount(() => {
  renderToken += 1;
});
</script>

<template>
  <div class="rounded-xl border border-slate-200 bg-white">
    <div class="flex flex-wrap items-center gap-2 border-b border-slate-100 px-3 py-2 text-sm">
      <button class="rounded border border-slate-200 px-2 py-1" type="button" @click="move(pageNumber - 1)">Previous</button>
      <span>Page {{ pageNumber }}<span v-if="pageCount"> / {{ pageCount }}</span></span>
      <button class="rounded border border-slate-200 px-2 py-1" type="button" @click="move(pageNumber + 1)">Next</button>
      <button class="rounded border border-slate-200 px-2 py-1" type="button" @click="scale = Math.max(0.6, scale - 0.15)">−</button>
      <button class="rounded border border-slate-200 px-2 py-1" type="button" @click="scale = Math.min(2.4, scale + 0.15)">+</button>
    </div>
    <p v-if="problem" class="px-4 py-8 text-sm text-rose-700">{{ problem }}</p>
    <div v-show="!problem" ref="stage" class="relative max-h-[760px] overflow-auto bg-slate-100 p-4">
      <div class="relative inline-block">
        <canvas ref="canvas" class="bg-white shadow"></canvas>
        <div
          v-for="box in boxes.filter((item) => item.page_number === pageNumber && item.bounding_box)"
          :key="box.field_name"
          class="absolute border-2"
          :class="box.field_name === 'vendor_address' ? 'border-amber-600 bg-amber-400/20' : 'border-blue-700 bg-blue-500/20'"
          :style="{
            left: `${box.bounding_box[0] * 100}%`,
            top: `${box.bounding_box[1] * 100}%`,
            width: `${(box.bounding_box[2] - box.bounding_box[0]) * 100}%`,
            height: `${(box.bounding_box[3] - box.bounding_box[1]) * 100}%`,
          }"
        >
          <span class="absolute -top-5 left-0 whitespace-nowrap bg-white px-1 text-[10px] font-medium">
            {{ box.field_name === "vendor_address" ? "Vendor" : "Customer" }}
            <template v-if="box.demo"> · demo</template>
          </span>
        </div>
      </div>
    </div>
  </div>
</template>
