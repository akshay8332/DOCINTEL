<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRoute } from "vue-router";
import PdfPane from "../components/PdfPane.vue";
import WordDiff from "../components/WordDiff.vue";
import { FIELD_LABELS, errorText, http } from "../api";

const route = useRoute();
const invoice = ref(null);
const run = ref(null);
const fields = ref([]);
const problem = ref("");
const notice = ref("");
const busy = ref(false);
const pageNumber = ref(1);
const comments = ref({});
const reasons = ref({});
const drafts = ref({});
let timer = null;

const fileUrl = computed(() => `/api/invoices/${route.params.id}/file`);
const warnings = computed(() => run.value?.configuration?.warnings || []);
const stages = [
  ["uploaded", "PDF uploaded"],
  ["rendered", "PDF rendered"],
  ["ocr", "OCR completed"],
  ["inference", "Model inference"],
  ["corrections", "Corrections"],
  ["completed", "Results ready"],
];
const stageIndex = computed(() => {
  const status = invoice.value?.processing_status;
  if (status === "processing") return 1;
  const index = stages.findIndex(([name]) => name === status);
  return index === -1 ? 0 : index;
});

async function load() {
  window.clearTimeout(timer);
  try {
    const payload = (await http.get(`/api/invoices/${route.params.id}/results`)).data;
    invoice.value = payload.invoice;
    run.value = payload.run;
    fields.value = payload.fields;
    for (const field of payload.fields) {
      if (drafts.value[field.field_name] === undefined) {
        drafts.value[field.field_name] = field.approved_value ?? field.corrected_text;
      }
    }
    const active = ["processing", "rendered", "ocr", "inference", "corrections"];
    if (active.includes(payload.invoice.processing_status)) {
      timer = window.setTimeout(load, 1500);
    }
  } catch (error) {
    problem.value = errorText(error);
  }
}

async function processInvoice() {
  busy.value = true;
  problem.value = "";
  notice.value = "";
  try {
    await http.post(`/api/invoices/${route.params.id}/process`);
    notice.value = "Processing started.";
    await load();
  } catch (error) {
    problem.value = errorText(error);
  } finally {
    busy.value = false;
  }
}

async function saveField(field) {
  problem.value = "";
  try {
    const saved = (
      await http.put(`/api/invoices/${route.params.id}/review`, {
        field_name: field.field_name,
        approved_value: drafts.value[field.field_name] || "",
        reviewer_comment: comments.value[field.field_name] || "",
        correction_reason: reasons.value[field.field_name] || "",
      })
    ).data;
    const index = fields.value.findIndex((item) => item.field_name === field.field_name);
    fields.value[index] = saved;
    drafts.value[field.field_name] = saved.approved_value || "";
    notice.value = "Manual correction saved.";
    return true;
  } catch (error) {
    problem.value = errorText(error);
    return false;
  }
}

async function decide(action) {
  problem.value = "";
  try {
    for (const field of fields.value) {
      const typed = drafts.value[field.field_name] ?? "";
      const stored = field.approved_value ?? field.corrected_text ?? "";
      if (typed !== stored && !(await saveField(field))) return;
    }
    const updated = (await http.post(`/api/invoices/${route.params.id}/${action}`, { reviewer_comment: "" })).data;
    invoice.value = { ...invoice.value, ...updated };
    await load();
    notice.value = action === "approve" ? "Invoice approved." : "Invoice rejected.";
  } catch (error) {
    problem.value = errorText(error);
  }
}

function scoreText(field) {
  const signals = field.available_confidence_signals || {};
  if (signals.demo) return "No model score. Demo mode is on.";
  if (signals.model_score === null || signals.model_score === undefined) {
    return signals.note || "No score was produced for this field.";
  }
  return `Mean word probability ${Number(signals.model_score).toFixed(3)}. ${signals.score_meaning || ""}`;
}

onMounted(load);
watch(
  () => route.params.id,
  () => {
    drafts.value = {};
    comments.value = {};
    reasons.value = {};
    load();
  },
);
onBeforeUnmount(() => window.clearTimeout(timer));
</script>

<template>
  <section v-if="invoice">
    <div class="flex flex-wrap items-start justify-between gap-3">
      <div>
        <h2 class="text-2xl font-semibold">{{ invoice.filename }}</h2>
        <p class="mt-1 text-sm text-slate-500">
          Processing {{ invoice.processing_status }} · Review {{ invoice.review_status }}
          <template v-if="run">
            · {{ run.model_version || "No model" }} · {{ run.correction_version }}
          </template>
        </p>
      </div>
      <button class="rounded-md bg-ink px-4 py-2 text-sm text-white disabled:opacity-50" :disabled="busy" @click="processInvoice">
        Run extraction
      </button>
    </div>
    <ol class="mt-4 flex flex-wrap gap-2 text-xs">
      <li
        v-for="([name, label], index) in stages"
        :key="name"
        class="rounded-full px-3 py-1"
        :class="
          invoice.error_message && name === invoice.processing_status
            ? 'bg-rose-100 text-rose-900'
            : invoice.processing_status === 'completed' || index < stageIndex
              ? 'bg-teal-100 text-teal-900'
              : index === stageIndex
                ? 'bg-amber-100 text-amber-950'
                : 'bg-slate-100 text-slate-500'
        "
      >
        {{ label }}
      </li>
    </ol>
    <p v-if="run?.configuration?.ocr_reader" class="mt-3 text-sm text-slate-600">
      OCR {{ run.configuration.ocr_reader }}
      <template v-if="run.configuration.ocr_version"> {{ run.configuration.ocr_version }}</template>
      · {{ run.configuration.ocr_source }}
    </p>
    <p v-if="problem" class="mt-4 rounded-md bg-rose-50 px-3 py-2 text-sm text-rose-800">{{ problem }}</p>
    <p v-if="invoice.error_message" class="mt-4 rounded-md bg-rose-50 px-3 py-2 text-sm text-rose-800">
      {{ invoice.error_message }}
    </p>
    <p v-if="notice" class="mt-4 rounded-md bg-teal-50 px-3 py-2 text-sm text-teal-900">{{ notice }}</p>
    <ul v-if="warnings.length" class="mt-4 space-y-1 text-sm text-amber-900">
      <li v-for="warning in warnings" :key="warning">{{ warning }}</li>
    </ul>

    <div v-if="!run" class="mt-8 rounded-xl border border-dashed border-slate-300 bg-white px-4 py-10 text-center text-sm text-slate-500">
      This invoice has not been processed.
    </div>

    <div v-else class="mt-6 grid items-start gap-5 xl:grid-cols-[minmax(0,1.2fr)_minmax(320px,0.8fr)]">
      <PdfPane v-model:page-number="pageNumber" :file-url="fileUrl" :boxes="fields" />
      <div class="space-y-4">
        <article v-for="field in fields" :key="field.field_name" class="rounded-xl border border-slate-200 bg-white p-4">
          <div class="flex items-center justify-between gap-2">
            <h3 class="font-medium">{{ FIELD_LABELS[field.field_name] }}</h3>
            <button class="text-sm text-tide" type="button" @click="pageNumber = field.page_number || 1">
              Page {{ field.page_number || "—" }}
            </button>
          </div>
          <p class="mt-1 text-xs uppercase tracking-wide text-slate-500">Review {{ field.review_status }}</p>
          <div class="mt-3 grid gap-3">
            <div>
              <p class="text-xs font-medium text-slate-500">Original prediction</p>
              <pre class="mt-1 whitespace-pre-wrap rounded-md bg-slate-50 p-2 text-sm">{{ field.original_text || "—" }}</pre>
            </div>
            <div>
              <p class="text-xs font-medium text-teal-800">Saved value</p>
              <pre class="mt-1 whitespace-pre-wrap rounded-md bg-teal-50 p-2 text-sm">{{ field.approved_value || "Not saved yet" }}</pre>
            </div>
            <div>
              <p class="text-xs font-medium text-slate-500">Automatic correction</p>
              <pre class="mt-1 whitespace-pre-wrap rounded-md bg-slate-50 p-2 text-sm">{{ field.corrected_text || "—" }}</pre>
            </div>
          </div>
          <p class="mt-3 text-xs text-slate-500">{{ scoreText(field) }}</p>
          <p v-if="field.bounding_box" class="mt-1 text-xs text-slate-500">
            Box {{ field.bounding_box.map((value) => Number(value).toFixed(3)).join(", ") }}
            <template v-if="field.demo"> · demo highlight</template>
          </p>
          <p v-if="field.layers_applied?.length" class="mt-1 text-xs text-slate-500">
            Layers that changed the text: {{ field.layers_applied.join(", ") }}
          </p>
          <div class="mt-4">
            <p class="text-xs font-medium text-slate-500">Word comparison</p>
            <div class="mt-2">
              <WordDiff :tokens="field.token_changes || []" />
            </div>
          </div>
          <label class="mt-4 block text-xs font-medium text-slate-500">
            Human value
            <textarea v-model="drafts[field.field_name]" rows="3" class="mt-1 w-full rounded-md border border-slate-200 p-2 text-sm"></textarea>
          </label>
          <label class="mt-2 block text-xs font-medium text-slate-500">
            Comment
            <input v-model="comments[field.field_name]" class="mt-1 w-full rounded-md border border-slate-200 p-2 text-sm" />
          </label>
          <label class="mt-2 block text-xs font-medium text-slate-500">
            Correction reason
            <input v-model="reasons[field.field_name]" class="mt-1 w-full rounded-md border border-slate-200 p-2 text-sm" />
          </label>
          <button class="mt-3 rounded-md border border-slate-300 px-3 py-1.5 text-sm" type="button" @click="saveField(field)">
            Save manual correction
          </button>
        </article>
        <div class="flex gap-2">
          <button class="rounded-md bg-tide px-4 py-2 text-sm text-white" type="button" @click="decide('approve')">Approve</button>
          <button class="rounded-md border border-rose-300 px-4 py-2 text-sm text-rose-800" type="button" @click="decide('reject')">Reject</button>
        </div>
      </div>
    </div>
  </section>
</template>
