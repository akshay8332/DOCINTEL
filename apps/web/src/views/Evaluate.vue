<script setup>
import { onMounted, ref, watch } from "vue";
import { RouterLink } from "vue-router";
import { FIELD_LABELS, errorText, http } from "../api";

const invoices = ref([]);
const selected = ref("");
const history = ref(null);
const problem = ref("");
const comparison = ref(null);
const reference = ref({ vendor_address: "", customer_billing_address: "" });

onMounted(async () => {
  try {
    invoices.value = (await http.get("/api/invoices")).data.invoices;
    if (invoices.value[0]) selected.value = invoices.value[0].id;
  } catch (error) {
    problem.value = errorText(error);
  }
});

async function loadHistory() {
  comparison.value = null;
  if (!selected.value) {
    history.value = null;
    return;
  }
  try {
    history.value = (await http.get(`/api/invoices/${selected.value}/history`)).data;
  } catch (error) {
    problem.value = errorText(error);
  }
}

async function compare() {
  problem.value = "";
  try {
    comparison.value = (await http.post(`/api/invoices/${selected.value}/compare`, reference.value)).data;
  } catch (error) {
    problem.value = errorText(error);
  }
}

watch(selected, loadHistory);
</script>

<template>
  <section>
    <h2 class="text-2xl font-semibold">Evaluation</h2>
    <p class="mt-1 max-w-3xl text-sm text-slate-500">
      Compare original and corrected text, and compare processing runs of the same invoice.
      A match is calculated only when you supply reference text. Predictions are not ground truth.
    </p>
    <p v-if="problem" class="mt-4 rounded-md bg-rose-50 px-3 py-2 text-sm text-rose-800">{{ problem }}</p>
    <p v-if="!invoices.length" class="mt-8 text-sm text-slate-500">No invoices to evaluate.</p>
    <div v-else class="mt-6 space-y-5">
      <label class="block text-sm">
        Invoice
        <select v-model="selected" class="mt-1 block rounded-md border border-slate-200 bg-white px-3 py-2">
          <option v-for="invoice in invoices" :key="invoice.id" :value="invoice.id">
            {{ invoice.filename }} · {{ invoice.processing_status }} · {{ invoice.review_status }}
          </option>
        </select>
      </label>
      <div class="flex flex-wrap gap-3 text-sm">
        <a class="text-tide" :href="`/api/invoices/${selected}/export?format=json`">Export JSON</a>
        <a class="text-tide" :href="`/api/invoices/${selected}/export?format=csv`">Export CSV</a>
        <RouterLink class="text-tide" :to="`/invoices/${selected}`">Open invoice</RouterLink>
      </div>
      <div v-if="history" class="space-y-4">
        <article v-for="run in history.runs" :key="run.id" class="rounded-xl border border-slate-200 bg-white p-4">
          <p class="text-sm text-slate-500">
            {{ new Date(run.created_at).toLocaleString() }} · {{ run.mode }} ·
            {{ run.model_version || "No model" }} · {{ run.correction_version }}
            <template v-if="run.processing_duration != null"> · {{ run.processing_duration }}s</template>
          </p>
          <p v-if="run.error_message" class="mt-2 text-sm text-rose-700">{{ run.error_message }}</p>
          <div v-for="field in run.fields" :key="field.id" class="mt-3 grid gap-2 border-t border-slate-100 pt-3 md:grid-cols-3">
            <div>
              <p class="text-xs font-medium text-slate-500">{{ FIELD_LABELS[field.field_name] }} original</p>
              <pre class="mt-1 whitespace-pre-wrap text-sm">{{ field.original_text || "—" }}</pre>
            </div>
            <div>
              <p class="text-xs font-medium text-slate-500">Automatic correction</p>
              <pre class="mt-1 whitespace-pre-wrap text-sm">{{ field.corrected_text || "—" }}</pre>
            </div>
            <div>
              <p class="text-xs font-medium text-teal-800">Saved value</p>
              <pre class="mt-1 whitespace-pre-wrap text-sm">{{ field.approved_value || "Not saved yet" }}</pre>
            </div>
          </div>
        </article>
      </div>
      <form class="rounded-xl border border-slate-200 bg-white p-4" @submit.prevent="compare">
        <h3 class="font-medium">Optional reference</h3>
        <p class="mt-1 text-sm text-slate-500">Leave a field blank when you do not have reference text for it.</p>
        <label class="mt-3 block text-sm">
          Vendor address reference
          <textarea v-model="reference.vendor_address" rows="3" class="mt-1 w-full rounded-md border border-slate-200 p-2"></textarea>
        </label>
        <label class="mt-3 block text-sm">
          Customer billing address reference
          <textarea v-model="reference.customer_billing_address" rows="3" class="mt-1 w-full rounded-md border border-slate-200 p-2"></textarea>
        </label>
        <button class="mt-3 rounded-md bg-ink px-4 py-2 text-sm text-white" type="submit">Compare to supplied reference</button>
        <div v-if="comparison" class="mt-4 text-sm">
          <p>{{ comparison.note }}</p>
          <ul class="mt-2 space-y-1">
            <li v-for="field in comparison.fields" :key="field.field_name">
              {{ FIELD_LABELS[field.field_name] }}:
              <template v-if="!field.reference_supplied">no reference supplied</template>
              <template v-else>
                original {{ field.original_exact ? "matches" : "differs" }},
                automatic correction {{ field.corrected_exact ? "matches" : "differs" }},
                saved value {{ field.approved_exact === null ? "not saved" : field.approved_exact ? "matches" : "differs" }}
              </template>
            </li>
          </ul>
        </div>
      </form>
    </div>
  </section>
</template>
