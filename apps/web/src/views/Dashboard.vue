<script setup>
import { onMounted, ref } from "vue";
import { RouterLink } from "vue-router";
import { errorText, http } from "../api";

const summary = ref(null);
const invoices = ref([]);
const problem = ref("");

const cards = [
  ["total", "Uploaded invoices"],
  ["processed", "Successfully processed"],
  ["pending_review", "Pending human review"],
  ["approved", "Approved"],
  ["failed", "Failed jobs"],
];

onMounted(async () => {
  try {
    const payload = (await http.get("/api/invoices")).data;
    summary.value = payload.summary;
    invoices.value = payload.invoices.slice(0, 8);
  } catch (error) {
    problem.value = errorText(error);
  }
});

function when(value) {
  return value ? new Date(value).toLocaleString() : "—";
}
</script>

<template>
  <section>
    <h2 class="text-2xl font-semibold">Dashboard</h2>
    <p class="mt-1 text-sm text-slate-500">Counts come from invoices stored on this computer.</p>
    <p v-if="problem" class="mt-4 rounded-md bg-rose-50 px-3 py-2 text-sm text-rose-800">{{ problem }}</p>
    <div v-else-if="summary" class="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
      <article v-for="[key, label] in cards" :key="key" class="rounded-xl border border-slate-200 bg-white p-4">
        <p class="text-sm text-slate-500">{{ label }}</p>
        <p class="mt-2 text-3xl font-semibold">{{ summary[key] }}</p>
      </article>
    </div>
    <div class="mt-8 rounded-xl border border-slate-200 bg-white">
      <div class="flex items-center justify-between border-b border-slate-100 px-4 py-3">
        <h3 class="font-medium">Recent activity</h3>
        <RouterLink to="/upload" class="text-sm text-tide">Upload an invoice</RouterLink>
      </div>
      <p v-if="!invoices.length" class="px-4 py-10 text-center text-sm text-slate-500">
        No invoices yet. Upload a PDF to start a processing run.
      </p>
      <table v-else class="w-full text-left text-sm">
        <thead class="text-slate-500">
          <tr>
            <th class="px-4 py-2 font-medium">File</th>
            <th class="px-4 py-2 font-medium">Uploaded</th>
            <th class="px-4 py-2 font-medium">Processing</th>
            <th class="px-4 py-2 font-medium">Review</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="invoice in invoices" :key="invoice.id" class="border-t border-slate-100">
            <td class="px-4 py-3">
              <RouterLink :to="`/invoices/${invoice.id}`" class="font-medium text-tide">
                {{ invoice.filename }}
              </RouterLink>
            </td>
            <td class="px-4 py-3">{{ when(invoice.uploaded_at) }}</td>
            <td class="px-4 py-3">{{ invoice.processing_status }}</td>
            <td class="px-4 py-3">{{ invoice.review_status }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
</template>
