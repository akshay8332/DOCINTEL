<script setup>
import { onMounted, ref } from "vue";
import { RouterLink } from "vue-router";
import { errorText, http } from "../api";

const invoices = ref([]);
const problem = ref("");

onMounted(async () => {
  try {
    invoices.value = (await http.get("/api/invoices")).data.invoices;
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
    <h2 class="text-2xl font-semibold">Processing history</h2>
    <p class="mt-1 text-sm text-slate-500">Open a row to see that invoice’s runs, extraction, and review.</p>
    <p v-if="problem" class="mt-4 rounded-md bg-rose-50 px-3 py-2 text-sm text-rose-800">{{ problem }}</p>
    <div v-else class="mt-6 overflow-hidden rounded-xl border border-slate-200 bg-white">
      <p v-if="!invoices.length" class="px-4 py-10 text-center text-sm text-slate-500">No processing history yet.</p>
      <table v-else class="w-full text-left text-sm">
        <thead class="bg-slate-50 text-slate-500">
          <tr>
            <th class="px-4 py-3 font-medium">File</th>
            <th class="px-4 py-3 font-medium">Uploaded</th>
            <th class="px-4 py-3 font-medium">Processing</th>
            <th class="px-4 py-3 font-medium">Review</th>
            <th class="px-4 py-3 font-medium">Error</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="invoice in invoices" :key="invoice.id" class="border-t border-slate-100">
            <td class="px-4 py-3">
              <RouterLink :to="`/invoices/${invoice.id}`" class="font-medium text-tide">{{ invoice.filename }}</RouterLink>
            </td>
            <td class="px-4 py-3">{{ when(invoice.uploaded_at) }}</td>
            <td class="px-4 py-3">{{ invoice.processing_status }}</td>
            <td class="px-4 py-3">{{ invoice.review_status }}</td>
            <td class="px-4 py-3 text-rose-700">{{ invoice.error_message || "—" }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
</template>
