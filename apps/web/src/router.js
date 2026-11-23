import { createRouter, createWebHistory } from "vue-router";
import Dashboard from "./views/Dashboard.vue";
import Upload from "./views/Upload.vue";
import Invoices from "./views/Invoices.vue";
import InvoiceDetail from "./views/InvoiceDetail.vue";
import Evaluate from "./views/Evaluate.vue";

export default createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", name: "dashboard", component: Dashboard },
    { path: "/upload", name: "upload", component: Upload },
    { path: "/invoices", name: "invoices", component: Invoices },
    { path: "/invoices/:id", name: "invoice", component: InvoiceDetail },
    { path: "/evaluate", name: "evaluate", component: Evaluate },
  ],
});
