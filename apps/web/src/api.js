import axios from "axios";

export const http = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "",
});

export function errorText(error) {
  const detail = error?.response?.data?.detail;
  if (!detail) return "The request failed.";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail.rejected)) {
    return detail.rejected.map((item) => `${item.filename}: ${item.reason}`).join(" ");
  }
  return "The request failed.";
}

export const FIELD_LABELS = {
  vendor_address: "Vendor address",
  customer_billing_address: "Customer billing address",
};

export const LAYER_LABELS = {
  correct: "Character fix",
  boundary: "Boundary cleanup",
  fallback: "Reader agreement",
};
