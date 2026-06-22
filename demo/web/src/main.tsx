import React from "react";
import { createRoot } from "react-dom/client";
import "./styles/fonts.css";
import "katex/dist/katex.min.css";
import "./styles/global.css";
import "./styles/prism.css";
import { App } from "./App";

const el = document.getElementById("root");
if (!el) throw new Error("missing #root");
createRoot(el).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
