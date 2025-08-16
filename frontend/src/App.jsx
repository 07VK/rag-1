import React from "react";
import { Routes, Route } from "react-router-dom";
import About from "./pages/About";
import Demo from "./pages/Demo";

export default function App() {
  return (
    <>
      <Routes>
        <Route path="/" element={<About />} />
        <Route path="/demo" element={<Demo />} />
      </Routes>

      <footer className="mx-auto max-w-6xl px-6 py-10 text-xs text-slate-400 text-center">
        © 2025 ClearChartAI. Inc.,
      </footer>
    </>
  );
}
