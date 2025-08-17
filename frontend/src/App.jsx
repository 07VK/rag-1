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

      <footer className="site-footer">
        © {new Date().getFullYear()} ClearChartAI, Inc.
      </footer>
    </>
  );
}
