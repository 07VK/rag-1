import React, { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

export default function Demo() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [status, setStatus] = useState("");
  const [showChat, setShowChat] = useState(false);
  const [sources, setSources] = useState([]);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const [showPreview, setShowPreview] = useState(false);

   // drag & drop
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const chatBodyRef = useRef(null);

  // Auto-scroll chat to bottom on new messages
  useEffect(() => {
    if (chatBodyRef.current) {
      chatBodyRef.current.scrollTop = chatBodyRef.current.scrollHeight;
    }
  }, [messages]);

  // Prevent the browser from opening dropped files outside our dropzone
  useEffect(() => {
    const stop = (e) => {
      e.preventDefault();
      e.stopPropagation();
    };
    window.addEventListener("dragover", stop);
    window.addEventListener("drop", stop);
    return () => {
      window.removeEventListener("dragover", stop);
      window.removeEventListener("drop", stop);
    };
  }, []);

  // Build & clean up a blob URL for the selected file
  useEffect(() => {
    if (!selectedFile) {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedFile]);

  // Close PDF preview on Escape key
  useEffect(() => {
    function onKey(e) {
      if (e.key === "Escape") setShowPreview(false);
    }
    if (showPreview) window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [showPreview]);

  function handleFileChange(files) {
    const f = files?.[0];
    if (!f) return;
    if (f.type !== "application/pdf") {
      setSelectedFile(null);
      setStatus("Please select a valid PDF file.");
      return;
    }
    setSelectedFile(f);
    setStatus("");
  }

   // Drag & drop handlers
  function onDragEnter(e) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  }
  function onDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    if (!dragActive) setDragActive(true);
  }
  function onDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    if (e.currentTarget === e.target) setDragActive(false);
  }
  function onDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const files = e.dataTransfer?.files;
    if (files?.length) handleFileChange(files);
  }
  function onClickPick() {
    fileInputRef.current?.click();
  }

  async function processAndSwitch() {
    if (!selectedFile) {
      setStatus("Please select a PDF file first.");
      return;
    }
    setStatus("Processing document...");
    const form = new FormData();
    form.append("file", selectedFile);

    try {
      const res = await fetch("/upload-pdf", { method: "POST", body: form });
      const json = await res.json();
      if (res.ok && json.status === "success") {
        setSessionId(json.session_id); 
        setStatus("Success! Starting chat...");
        setShowChat(true);
        setMessages([
          { sender: "Bot", text: "The document has been processed. You can now ask questions." },
        ]);
        setSources([selectedFile.name]);
        // if (json.fileUrl) setPreviewUrl(json.fileUrl);
      } else {
        throw new Error(json.message || "Failed to process PDF.");
      }
    } catch (e) {
      setStatus(`Error: ${e.message}`);
    }
  }

  function openPreview() {
    if (previewUrl) setShowPreview(true);
  }

  function closePreview() {
    setShowPreview(false);
  }

  // --- UPDATED: sendQuestion function to include history ---
  async function sendQuestion() {
    const q = question.trim();
    if (!q) return;

    // Prepare history by filtering out any "typing" indicators
    const historyToSend = messages
      .filter(m => !m.typing)
      .map(m => ({
          // Ensure the sender key matches the backend's Pydantic model
          sender: m.sender === "You" ? "User" : "Bot", 
          text: m.text
      }));

    setMessages((m) => [...m, { sender: "You", text: q }, { sender: "Bot", text: "", typing: true }]);
    setQuestion("");

    try {
      const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          question: q, 
          session_id: sessionId,
          history: historyToSend // <-- Send the history
        }),
      });
      
      if (!res.ok) {
        // Handle non-200 responses more gracefully
        const errorJson = await res.json();
        throw new Error(errorJson.detail || "An API error occurred.");
      }

      setMessages((m) => {
        const updated = [...m];
        updated[updated.length - 1] = {
          sender: "Bot",
          text: res.ok ? json.answer || "" : json.answer || "An error occurred.",
        };
        return updated;
      });
    } catch (e) {
      setMessages((m) => {
        const updated = [...m];
        updated[updated.length - 1] = { sender: "Bot", text: `Error: ${e.message}` };
        return updated;
      });
    }
  }

  return (
    <div className="app-container">
      {!showChat ? (
        <section id="upload-section" aria-label="Upload PDF to start demo">
          <div className="upload-portal">
            <Link to="/" className="logo-container" style={{ flexDirection: "row", gap: "1rem" }}>
              <img
                src="/static/logo.png"
                alt="ClearChartAI"
                className="logo-image"
                style={{ width: "4rem", height: "auto", marginLeft: 0 }}
              />
              <div>
                <h1 className="logo-text" style={{ margin: 0 }}>ClearChartAI</h1>
                <p className="logo-subtitle" style={{ marginTop: 0 }}>
                  Understand Your Health. Own Your Future
                </p>
              </div>
            </Link>

            <p>Begin by uploading a document to activate the workspace.</p>

            {/* Drag & Drop / Click zone */}
            <div
              className={`upload-area ${dragActive ? "drag-active" : ""}`}
              onClick={onClickPick}
              onDragEnter={onDragEnter}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onClickPick()}
              aria-label="Upload PDF by clicking or dragging a file here"
            >
              <input
                ref={fileInputRef}
                id="pdf-file"
                type="file"
                accept="application/pdf,.pdf"
                style={{ display: "none" }}
                onChange={(e) => handleFileChange(e.target.files)}
              />
              <div style={{ pointerEvents: "none", textAlign: "center" }}>
                <div style={{ fontSize: "1.1rem", fontWeight: 600 }}>
                  {selectedFile ? selectedFile.name : "Click or Drag & Drop a PDF"}
                </div>
                <div style={{ fontSize: ".9rem", color: "var(--text-secondary)" }}>
                  Only .pdf files are accepted
                </div>
              </div>
            </div>

            {/* Process */}
            <button
              className="button"
              style={{
                background: "linear-gradient(to right, var(--medical-teal), var(--medical-teal-dark))",
                marginTop: "1.5rem",
                width: "100%",
              }}
              onClick={processAndSwitch}
            >
              Process PDF
            </button>

            {/* View in-page */}
            <button
              className="button"
              onClick={openPreview}
              disabled={!previewUrl}
              style={{
                marginTop: ".75rem",
                width: "100%",
                background: "white",
                color: "var(--accent-calm-blue)",
                border: "1px solid var(--border-subtle)",
              }}
              title={previewUrl ? "Preview in page" : "Select a PDF first"}
            >
              View PDF
            </button>

            <div
              style={{
                marginTop: "1rem",
                textAlign: "center",
                fontSize: "0.9rem",
                color: "var(--text-secondary)",
              }}
            >
              {status}
            </div>

            <Link
              to="/"
              className="button"
              style={{ marginTop: "1rem", textDecoration: "none", display: "inline-block" }}
            >
              ← Back
            </Link>
          </div>
        </section>
      ) : (
        <section id="chat-section" style={{ width: "100%", height: "100%" }}>
          <div className="app-layout">
            <aside className="glass-panel source-panel">
              <header className="panel-header">
                <Link
                  to="/"
                  className="logo"
                  style={{ display: "flex", alignItems: "center", gap: ".5rem", textDecoration: "none", color: "inherit" }}
                >
                  <img src="/static/logo.png" alt="ClearChartAI" style={{ height: 32, width: "auto" }} />
                  <span>ClearChartAI</span>
                </Link>
                <button className="pdf-action-btn" onClick={openPreview} disabled={!previewUrl}>
                  View PDF
                </button>
              </header>
              <main className="panel-body" id="source-list">
                {sources.map((name, i) => (
                  <div className="source-pill" key={i}>
                    {previewUrl ? (
                      <a href="#view" onClick={(e) => { e.preventDefault(); openPreview(); }}>
                        📄 {name}
                      </a>
                    ) : (
                      <span>📄 {name}</span>
                    )}
                  </div>
                ))}
              </main>
            </aside>

            <main className="glass-panel chat-panel">
              <div ref={chatBodyRef} className="panel-body" id="chat-window">
                {messages.map((m, idx) => (
                  <div
                    key={idx}
                    className={`chat-message ${m.sender === "You" ? "user-message" : "bot-message"} ${m.typing ? "typing" : ""}`}
                  >
                    <div
                      className="message-bubble"
                      dangerouslySetInnerHTML={{ __html: (m.text || "").replace(/\n/g, "<br>") }}
                    />
                  </div>
                ))}
              </div>
              <footer className="panel-footer">
                <div className="input-wrapper">
                  <textarea
                    id="question-input"
                    placeholder="Ask Clari"
                    rows={1}
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        sendQuestion();
                      }
                    }}
                  />
                  <button id="send-button" title="Send Message" onClick={sendQuestion} aria-label="Send">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                      <path d="M22 2L11 13" stroke="#4EC7C2" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                      <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="#4EC7C2" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </button>
                </div>
              </footer>
            </main>
          </div>
        </section>
      )}

      {/* ---------- PDF Modal (in-page preview) ---------- */}
      {showPreview && previewUrl && (
        <div className="pdf-overlay" onClick={closePreview} role="dialog" aria-modal="true" aria-label="PDF preview">
          <div className="pdf-modal" onClick={(e) => e.stopPropagation()}>
            <div className="pdf-modal-header">
              <div className="pdf-modal-title">
                {selectedFile ? selectedFile.name : "Document"}
              </div>
              <div className="pdf-modal-actions">
                <a className="pdf-action-btn" href={previewUrl} target="_blank" rel="noopener noreferrer">
                  Open in new tab
                </a>
                <a className="pdf-action-btn" href={previewUrl} download>
                  Download
                </a>
                <button className="pdf-action-btn" onClick={closePreview}>Close</button>
              </div>
            </div>
            <iframe className="pdf-frame" src={previewUrl} title="PDF preview" />
          </div>
        </div>
      )}
    </div>
  );
}