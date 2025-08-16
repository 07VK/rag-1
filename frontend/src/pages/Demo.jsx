import React, { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

export default function Demo() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [status, setStatus] = useState("");
  const [showChat, setShowChat] = useState(false);
  const [sources, setSources] = useState([]);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [sessionId, setSessionId] = useState(null); // <-- FIX: Add state for session ID

  const chatBodyRef = useRef(null);

  useEffect(() => {
    if (chatBodyRef.current) chatBodyRef.current.scrollTop = chatBodyRef.current.scrollHeight;
  }, [messages]);

  async function processAndSwitch() {
    if (!selectedFile) { setStatus("Please select a PDF file first."); return; }
    setStatus("Processing document...");
    const form = new FormData();
    form.append("file", selectedFile);
    try {
      const res = await fetch("/upload-pdf", { method: "POST", body: form });
      const json = await res.json();
      if (res.ok && json.status === "success") {
        setSessionId(json.session_id); // <-- FIX: Store the session ID from the response
        setStatus("Success! Starting chat...");
        setShowChat(true);
        setMessages([{ sender: "Bot", text: "The document has been processed. You can now ask questions." }]);
        setSources([selectedFile.name]);
      } else {
        throw new Error(json.message || "Failed to process PDF.");
      }
    } catch (e) {
      setStatus(`Error: ${e.message}`);
    }
  }

  async function sendQuestion() {
    const q = question.trim();
    if (!q) return;
    setMessages((m) => [...m, { sender: "You", text: q }, { sender: "Bot", text: "", typing: true }]);
    setQuestion("");
    try {
      const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // --- FIX: Include the session_id in the request body ---
        body: JSON.stringify({ question: q, session_id: sessionId }),
      });
      const json = await res.json();
      setMessages((m) => {
        const updated = [...m];
        updated[updated.length - 1] = { sender: "Bot", text: res.ok ? json.answer || "" : json.answer || "An error occurred." };
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
              <img src="/static/logo.png" alt="ClearChartAI" className="logo-image" style={{ width: "4rem", height: "auto", marginLeft: 0 }} />
              <div>
                <h1 className="logo-text" style={{ margin: 0 }}>ClearChartAI</h1>
                <p className="logo-subtitle" style={{ marginTop: 0 }}>Understand Your Health. Own Your Future</p>
              </div>
            </Link>
            <p>Begin by uploading a document to activate the workspace.</p>
            <label className="upload-area" htmlFor="pdf-file">
              <input id="pdf-file" type="file" accept=".pdf" style={{ display: "none" }}
                     onChange={(e) => e.target.files && setSelectedFile(e.target.files[0])}/>
              <span style={{ fontSize: "1.1rem", fontWeight: 500 }}>
                {selectedFile ? selectedFile.name : "Click or Drag & Drop a PDF"}
              </span>
            </label>
            <button className="button" style={{ background: "linear-gradient(to right, var(--medical-teal), var(--medical-teal-dark))", marginTop: "1.5rem", width: "100%" }} onClick={processAndSwitch}>
              Process PDF
            </button>
            <div style={{ marginTop: "1rem", textAlign: "center", fontSize: "0.9rem", color: "var(--text-secondary)" }}>{status}</div>
            <Link to="/" className="button" style={{ marginTop: "1rem", textDecoration: "none", display: "inline-block" }}>
              ← Back
            </Link>
          </div>
        </section>
      ) : (
        <section id="chat-section" style={{ width: "100%", height: "100%" }}>
          <div className="app-layout">
            <aside className="glass-panel source-panel">
              <header className="panel-header">
                <Link to="/" className="logo" style={{ display: "flex", alignItems: "center", gap: ".5rem", textDecoration: "none", color: "inherit" }}>
                  <img src="/static/logo.png" alt="ClearChartAI" style={{ height: 32, width: "auto" }} />
                  <span>ClearChartAI</span>
                </Link>
              </header>
              <main className="panel-body" id="source-list">
                {sources.map((s, i) => (
                  <div className="source-pill" key={i}><span>📄 {s}</span></div>
                ))}
              </main>
            </aside>
            <main className="glass-panel chat-panel">
              <div ref={chatBodyRef} className="panel-body" id="chat-window">
                {messages.map((m, idx) => (
                  <div key={idx} className={`chat-message ${m.sender === "You" ? "user-message" : "bot-message"} ${m.typing ? "typing" : ""}`}>
                    <div className="message-bubble" dangerouslySetInnerHTML={{ __html: (m.text || "").replace(/\n/g, "<br>") }} />
                  </div>
                ))}
              </div>
              <footer className="panel-footer">
                <div className="input-wrapper">
                  <textarea id="question-input" placeholder="Ask Clari" rows={1} value={question}
                            onChange={(e) => setQuestion(e.target.value)}
                            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendQuestion(); }}} />
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
    </div>
  );
}