import React, { useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import TeamCard from "../components/TeamCard";

export default function About() {
  const containerRef = useRef(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const sections = Array.from(el.querySelectorAll(".about-section"));
    const observer = new IntersectionObserver(
      (entries) => entries.forEach((entry) => entry.isIntersecting && entry.target.classList.add("visible")),
      { threshold: 0.2 }
    );
    sections.forEach((s) => observer.observe(s));
    return () => observer.disconnect();
  }, []);

  return (
    <div id="about-page" ref={containerRef}>
      <header className="about-header">
        <a href="/" onClick={(e) => e.preventDefault()} className="logo-container" aria-label="ClearChartAI">
          <img src="/static/logo.png" alt="ClearChartAI" className="logo-image" />
          <div>
            <span className="logo-text">ClearChartAI</span>
            <p className="logo-subtitle">Understand Your Health. Own Your Future</p>
          </div>
        </a>
      </header>

      <section className="about-section" id="about-hero">
        <h1 className="hero-title">Clarity in Complexity.</h1>
        <p className="hero-subtitle">
          Healthcare buries you under portals, PDFs, and jargon, ClearChartAI cuts through it all.
          Our Synapse engine pulls records from every hospital, lab, and clinic, stitching them into one place.
          It translates dense medical data into clear, actionable insights you can understand in seconds, not hours.
          No mystery, just your health made simple, secure, and truly yours. We will let you know what to ask next, so your limited time with the doctor counts.
        </p>

        <div className="heartbeat-container" role="img" aria-label="EKG heartbeat animation">
          <div className="heartbeat-line" />
          <div className="heart" />
        </div>

        <div className="features-grid">
          <div className="feature-card animated-border">
            <div className="text-sm font-medium">Unified Records</div>
            <div className="text-sm text-slate-600">All providers, one place.</div>
          </div>
          <div className="feature-card animated-border">
            <div className="text-sm font-medium">Plain English</div>
            <div className="text-sm text-slate-600">No jargon, no guesswork.</div>
          </div>
          <div className="feature-card animated-border">
            <div className="text-sm font-medium">Actionable</div>
            <div className="text-sm text-slate-600">Know what to ask next.</div>
          </div>
        </div>
      </section>

      <div className="demo-button-wrapper" style={{ marginBottom: "8rem" }}>
        <Link to="/demo" className="demo-button" aria-label="Try our Demo AI Synapse">
          Try our Demo AI Synapse
        </Link>
      </div>

      <section className="about-section" id="about-features">
        <h2 className="hero-title" style={{ fontSize: "2.8rem" }}>The Synapse Advantage</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="icon">
              {/* icon svg */}
            </div>
            <h3>Instant Understanding</h3>
            <p>Our AI structures complex medical jargon, labs, and notes into a simple, chronological summary.</p>
          </div>
          <div className="feature-card">
            <div className="icon">{/* icon */}</div>
            <h3>Secure &amp; Compliant</h3>
            <p>HIPAA-oriented; encrypted and anonymized.</p>
          </div>
          <div className="feature-card">
            <div className="icon">{/* icon */}</div>
            <h3>Actionable Insights</h3>
            <p>Ask questions and get answers sourced from the document.</p>
          </div>
        </div>
      </section>

      <section className="about-section" id="about-team">
        <h2 className="hero-title" style={{ fontSize: "2.8rem" }}>The Team</h2>
        <div className="team-grid">
          <TeamCard img="/static/nicholas-davis.jpg" name="Nicholas Davis" title="Founder & CEO, AGACNP, BSN"
                    bio="Over 8 years in the medical field, driving the mission with a deep understanding of patient care." />
          <TeamCard img="/static/dhruv-suraj.jpg" name="Dhruv Suraj" title="Lead AI Engineer"
                    bio="Leads the stack to keep the platform powerful, intuitive, and reliable." />
          <TeamCard img="/static/vishnu-koraganji.png" name="Vishnu Koraganji" title="Sr Full-Stack AI Engineer"
                    bio="Turns complex medical jargon into clear language and designs UI." />
        </div>
      </section>
    </div>
  );
}
