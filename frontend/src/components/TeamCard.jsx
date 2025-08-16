import React from "react";

export default function TeamCard({ img, name, title, bio }) {
  return (
    <div className="team-card-container">
      <div className="team-card">
        <div className="card-image-content">
          <img src={img} alt={name} className="card-image" />
        </div>
        <div className="card-info">
          <div className="card-front-content">
            <h3 className="card-name">{name}</h3>
            <p className="card-title">{title}</p>
          </div>
          <div className="card-back-content">
            <p className="card-bio">{bio}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
