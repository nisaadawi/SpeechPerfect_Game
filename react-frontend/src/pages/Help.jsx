import React from "react";

function Help() {
  return (
    <div
      style={{
        minHeight: "100vh",
        width: "100vw",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: "#0d1626",
        color: "#f5f7fb",
        gap: "1.5rem",
        padding: "2.5rem 1.5rem",
        textAlign: "center",
        boxSizing: "border-box",
      }}
    >
      <h1 style={{ fontSize: "2.3rem", letterSpacing: "0.06em" }}>Help & Support</h1>
      <p style={{ maxWidth: "540px", lineHeight: 1.6, opacity: 0.8 }}>
        Need assistance? Detailed guides will appear here soon. For now, ensure your camera
        and microphone permissions are enabled for the best SpeechPerfect experience.
      </p>
      <p style={{ fontSize: "0.95rem", opacity: 0.6 }}>
        Contact support: <a href="mailto:support@speechperfect.com" style={{ color: "#ffde59" }}>support@speechperfect.com</a>
      </p>
    </div>
  );
}

export default Help;

