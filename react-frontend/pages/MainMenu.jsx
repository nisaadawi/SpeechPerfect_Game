import React from "react";
import { useNavigate } from "react-router-dom";
import backgroundVideo from "../assets/meeting.mp4";

function MainMenu() {
  const navigate = useNavigate();

  let user = null;
  try {
    user = JSON.parse(localStorage.getItem("user") || "null");
  } catch (error) {
    user = null;
  }

  const handleStart = () => {
    navigate("/game/topic");
  };

  const handleDashboard = () => {
    navigate("/dashboard");
  };

  const handleHelp = () => {
    navigate("/help");
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        width: "100vw",
        margin: 0,
        position: "relative",
        overflow: "hidden",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "#fefefe",
        boxSizing: "border-box",
      }}
    >
      <video
        src={backgroundVideo}
        autoPlay
        loop
        muted
        playsInline
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
          objectFit: "cover",
        }}
      />

      <div
        style={{
          position: "relative",
          zIndex: 1,
          width: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          padding: "3rem 1.5rem",
          boxSizing: "border-box",
        }}
      >
        <div
          style={{
            maxWidth: "520px",
            width: "100%",
            backgroundColor: "rgba(15, 15, 26, 0.82)",
            borderRadius: "20px",
            padding: "2.5rem",
            boxShadow: "0 30px 60px rgba(0, 0, 0, 0.35)",
            textAlign: "center",
            border: "1px solid rgba(255, 255, 255, 0.05)",
          }}
        >
        <h1
          style={{
            fontSize: "2.8rem",
            marginBottom: "0.75rem",
            letterSpacing: "0.08em",
            textTransform: "uppercase",
          }}
        >
          SpeechPerfect
        </h1>
        <p
          style={{
            fontSize: "1rem",
            opacity: 0.75,
            marginBottom: "2.5rem",
          }}
        >
          {user ? `Welcome back, ${user.username}!` : "Welcome to SpeechPerfect."}
        </p>

          <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
            <button
              onClick={handleStart}
              style={buttonStyle({ primary: true })}
            >
              Start
            </button>
            <button onClick={handleDashboard} style={buttonStyle({})}>
              Dashboard
            </button>
            <button onClick={handleHelp} style={buttonStyle({})}>
              Help
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function buttonStyle({ primary = false }) {
  return {
    padding: "0.95rem 1.25rem",
    borderRadius: "12px",
    border: "none",
    fontSize: "1.15rem",
    fontWeight: 600,
    letterSpacing: "0.03em",
    cursor: "pointer",
    color: primary ? "#0b0b16" : "#fefefe",
    backgroundColor: primary ? "#ffde59" : "rgba(255, 255, 255, 0.08)",
    transition: "transform 0.2s ease, box-shadow 0.3s ease, background-color 0.3s ease",
    boxShadow: primary
      ? "0 20px 30px rgba(255, 222, 89, 0.35)"
      : "0 18px 28px rgba(0, 0, 0, 0.25)",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
  };
}

export default MainMenu;

